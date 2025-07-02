import numpy as np
import sys
import os
from typing import Dict, Any, Optional, Tuple, List
import re
from difflib import SequenceMatcher

def load_gguf_tensor(gguf_path: str, tensor_name: str) -> Optional[np.ndarray]:
    """Load a specific tensor from GGUF file."""
    try:
        import gguf
        reader = gguf.GGUFReader(gguf_path)
        
        for tensor in reader.tensors:
            if tensor.name == tensor_name:
                dtype_name = get_dtype_name(tensor.tensor_type)
                
                if dtype_name in ['F32', 'F16', 'F64']:
                    tensor_data = tensor.data
                    if tensor_data is not None:
                        if dtype_name == 'F32':
                            np_data = np.frombuffer(tensor_data, dtype=np.float32)
                        elif dtype_name == 'F16':
                            np_data = np.frombuffer(tensor_data, dtype=np.float16)
                        elif dtype_name == 'F64':
                            np_data = np.frombuffer(tensor_data, dtype=np.float64)
                        
                        return np_data.reshape(tensor.shape)
                else:
                    print(f"Warning: Tensor {tensor_name} has quantized type {dtype_name}, cannot compare directly")
                    return None
        
        print(f"Tensor {tensor_name} not found in GGUF file")
        return None
        
    except ImportError:
        print("gguf library not available")
        return None
    except Exception as e:
        print(f"Error loading GGUF tensor: {e}")
        return None

def load_safetensors_tensor(safetensors_path: str, tensor_name: str) -> Optional[np.ndarray]:
    """Load a specific tensor from SafeTensors file."""
    try:
        import safetensors
        from safetensors import safe_open
        
        with safe_open(safetensors_path, framework="numpy") as f:
            if tensor_name in f.keys():
                return f.get_tensor(tensor_name)
            else:
                print(f"Tensor {tensor_name} not found in SafeTensors file")
                return None
                
    except ImportError:
        print("safetensors library not available")
        print("Install with: pip install safetensors")
        return None
    except Exception as e:
        print(f"Error loading SafeTensors tensor: {e}")
        return None

def get_dtype_name(tensor_type) -> str:
    """Get human readable data type name."""
    type_map = {
        0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 
        8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 
        13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
        18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
        23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    }
    
    if hasattr(tensor_type, 'value'):
        type_val = tensor_type.value
    else:
        type_val = int(tensor_type)
        
    return type_map.get(type_val, f"UNKNOWN_{type_val}")

def normalize_tensor_name(name: str) -> str:
    """Normalize tensor name for comparison."""
    # Remove common prefixes/suffixes
    normalized = name.lower()
    
    # Remove model prefix patterns
    normalized = re.sub(r'^model\.', '', normalized)
    normalized = re.sub(r'^transformer\.', '', normalized)
    normalized = re.sub(r'^language_model\.', '', normalized)
    
    # Standardize weight/bias naming
    normalized = re.sub(r'\.weight$', '.w', normalized)
    normalized = re.sub(r'\.bias$', '.b', normalized)
    
    # Standardize layer numbering
    normalized = re.sub(r'layers\.(\d+)\.', r'layer.\1.', normalized)
    normalized = re.sub(r'h\.(\d+)\.', r'layer.\1.', normalized)
    
    # Standardize attention naming
    normalized = re.sub(r'\.self_attn\.', '.attn.', normalized)
    normalized = re.sub(r'\.attention\.', '.attn.', normalized)
    
    return normalized

def find_tensor_mappings(gguf_names: List[str], safe_names: List[str]) -> Dict[str, List[str]]:
    """Find potential mappings between GGUF and SafeTensors tensor names."""
    mappings = {}
    
    # Normalize all names for comparison
    gguf_normalized = {name: normalize_tensor_name(name) for name in gguf_names}
    safe_normalized = {name: normalize_tensor_name(name) for name in safe_names}
    
    # Try exact matches on normalized names first
    for gguf_name in gguf_names:
        gguf_norm = gguf_normalized[gguf_name]
        exact_matches = [safe_name for safe_name, safe_norm in safe_normalized.items() 
                        if safe_norm == gguf_norm]
        if exact_matches:
            mappings[gguf_name] = exact_matches
            continue
    
    # Try fuzzy matching for remaining tensors
    for gguf_name in gguf_names:
        if gguf_name in mappings:
            continue
            
        gguf_norm = gguf_normalized[gguf_name]
        candidates = []
        
        for safe_name in safe_names:
            safe_norm = safe_normalized[safe_name]
            
            # Calculate similarity
            similarity = SequenceMatcher(None, gguf_norm, safe_norm).ratio()
            
            # Also check if they have the same "core" parts
            gguf_parts = set(gguf_norm.split('.'))
            safe_parts = set(safe_norm.split('.'))
            common_parts = len(gguf_parts.intersection(safe_parts))
            total_parts = len(gguf_parts.union(safe_parts))
            part_similarity = common_parts / total_parts if total_parts > 0 else 0
            
            # Combined score
            score = (similarity * 0.7) + (part_similarity * 0.3)
            
            if score > 0.6:  # Threshold for potential matches
                candidates.append((safe_name, score))
        
        # Sort by score and take top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        if candidates:
            mappings[gguf_name] = [name for name, score in candidates[:3]]  # Top 3 candidates
    
    return mappings

def get_all_tensor_info(gguf_path: str, safetensors_path: str) -> Tuple[Dict, Dict, Dict]:
    """Get comprehensive tensor information from both files."""
    gguf_info = {}
    safe_info = {}
    
    try:
        import gguf
        import safetensors
        from safetensors import safe_open
        
        # Get GGUF tensor info
        gguf_reader = gguf.GGUFReader(gguf_path)
        for tensor in gguf_reader.tensors:
            gguf_info[tensor.name] = {
                'shape': tensor.shape,
                'dtype': get_dtype_name(tensor.tensor_type),
                'n_elements': tensor.n_elements
            }
        
        # Get SafeTensors tensor info
        with safe_open(safetensors_path, framework="numpy") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                safe_info[name] = {
                    'shape': tensor.shape,
                    'dtype': str(tensor.dtype),
                    'n_elements': tensor.size
                }
        
        # Find mappings
        mappings = find_tensor_mappings(list(gguf_info.keys()), list(safe_info.keys()))
        
        return gguf_info, safe_info, mappings
        
    except Exception as e:
        print(f"Error getting tensor info: {e}")
        return {}, {}, {}

def analyze_raw_bytes(gguf_data: bytes, safetensors_data: bytes, num_samples: int = 20) -> None:
    """Analyze raw byte differences between GGUF and SafeTensors data."""
    print("\nRaw Byte Analysis:")
    print("-" * 50)
    
    min_len = min(len(gguf_data), len(safetensors_data))
    if min_len < 4:
        print("Not enough data for byte analysis")
        return
    
    sample_indices = np.linspace(0, min_len-4, min(num_samples, min_len//4), dtype=int)
    
    for i, idx in enumerate(sample_indices):
        if idx + 4 <= min_len:
            gguf_bytes = gguf_data[idx:idx+4]
            safe_bytes = safetensors_data[idx:idx+4]
            
            # Interpret as float32
            gguf_float = np.frombuffer(gguf_bytes, dtype=np.float32)[0]
            safe_float = np.frombuffer(safe_bytes, dtype=np.float32)[0]
            
            # Show hex representation
            gguf_hex = ' '.join(f'{b:02x}' for b in gguf_bytes)
            safe_hex = ' '.join(f'{b:02x}' for b in safe_bytes)
            
            print(f"Sample {i+1:2d} (offset {idx:6d}):")
            print(f"  GGUF:        {gguf_float:12.6f} | Hex: {gguf_hex}")
            print(f"  SafeTensors: {safe_float:12.6f} | Hex: {safe_hex}")
            print(f"  Difference:  {gguf_float - safe_float:12.6f}")
            
            if gguf_bytes != safe_bytes:
                print(f"  Byte diff:   {''.join('*' if a != b else '.' for a, b in zip(gguf_bytes, safe_bytes))}")
            print()

def compare_tensors(gguf_tensor: np.ndarray, safe_tensor: np.ndarray, gguf_name: str, safe_name: str) -> None:
    """Compare two tensors and analyze differences."""
    print(f"\nComparing tensors:")
    print(f"  GGUF:        {gguf_name}")
    print(f"  SafeTensors: {safe_name}")
    print("=" * 80)
    
    # Basic info
    print(f"GGUF shape:        {gguf_tensor.shape}")
    print(f"SafeTensors shape: {safe_tensor.shape}")
    print(f"GGUF dtype:        {gguf_tensor.dtype}")
    print(f"SafeTensors dtype: {safe_tensor.dtype}")
    
    if gguf_tensor.shape != safe_tensor.shape:
        print("ERROR: Shape mismatch!")
        return
    
    # Convert to same dtype for comparison
    if gguf_tensor.dtype != safe_tensor.dtype:
        print(f"Converting to common dtype: float32")
        gguf_tensor = gguf_tensor.astype(np.float32)
        safe_tensor = safe_tensor.astype(np.float32)
    
    # Flatten for easier analysis
    gguf_flat = gguf_tensor.flatten()
    safe_flat = safe_tensor.flatten()
    
    # Statistical analysis
    print(f"\nStatistical Analysis:")
    print(f"GGUF:        min={np.min(gguf_flat):.6f}, max={np.max(gguf_flat):.6f}, mean={np.mean(gguf_flat):.6f}")
    print(f"SafeTensors: min={np.min(safe_flat):.6f}, max={np.max(safe_flat):.6f}, mean={np.mean(safe_flat):.6f}")
    
    # Difference analysis
    diff = gguf_flat - safe_flat
    print(f"\nDifference Analysis:")
    print(f"Min difference:  {np.min(diff):.6f}")
    print(f"Max difference:  {np.max(diff):.6f}")
    print(f"Mean difference: {np.mean(diff):.6f}")
    print(f"Std difference:  {np.std(diff):.6f}")
    
    # Check for patterns
    unique_diffs = np.unique(np.round(diff, 6))
    print(f"Unique differences (rounded): {len(unique_diffs)}")
    if len(unique_diffs) <= 10:
        print(f"Unique diff values: {unique_diffs}")
    
    # Check for consistent offset
    is_consistent_offset = np.allclose(diff, diff[0], atol=1e-6)
    if is_consistent_offset:
        print(f"✓ Consistent offset detected: {diff[0]:.6f}")
    else:
        print("✗ No consistent offset")
    
    # Sample comparisons
    print(f"\nSample Value Comparisons (first 10):")
    for i in range(min(10, len(gguf_flat))):
        print(f"  [{i:2d}] GGUF: {gguf_flat[i]:12.6f}, Safe: {safe_flat[i]:12.6f}, Diff: {diff[i]:12.6f}")
    
    # Check if it's exactly +1
    if np.allclose(diff, 1.0, atol=1e-6):
        print("\n🚨 CONFIRMED: GGUF values are consistently +1 compared to SafeTensors!")
    elif np.allclose(diff, -1.0, atol=1e-6):
        print("\n🚨 CONFIRMED: GGUF values are consistently -1 compared to SafeTensors!")
    
    # Raw byte analysis for first few values
    if len(gguf_flat) > 0:
        gguf_bytes = gguf_tensor.astype(gguf_tensor.dtype).tobytes()
        safe_bytes = safe_tensor.astype(safe_tensor.dtype).tobytes()
        analyze_raw_bytes(gguf_bytes, safe_bytes, 5)

def main():
    """Main comparison function with improved name matching."""
    if len(sys.argv) < 3:
        print("Usage: python compare_gguf_safetensors.py <gguf_file> <safetensors_file> [gguf_tensor_name] [safe_tensor_name]")
        print("\nModes:")
        print("1. Auto-discovery: python compare_gguf_safetensors.py model.gguf model.safetensors")
        print("2. Specific mapping: python compare_gguf_safetensors.py model.gguf model.safetensors \"blk.0.attn_q.weight\" \"model.layers.0.self_attn.q_proj.weight\"")
        print("\nThe tool will automatically find tensor name mappings between formats.")
        sys.exit(1)
    
    gguf_path = sys.argv[1]
    safetensors_path = sys.argv[2]
    
    if not os.path.exists(gguf_path):
        print(f"Error: GGUF file '{gguf_path}' not found")
        sys.exit(1)
    
    if not os.path.exists(safetensors_path):
        print(f"Error: SafeTensors file '{safetensors_path}' not found")
        sys.exit(1)
    
    print("Analyzing tensor names and finding mappings...")
    gguf_info, safe_info, mappings = get_all_tensor_info(gguf_path, safetensors_path)
    
    print(f"\nFile Summary:")
    print(f"GGUF tensors: {len(gguf_info)}")
    print(f"SafeTensors tensors: {len(safe_info)}")
    print(f"Potential mappings found: {len(mappings)}")
    
    if len(sys.argv) >= 5:
        # Manual tensor specification
        gguf_tensor_name = sys.argv[3]
        safe_tensor_name = sys.argv[4]
        
        print(f"\nManual comparison specified:")
        print(f"GGUF: {gguf_tensor_name}")
        print(f"SafeTensors: {safe_tensor_name}")
        
        gguf_tensor = load_gguf_tensor(gguf_path, gguf_tensor_name)
        safe_tensor = load_safetensors_tensor(safetensors_path, safe_tensor_name)
        
        if gguf_tensor is not None and safe_tensor is not None:
            compare_tensors(gguf_tensor, safe_tensor, gguf_tensor_name, safe_tensor_name)
        
    else:
        # Auto-discovery mode
        print(f"\nTensor Name Mappings (Top matches):")
        print("-" * 80)
        
        successful_comparisons = 0
        for gguf_name, safe_candidates in list(mappings.items())[:10]:  # Show first 10 mappings
            print(f"\nGGUF: {gguf_name}")
            for safe_name in safe_candidates[:1]:  # Take best match
                print(f"  -> SafeTensors: {safe_name}")
                
                # Check if shapes match
                if gguf_name in gguf_info and safe_name in safe_info:
                    gguf_shape = gguf_info[gguf_name]['shape']
                    safe_shape = safe_info[safe_name]['shape']
                    
                    if gguf_shape == safe_shape:
                        print(f"     Shapes match: {gguf_shape}")
                        
                        # Only compare first few with actual data
                        if successful_comparisons < 3:
                            gguf_tensor = load_gguf_tensor(gguf_path, gguf_name)
                            safe_tensor = load_safetensors_tensor(safetensors_path, safe_name)
                            
                            if gguf_tensor is not None and safe_tensor is not None:
                                compare_tensors(gguf_tensor, safe_tensor, gguf_name, safe_name)
                                successful_comparisons += 1
                                print("\n" + "="*80)
                    else:
                        print(f"     Shape mismatch: GGUF={gguf_shape} vs Safe={safe_shape}")
        
        if successful_comparisons == 0:
            print("\nNo matching tensors found for comparison.")
            print("Try manual specification with tensor names.")

if __name__ == "__main__":
    main()
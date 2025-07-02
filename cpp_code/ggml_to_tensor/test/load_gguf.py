import numpy as np
from typing import List, Any
import sys
import os
from safetensors.numpy import load_file

# Modify this mapping as needed
tensor_name_map = {
    # GGUF name           : Safetensors name
    "model.embed.weight" : "model.embed_tokens.weight",
    "model.lm_head.weight" : "lm_head.weight",
    # ...
}

def to_string_value(value: Any) -> str:
    """Convert a value to string representation similar to C++ version."""
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"
    elif isinstance(value, (int, np.integer)):
        return str(value)
    elif isinstance(value, bool):
        return "true" if value else "false"
    else:
        return str(value)

def to_string_snipped(tensor_data: np.ndarray, n: int = 8) -> str:
    """
    Convert tensor data to string representation similar to the C++ snippet.
    
    Args:
        tensor_data: NumPy array containing tensor data
        n: Number of elements to show from start and end (default: 8)
    
    Returns:
        String representation of tensor data
    """
    # Flatten the tensor to work with 1D indexing
    flat_data = tensor_data.flatten()
    nitems = len(flat_data)
    
    if n == 0 or (n * 2) >= nitems:
        # Show all elements
        values = [to_string_value(flat_data[i]) for i in range(nitems)]
        return "[" + ", ".join(values) + "]"
    else:
        # Show first N and last N elements with ellipsis
        head_end = min(n, nitems)
        tail_start = max(nitems - n, head_end)
        
        head_values = [to_string_value(flat_data[i]) for i in range(head_end)]
        tail_values = [to_string_value(flat_data[i]) for i in range(tail_start, nitems)]
        
        result = "[" + ", ".join(head_values) + ", ..., " + ", ".join(tail_values) + "]"
        return result

def format_shape(shape: List[int]) -> str:
    """Format tensor shape for display."""
    return "(" + ", ".join(map(str, shape)) + ")"

def format_size(num_bytes: int) -> str:
    """Format size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"

def get_dtype_name(tensor_type) -> str:
    """Get human readable data type name."""
    # Map from gguf tensor types to readable names
    type_map = {
        0: "F32",
        1: "F16", 
        2: "Q4_0",
        3: "Q4_1",
        6: "Q5_0",
        7: "Q5_1", 
        8: "Q8_0",
        9: "Q8_1",
        10: "Q2_K",
        11: "Q3_K",
        12: "Q4_K",
        13: "Q5_K",
        14: "Q6_K",
        15: "Q8_K",
        16: "IQ2_XXS",
        17: "IQ2_XS",
        18: "IQ3_XXS",
        19: "IQ1_S",
        20: "IQ4_NL",
        21: "IQ3_S",
        22: "IQ2_S",
        23: "IQ4_XS",
        24: "I8",
        25: "I16",
        26: "I32",
        27: "I64",
        28: "F64",
        29: "IQ1_M",
    }
    
    if hasattr(tensor_type, 'value'):
        type_val = tensor_type.value
    else:
        type_val = int(tensor_type)
        
    return type_map.get(type_val, f"UNKNOWN_{type_val}")

def print_gguf_tensors(gguf_path: str, preview_elements: int = 8, max_tensors: int = None):
    """
    Print all tensor data from a GGUF file using gguf library.
    
    Args:
        gguf_path: Path to the GGUF file
        preview_elements: Number of elements to preview from start/end of each tensor
        max_tensors: Maximum number of tensors to display (None for all)
    """
    if not os.path.exists(gguf_path):
        print(f"Error: File '{gguf_path}' not found")
        return
    
    try:
        import gguf
        print(f"Loading GGUF file: {gguf_path}")
        print("=" * 80)
        
        # Create GGUF reader
        reader = gguf.GGUFReader(gguf_path)
        
        # Print file metadata
        print("GGUF File Metadata:")
        print("-" * 40)
        
        metadata_items = list(reader.fields.items())
        for key, field in metadata_items[:20]:  # Limit metadata display
            if hasattr(field, 'types') and hasattr(field, 'data'):
                print(f"{key}: {field.data}")
            else:
                print(f"{key}: {field}")
        
        if len(metadata_items) > 20:
            print(f"... and {len(metadata_items) - 20} more metadata entries")
        
        print(f"\nTotal tensors in file: {len(reader.tensors)}")
        print("=" * 80)
        
        # Limit display if requested
        tensors_to_show = reader.tensors
        if max_tensors and len(reader.tensors) > max_tensors:
            tensors_to_show = reader.tensors[:max_tensors]
            print(f"Showing first {max_tensors} tensors out of {len(reader.tensors)} total")
            print("-" * 80)
        
        # Iterate through tensors
        for i, tensor in enumerate(tensors_to_show):
            print(f"\nTensor {i + 1}: {tensor.name}")
            print(f"  Shape: {format_shape(tensor.shape)}")
            print(f"  Data type: {get_dtype_name(tensor.tensor_type)}")
            print(f"  Elements: {tensor.n_elements:,}")
            print(f"  Size: {format_size(tensor.n_bytes)}")
            
            # Try to read tensor data for preview
            try:
                # For quantized types, we can't easily convert to numpy arrays
                dtype_name = get_dtype_name(tensor.tensor_type)
                
                if dtype_name in ['F32', 'F16', 'F64', 'I8', 'I16', 'I32', 'I64']:
                    # Regular numeric types we can preview
                    tensor_data = tensor.data
                    if tensor_data is not None and len(tensor_data) > 0:
                        # Convert to numpy array based on type
                        if dtype_name == 'F32':
                            np_data = np.frombuffer(tensor_data, dtype=np.float32)
                            
                            # Convert memmap to raw bytes (if needed)
                            if isinstance(tensor_data, np.memmap):
                                raw_bytes = tensor_data.tobytes()
                            else:
                                raw_bytes = tensor_data  # Already bytes
                            
                            # Optional: Print raw float bits for debugging
                            print(f"  Raw float32 values and bit patterns:")
                            preview_count = min(preview_elements, len(np_data))
                            for i in range(preview_count):
                                b = raw_bytes[i*4:(i+1)*4]
                                float_val = np_data[i]
                                hex_repr = b.hex()
                                ieee_bits = int.from_bytes(b, byteorder='little')
                                print(f"    [{i:02d}] {float_val:.6f}  (hex: {hex_repr}, IEEE754: 0x{ieee_bits:08x})")

                            # Reshape to original shape for clean preview
                            np_data = np_data.reshape(tensor.shape)
                            print(f"  Data: {to_string_snipped(np_data, preview_elements)}")                        
                        elif dtype_name == 'F16':
                            np_data = np.frombuffer(tensor_data, dtype=np.float16)
                        elif dtype_name == 'F64':
                            np_data = np.frombuffer(tensor_data, dtype=np.float64)
                        elif dtype_name == 'I8':
                            np_data = np.frombuffer(tensor_data, dtype=np.int8)
                        elif dtype_name == 'I16':
                            np_data = np.frombuffer(tensor_data, dtype=np.int16)
                        elif dtype_name == 'I32':
                            np_data = np.frombuffer(tensor_data, dtype=np.int32)
                        elif dtype_name == 'I64':
                            np_data = np.frombuffer(tensor_data, dtype=np.int64)
                        else:
                            np_data = None
                        
                        if np_data is not None:
                            # Reshape to original shape
                            np_data = np_data.reshape(tensor.shape)
                            print(f"  Data: {to_string_snipped(np_data, preview_elements)}")
                        else:
                            print(f"  Data: [Cannot preview {dtype_name} data]")
                    else:
                        print(f"  Data: [No data available]")
                else:
                    # Quantized types - show raw bytes preview
                    tensor_data = tensor.data
                    if tensor_data is not None and len(tensor_data) > 0:
                        # Show first few bytes as hex
                        preview_bytes = min(32, len(tensor_data))
                        hex_preview = ' '.join(f'{b:02x}' for b in tensor_data[:preview_bytes])
                        if len(tensor_data) > preview_bytes:
                            hex_preview += " ..."
                        print(f"  Data: [Quantized {dtype_name}] Raw bytes: {hex_preview}")
                    else:
                        print(f"  Data: [No data available]")
                        
            except Exception as e:
                print(f"  Data: [Error reading tensor data: {e}]")
            
            print("-" * 60)
        
        if max_tensors and len(reader.tensors) > max_tensors:
            print(f"\n... {len(reader.tensors) - max_tensors} more tensors not shown")
            print("Use max_tensors=None to show all tensors")
        
    except ImportError:
        print("Error: gguf library not found")
        print("Please install it with: pip install gguf")
        print("\nAlternatively, you can install from source:")
        print("git clone https://github.com/ggerganov/ggml")
        print("cd ggml/gguf-py")
        print("pip install -e .")
        
    except Exception as e:
        print(f"Error reading GGUF file: {e}")
        import traceback
        print("\nFull error traceback:")
        traceback.print_exc()

def main():
    """Main function with command line argument parsing."""
    if len(sys.argv) < 2:
        print("Usage: python gguf_reader.py <path_to_gguf_file> [max_tensors] [preview_elements]")
        print("\nArguments:")
        print("  path_to_gguf_file: Path to the GGUF file to read")
        print("  max_tensors: Maximum number of tensors to display (optional, default: 50)")
        print("  preview_elements: Number of elements to preview from start/end (optional, default: 8)")
        print("\nExamples:")
        print("  python gguf_reader.py model.gguf")
        print("  python gguf_reader.py model.gguf 10")
        print("  python gguf_reader.py model.gguf 10 4")
        sys.exit(1)
    
    gguf_file = sys.argv[1]
    
    # Parse optional arguments
    max_tensors = None  # Default limit to prevent overwhelming output
    if len(sys.argv) > 2:
        try:
            max_tensors = int(sys.argv[2])
            if max_tensors <= 0:
                max_tensors = None
        except ValueError:
            print(f"Warning: Invalid max_tensors value '{sys.argv[2]}', using default 50")
    
    preview_elements = 8  # Default preview size
    if len(sys.argv) > 3:
        try:
            preview_elements = int(sys.argv[3])
            if preview_elements < 0:
                preview_elements = 8
        except ValueError:
            print(f"Warning: Invalid preview_elements value '{sys.argv[3]}', using default 8")
    
    print_gguf_tensors(gguf_file, preview_elements, max_tensors)

if __name__ == "__main__":
    main()
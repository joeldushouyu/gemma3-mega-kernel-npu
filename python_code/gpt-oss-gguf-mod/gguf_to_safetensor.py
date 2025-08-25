#!/usr/bin/env python3
import logging
import sys
from pathlib import Path
import torch

from torch import Tensor
import numpy as np
from safetensors.torch import save_file, load_file
import json
import os
import numpy.typing as npt
logger = logging.getLogger("reader")

# Necessary to load the local gguf package
gguf_packet_dir = str(Path(__file__).parent.parent.parent.joinpath("SubModules/llama.cpp/gguf-py"))
print("gguf_packet_dir:", gguf_packet_dir)
sys.path.insert(0, gguf_packet_dir)

from gguf.constants import GGMLQuantizationType
from gguf.gguf_reader import GGUFReader
from gguf.quants import dequantize
from gguf import ReaderTensor
import gguf
from util import print_tensor_erros, print_numpy_errors
















# def dequantize_e2m1(value: int) -> float:
#     """Dequantizes a 4-bit E2M1 float."""
#     if value < 0 or value > 15:
#         raise ValueError("Input must be a 4-bit integer (0-15)")

#     sign = (value >> 3) & 1
#     exponent = (value >> 1) & 3
#     mantissa = value & 1

#     if exponent == 0:
#         # Subnormal number: (-1)^sign * 2^(-1) * (mantissa / 2)
#         if mantissa == 0:
#             return 0.0 if sign == 0 else -0.0
#         else:
#             return ((-1) ** sign) * (2 ** (-1)) * (mantissa / 2.0)
#     else:
#         # Normal number: (-1)^sign * 2^(exponent - 1) * (1 + mantissa / 2)
#         return ((-1) ** sign) * (2 ** (exponent - 1)) * (1.0 + mantissa / 2.0)

# def dequantize_mxf4_tensor(blocks_tensor: np.ndarray, scales_tensor: np.ndarray, block_size: int = 32) -> np.ndarray:
#     """
#     Dequantizes an MXF4 tensor given the quantized blocks and their scales.
    
#     Args:
#         blocks_tensor: Quantized blocks of shape (32, 2880, 90, 16) containing packed 4-bit values
#         scales_tensor: Scale factors of shape (32, 2880, 90) for each block
#         block_size: Size of each block (default: 32, but actually refers to elements per scale)
    
#     Returns:
#         Dequantized float32 array
#     """
#     # Validate inputs
#     assert blocks_tensor.dtype == np.uint8, f"Expected uint8 blocks, got {blocks_tensor.dtype}"
#     assert not np.isnan(scales_tensor).any(), "Scale tensor contains NaN values"
#     assert not np.isinf(scales_tensor).any(), "Scale tensor contains infinite values"
    
#     # Get shapes
#     batch_size, num_experts, num_blocks, bytes_per_block = blocks_tensor.shape
#     assert scales_tensor.shape == (batch_size, num_experts, num_blocks), \
#         f"Scale shape {scales_tensor.shape} doesn't match expected {(batch_size, num_experts, num_blocks)}"
    
#     # Each byte contains 2 4-bit values, so we get 32 values per block (16 bytes * 2)
#     values_per_block = bytes_per_block * 2
    
#     # Output shape: (batch_size, num_experts, num_blocks * values_per_block)
#     output_shape = (batch_size, num_experts, num_blocks * values_per_block)
#     dequantized = np.zeros(output_shape, dtype=np.float32)
    
#     for b in range(batch_size):
#         for e in range(num_experts):
#             for block_idx in range(num_blocks):
#                 scale = scales_tensor[b, e, block_idx]
#                 block_bytes = blocks_tensor[b, e, block_idx, :]
                
#                 # Dequantize each byte in the block
#                 for byte_idx, byte_val in enumerate(block_bytes):
#                     # Ensure byte_val is in valid range
#                     byte_val = int(byte_val) & 0xFF
                    
#                     # Unpack two 4-bit values from each byte
#                     val1 = byte_val & 0x0F
#                     val2 = (byte_val >> 4) & 0x0F
                    
#                     # Calculate output indices
#                     base_idx = block_idx * values_per_block + byte_idx * 2
#                     idx1 = base_idx
#                     idx2 = base_idx + 1
                    
#                     # Dequantize and scale
#                     if idx1 < output_shape[2]:
#                         dequantized[b, e, idx1] = dequantize_e2m1(val1) * scale
#                     if idx2 < output_shape[2]:
#                         dequantized[b, e, idx2] = dequantize_e2m1(val2) * scale
    
#     # Validate output
#     nan_count = np.isnan(dequantized).sum()
#     inf_count = np.isinf(dequantized).sum()
#     if nan_count > 0:
#         print(f"Warning: Dequantized tensor contains {nan_count} NaN values")
#     if inf_count > 0:
#         print(f"Warning: Dequantized tensor contains {inf_count} infinite values")
    
#     return dequantized


def dequantize_e2m1_vectorized(values: np.ndarray) -> np.ndarray:
    """Vectorized dequantization for an array of 4-bit E2M1 values."""
    
    # 1. Extract sign, exponent, and mantissa for the entire array
    sign = np.bitwise_and(np.right_shift(values, 3), 1)
    exponent = np.bitwise_and(np.right_shift(values, 1), 3)
    mantissa = np.bitwise_and(values, 1)

    # Use a numerically stable way to apply the sign: 1.0 for positive, -1.0 for negative
    sign_mult = 1.0 - 2.0 * sign

    # 2. Calculate results for normal and subnormal cases across the whole array
    # Note: Replicating your original logic with 2**(-1) for subnormals
    subnormal_vals = sign_mult * (0.5) * (mantissa / 2.0)
    normal_vals = sign_mult * np.power(2.0, exponent - 1) * (1.0 + mantissa / 2.0)
    
    # 3. Use np.where to select the correct value based on the exponent
    is_subnormal = (exponent == 0)
    dequantized_base = np.where(is_subnormal, subnormal_vals, normal_vals)
    
    # Handle the zero case separately to ensure correct sign
    dequantized_base = np.where(values == 0, 0.0, dequantized_base)
    dequantized_base = np.where(values == 8, -0.0, dequantized_base)

    return dequantized_base

def dequantize_mxf4_tensor_optimized(blocks_tensor: np.ndarray, scales_tensor: np.ndarray) -> np.ndarray:
    """
    Vectorized dequantization of an MXF4 tensor.
    
    Args:
        blocks_tensor: Quantized blocks of shape (32, 2880, 90, 16)
        scales_tensor: Scale factors of shape (32, 2880, 90)
    
    Returns:
        Dequantized float32 array of shape (32, 2880, 2880)
    """
    # 1. Unpack 4-bit values from uint8 bytes in a vectorized way
    # Get lower 4 bits (val1) and upper 4 bits (val2) for all bytes at once
    val1 = np.bitwise_and(blocks_tensor, 0x0F)
    val2 = np.bitwise_and(np.right_shift(blocks_tensor, 4), 0x0F)

    # 2. Interleave val1 and val2 to form the full block of 32 values
    # Create an output array with the final dimension doubled
    unpacked_shape = list(blocks_tensor.shape)
    unpacked_shape[-1] *= 2 # From 16 bytes to 32 values
    unpacked_values = np.empty(unpacked_shape, dtype=np.uint8)
    
    unpacked_values[..., ::2] = val1  # Place val1 in even indices
    unpacked_values[..., 1::2] = val2  # Place val2 in odd indices

    # 3. Dequantize the entire array of unpacked values at once
    dequantized_base = dequantize_e2m1_vectorized(unpacked_values)

    # 4. Apply scales using broadcasting
    # Reshape scales_tensor to (32, 2880, 90, 1) to broadcast across the last dimension (32 values)
    scales_broadcastable = scales_tensor[..., np.newaxis]
    dequantized_scaled = dequantized_base * scales_broadcastable

    # 5. Reshape to the final target shape (32, 2880, 90 * 32)
    batch, experts, num_blocks, _ = unpacked_shape
    final_shape = (batch, experts, num_blocks * 32)
    
    return dequantized_scaled.reshape(final_shape).astype(np.float32)

def read_gguf_file(gguf_file_path):
    """
    Reads and prints key-value pairs and tensor information from a GGUF file in an improved format.

    Parameters:
    - gguf_file_path: Path to the GGUF file.
    """

    reader = GGUFReader(gguf_file_path)

    # List all key-value pairs in a columnized format
    print("Key-Value Pairs:") # noqa: NP100
    max_key_length = max(len(key) for key in reader.fields.keys())
    for key, field in reader.fields.items():
        value = field.parts[field.data[0]]
        print(f"{key:{max_key_length}} : {value}") # noqa: NP100
    print("----") # noqa: NP100

    # List all tensors
    print("Tensors:") # noqa: NP100
    tensor_info_format = "{:<30} | Shape: {:<15} | Size: {:<12} | Quantization: {}"
    print(tensor_info_format.format("Tensor Name", "Shape", "Size", "Quantization")) # noqa: NP100
    print("-" * 80) # noqa: NP100
    for tensor in reader.tensors:
        shape_str = "x".join(map(str, tensor.shape))
        size_str = str(tensor.n_elements)
        quantization_str = tensor.tensor_type.name
        print(tensor_info_format.format(tensor.name, shape_str, size_str, quantization_str)) # noqa: NP100

    
    return reader

    




#  --- Restoration Functions ---

def reverse_transform_nibble_layout(tensor: Tensor) -> Tensor:
    """Reverses the custom nibble layout transformation."""
    assert tensor.dtype == torch.uint8
    assert tensor.shape[-1] == 16

    # 1. Reverse the final nibble swap
    t_lo = tensor & 0x0F
    t_hi = tensor & 0xF0
    interleaved = (t_lo << 4) | (t_hi >> 4)

    # 2. De-interleave the nibbles from abababab... back to aaaa...bbbb...
    # The high nibbles of 'interleaved' contain the nibbles for the first half (blk_a)
    nibbles_a_parts = interleaved & 0xF0
    # The low nibbles of 'interleaved' contain the nibbles for the second half (blk_b)
    nibbles_b_parts = interleaved & 0x0F

    # Reconstruct blk_a by packing the high nibbles back together
    # Pair up nibbles: (1st high nibble) | (2nd high nibble >> 4)
    blk_a = nibbles_a_parts[..., 0::2] | (nibbles_a_parts[..., 1::2] >> 4)

    # Reconstruct blk_b by packing the low nibbles back together
    # Pair up nibbles: (1st low nibble << 4) | (2nd low nibble)
    blk_b = (nibbles_b_parts[..., 0::2] << 4) | nibbles_b_parts[..., 1::2]

    deinterleaved = torch.cat((blk_a, blk_b), dim=-1)

    # 3. Reverse the initial nibble swap
    t_lo = deinterleaved & 0x0F
    t_hi = deinterleaved & 0xF0
    original_tensor = (t_lo << 4) | (t_hi >> 4)

    return original_tensor

def restore_from_repack_mxfp4_direct(structured_data: np.ndarray) -> tuple[Tensor, Tensor]:
    """
    Restores the original blocks and scales from structured MXFP4 data.
    
    Args:
        structured_data: Numpy array of shape (32, 2880, 90, 17) from GGUF MXFP4 tensor
    """
    # Make a copy to ensure it's writable and convert to PyTorch tensor
    structured_tensor = torch.from_numpy(structured_data.copy()).to(torch.uint8)
    
    # Split the scales and blocks
    # Scales are the first element in each 17-byte group
    scales_packed = structured_tensor[..., 0]
    # Blocks are the remaining 16 elements
    blocks_packed = structured_tensor[..., 1:17]

    # Restore the blocks by applying the reverse transformation
    blocks_restored = reverse_transform_nibble_layout(blocks_packed)
    
    return blocks_restored, scales_packed


def restore_from_repack_mxfp4(repacked_data: np.ndarray, original_shape: tuple) -> tuple[Tensor, Tensor]:
    """
    Restores the original blocks and scales from the repacked numpy array.
    
    Args:
        repacked_data: The numpy array output by repack_mxfp4.
        original_shape: The original shape of the 'blocks' tensor (e.g., (1, 64, 32, 16)).
    """
    # 1. Convert numpy array back to a PyTorch tensor
    repacked_tensor = torch.from_numpy(repacked_data).to(torch.uint8)

    # 2. Reshape the data to restore the last dimension
    # The packed format has 1 byte for scale + 16 bytes for blocks = 17 bytes per group.
    num_groups = original_shape[2]
    structured_shape = (*original_shape[:2], num_groups, 17)
    structured_data = repacked_tensor.view(structured_shape)
    
    # 3. Split the scales and blocks
    # Scales are the first element in each 17-byte group
    scales_repacked = structured_data[..., 0]
    # Blocks are the remaining 16 elements
    blocks_repacked = structured_data[..., 1:17]

    # 4. Restore the blocks by applying the reverse transformation
    blocks_restored = reverse_transform_nibble_layout(blocks_repacked)
    
    # The scales tensor is already correct, just ensure it has the right shape
    scales_restored = scales_repacked
    
    return blocks_restored, scales_restored



def replace_blk_with_model_layer_str(in_str:str):
    return in_str.replace("blk", "model.layers")
    


def convert_q4_gguf_to_safetensor(gguf_file_path:str, 
                                  data_to_save: dict) -> None:


    
    gguf_reader = GGUFReader(gguf_file_path)
    
    

    # # extract jinja_str
    # chat_template_field = gguf_reader.get_field(gguf.Keys.Tokenizer.CHAT_TEMPLATE)
    # if chat_template_field:
    #     template_str = chat_template_field.contents().strip().replace('\n', '\\n')
    
    
    
    # create a map of gguf_tensor_name with gguf.ReaderTensors
    gguf_name_to_tensors:dict[str, ReaderTensor] = {}
    
    for tensors in gguf_reader.tensors:
        gguf_name_to_tensors[tensors.name] = tensors
    
    
    
    
    # Tensors	Shape	Precision
    # (lm_head.weight) -> output.weight	[2880, 201088]	Q8_0
    # model.norm.weight-> output_norm.weight	[2880]	F32
    #   (model.embed_tokens.weight)-> token_embd.weight [2880, 201088]
    
    
    lm_head_weight_read_tensor = gguf_name_to_tensors["output.weight"]
    assert lm_head_weight_read_tensor.tensor_type.name == "Q8_0"
    data_to_save["lm_head.weight"] = torch.tensor( dequantize(lm_head_weight_read_tensor.data, lm_head_weight_read_tensor.tensor_type), dtype=torch.bfloat16)

    model_weight_reade_tensor: ReaderTensor = gguf_name_to_tensors["output_norm.weight"]
    data_to_save["model.norm.weight"] = torch.tensor( dequantize(model_weight_reade_tensor.data, model_weight_reade_tensor.tensor_type ), dtype=torch.bfloat16 )
    
    model_embed_token_weight = gguf_name_to_tensors["token_embd.weight"]
    assert model_embed_token_weight.tensor_type.name == "Q4_1"
    data_to_save["model.embed_tokens.weight"] = torch.tensor(dequantize(model_embed_token_weight.data, model_embed_token_weight.tensor_type), dtype=torch.bfloat16)
    
    
    TOTAL_LAYER= 24
    
    for layer_idx in range(TOTAL_LAYER):
    # for layer_idx in range(7,8):
        gguf_layer_common_prefix = f"blk.{layer_idx}."
        safetensor_layer_common_prefix = f"model.layers.{layer_idx}."

        mlp_expert_down_proj_bias = gguf_name_to_tensors[gguf_layer_common_prefix + f"ffn_down_exps.bias"]
        data_to_save[safetensor_layer_common_prefix + "mlp.experts.down_proj_bias"] = torch.tensor(dequantize(mlp_expert_down_proj_bias.data, mlp_expert_down_proj_bias.tensor_type), dtype=torch.bfloat16)

        mlp_exp_up_bias = gguf_name_to_tensors[gguf_layer_common_prefix + "ffn_up_exps.bias"]
        mlp_exp_gate_bas = gguf_name_to_tensors[gguf_layer_common_prefix + "ffn_gate_exps.bias"]
        mlp_exp_up_bias = torch.tensor(dequantize(mlp_exp_up_bias.data, mlp_exp_up_bias.tensor_type), dtype=torch.bfloat16)
        mlp_exp_gate_bas = torch.tensor(dequantize(mlp_exp_gate_bas.data, mlp_exp_gate_bas.tensor_type), dtype=torch.bfloat16)
        
        merged_bias = torch.empty( (32, 2880*2), dtype=mlp_exp_gate_bas.dtype )
        merged_bias[..., ::2] = mlp_exp_gate_bas
        merged_bias[..., 1::2] = mlp_exp_up_bias
        data_to_save[safetensor_layer_common_prefix + "mlp.experts.gate_up_proj_bias"] = merged_bias
        
        attn_sinks = gguf_name_to_tensors[gguf_layer_common_prefix + "attn_sinks.weight"]
        attn_sinks = torch.tensor(dequantize(attn_sinks.data, attn_sinks.tensor_type), dtype=torch.bfloat16)
        data_to_save[safetensor_layer_common_prefix + "self_attn.sinks"] = attn_sinks
        
        mlp_router_bias = gguf_name_to_tensors[gguf_layer_common_prefix + "ffn_gate_inp.bias"]
        mlp_router_bias = torch.tensor(dequantize(mlp_router_bias.data, mlp_router_bias.tensor_type), dtype=torch.bfloat16)
        data_to_save[safetensor_layer_common_prefix + "mlp.router.bias"] = mlp_router_bias
        
        mlp_router_weight = gguf_name_to_tensors[gguf_layer_common_prefix + "ffn_gate_inp.weight"] 
        mlp_router_weight = torch.tensor(dequantize(mlp_router_weight.data, mlp_router_weight.tensor_type), dtype=torch.bfloat16)
        data_to_save[safetensor_layer_common_prefix + "mlp.router.weight"] = mlp_router_weight
                
        
        q_proj_weight = gguf_name_to_tensors[gguf_layer_common_prefix + "attn_q.weight"]
        q_proj_weight = torch.tensor(dequantize(q_proj_weight.data, q_proj_weight.tensor_type), dtype=torch.bfloat16)
        data_to_save[safetensor_layer_common_prefix + "self_attn.q_proj.weight"] = q_proj_weight
        
        q_proj_bias = gguf_name_to_tensors[gguf_layer_common_prefix + "attn_q.bias"]
        q_proj_bias = torch.tensor(dequantize(q_proj_bias.data, q_proj_bias.tensor_type), dtype=torch.bfloat16)
        data_to_save[safetensor_layer_common_prefix + "self_attn.q_proj.bias"] = q_proj_bias
        
        k_proj_weight = gguf_name_to_tensors[gguf_layer_common_prefix + "attn_k.weight"]
        k_proj_weight = torch.tensor(dequantize(k_proj_weight.data, k_proj_weight.tensor_type), dtype=torch.bfloat16)
        data_to_save[safetensor_layer_common_prefix + "self_attn.k_proj.weight"] = k_proj_weight
        
        k_proj_bias = gguf_name_to_tensors[gguf_layer_common_prefix + "attn_k.bias"]
        k_proj_bias = torch.tensor(dequantize(k_proj_bias.data, k_proj_bias.tensor_type), dtype=torch.bfloat16)
        data_to_save[safetensor_layer_common_prefix + "self_attn.k_proj.bias"] = k_proj_bias
        
        v_proj_weight = gguf_name_to_tensors[gguf_layer_common_prefix + "attn_v.weight"]
        v_proj_weight = torch.tensor(dequantize(v_proj_weight.data, v_proj_weight.tensor_type), dtype=torch.bfloat16)
        data_to_save[safetensor_layer_common_prefix + "self_attn.v_proj.weight"] = v_proj_weight
        
        v_proj_bias = gguf_name_to_tensors[gguf_layer_common_prefix + "attn_v.bias"]
        v_proj_bias = torch.tensor(dequantize(v_proj_bias.data, v_proj_bias.tensor_type), dtype=torch.bfloat16)
        data_to_save[safetensor_layer_common_prefix + "self_attn.v_proj.bias"] = v_proj_bias
        
        output_proj_weight = gguf_name_to_tensors[gguf_layer_common_prefix + "attn_output.weight"]
        output_proj_weight = torch.tensor(dequantize(output_proj_weight.data, output_proj_weight.tensor_type), dtype=torch.bfloat16)
        data_to_save[safetensor_layer_common_prefix + "self_attn.o_proj.weight"] = output_proj_weight
        
        output_proj_bias = gguf_name_to_tensors[gguf_layer_common_prefix  + "attn_output.bias"]
        output_proj_bias = torch.tensor(dequantize(output_proj_bias.data, output_proj_bias.tensor_type), dtype=torch.bfloat16)
        data_to_save[safetensor_layer_common_prefix + "self_attn.o_proj.bias"] = output_proj_bias
        
        
        post_attention_norm_weight = gguf_name_to_tensors[gguf_layer_common_prefix + "post_attention_norm.weight"]
        post_attention_norm_weight = torch.tensor(dequantize(post_attention_norm_weight.data, post_attention_norm_weight.tensor_type), dtype=torch.bfloat16)
        data_to_save[safetensor_layer_common_prefix + "post_attention_layernorm.weight"] = post_attention_norm_weight
        
        # Add input layernorm
        input_norm_weight = gguf_name_to_tensors[gguf_layer_common_prefix + "attn_norm.weight"]
        input_norm_weight = torch.tensor(dequantize(input_norm_weight.data, input_norm_weight.tensor_type), dtype=torch.bfloat16)
        data_to_save[safetensor_layer_common_prefix + "input_layernorm.weight"] = input_norm_weight
        
        # ffn_down_exps_weights is already being rearranged into GGML type of MXFP4
        # The data is stored as (32, 2880, 1530) where 1530 = 90 * 17 bytes
        ffn_down_exps_weights = gguf_name_to_tensors[gguf_layer_common_prefix + "ffn_down_exps.weight"]
        assert ffn_down_exps_weights.tensor_type.name == "MXFP4", f"Expected MXFP4, got {ffn_down_exps_weights.tensor_type.name}"
        # The data is already shaped as (32, 2880, 1530), need to reshape to (32, 2880, 90, 17)
        ffn_down_data = ffn_down_exps_weights.data.reshape(32, 2880, 90, 17)
        ffn_down_weight, ffn_down_scale = restore_from_repack_mxfp4_direct(ffn_down_data)

        data_to_save[safetensor_layer_common_prefix + "mlp.experts.down_proj_blocks"] = ffn_down_weight
        data_to_save[safetensor_layer_common_prefix + "mlp.experts.down_proj_scales"] = ffn_down_scale
        
        ffn_up_exps_weights = gguf_name_to_tensors[gguf_layer_common_prefix + "ffn_up_exps.weight"]
        assert ffn_up_exps_weights.tensor_type.name == "MXFP4", f"Expected MXFP4, got {ffn_up_exps_weights.tensor_type.name}"
        ffn_up_data = ffn_up_exps_weights.data.reshape(32, 2880, 90, 17)
        ffn_up_weight, ffn_up_scale = restore_from_repack_mxfp4_direct(ffn_up_data)
        
        ffn_gate_exps_weights = gguf_name_to_tensors[gguf_layer_common_prefix  + "ffn_gate_exps.weight"]
        assert ffn_gate_exps_weights.tensor_type.name == "MXFP4", f"Expected MXFP4, got {ffn_gate_exps_weights.tensor_type.name}"
        ffn_gate_data = ffn_gate_exps_weights.data.reshape(32, 2880, 90, 17)
        ffn_gate_weight, ffn_gate_scale = restore_from_repack_mxfp4_direct(ffn_gate_data)
        
        merged_scale = torch.empty( (32, 2880*2, 90), dtype=ffn_up_scale.dtype )
        merged_scale[:, ::2, :] = ffn_gate_scale
        merged_scale[:, 1::2, :] = ffn_up_scale
        
        
        merge_weight =torch.empty( (32, 2880*2, 90, 16 ), dtype=ffn_gate_weight.dtype)
        merge_weight[:, ::2, :, :] = ffn_gate_weight
        merge_weight[:, 1::2, :, :] = ffn_up_weight
        
        
        data_to_save[safetensor_layer_common_prefix  + "mlp.experts.gate_up_proj_scales"] = merged_scale
        data_to_save[safetensor_layer_common_prefix +  "mlp.experts.gate_up_proj_blocks"] = merge_weight
    
        

def debug_dequant_tensors(name:str, data_dequant:dict[str, torch.Tensor], data_ref:dict[str, torch.Tensor]):
    
    print(f"Tensor name: {name}")
    print_tensor_erros(data_dequant[name], data_ref[name])

def debug_dequant_MXF4_dequant(name_block:str, name_scale:str, data_dequant:dict[str, torch.Tensor], data_ref:dict[str, torch.Tensor]):
    
    print(f"Debugging MXF4 dequantization for block: {name_block}, scale: {name_scale}")
    data_dequant_block = data_dequant[name_block]
    data_dequant_scale = data_dequant[name_scale]


    data_ref_block = data_ref[name_block]
    data_ref_scale = data_ref[name_scale]

    data_ref_dequantized = dequantize_mxf4_tensor_optimized(
        data_ref_block.numpy(),
        data_ref_scale.numpy()
    )
    
    data_dequantized = dequantize_mxf4_tensor_optimized(
        data_dequant_block.numpy(),
        data_dequant_scale.numpy()
    )
    
    # assert data_ref_dequantized is not all zeros
    assert not np.all(data_ref_dequantized == 0), "Reference dequantized data is all zeros"
    assert not np.all(data_dequantized == 0), "Dequantized data is all zeros"
    
    # PRINT FIRST 10 VALUE OF EACH
    print("First 10 values of reference dequantized data:")
    print(data_ref_dequantized.flatten()[:10])
    print("First 10 values of dequantized data:")
    print(data_dequantized.flatten()[:10])
    print_numpy_errors(data_dequantized, data_ref_dequantized)

    
    
    # # actually, just compare if the data_dequant_block, data_ref_block are # 1, uint8 and same type
    
    # def assert_equal_tensor(ref, test, name="tensor", max_print=10):
    #     if not torch.equal(ref, test):
    #         mismatches = (ref != test).nonzero(as_tuple=False)
    #         print(f"[Mismatch in {name}] {mismatches.size(0)} differences found")
    #         if mismatches.numel() > 0:
    #             for idx in mismatches[:max_print]:
    #                 idx_tuple = tuple(idx.tolist())
    #                 print(
    #                     f"  idx {idx_tuple}: ref={ref[idx_tuple].item()} "
    #                     f"vs test={test[idx_tuple].item()}"
    #                 )
    #         raise AssertionError(f"{name} does not match")

    # # similar for data_dequant_scale and data_ref_scale
    # print(f"Datatype of data_ref_block: {data_ref_block.dtype} and data_dequant_block {data_dequant_block.dtype}")
    # assert_equal_tensor(data_ref_block, data_dequant_block, "block")
    # assert_equal_tensor(data_ref_scale, data_dequant_scale, "scale")


if __name__ == '__main__':
    # if len(sys.argv) < 2:
    #     logger.info("Usage: reader.py <path_to_gguf_file>")
    #     sys.exit(1)
    # gguf_file_path = sys.argv[1]

    # #FOR NOW debug
    gguf_file_path = "/home/shouyud/gpt-oss-mega-kernel/gpt-oss/unsloth_q4_1_gguf/gpt-oss-20b-Q4_1.gguf"
    # # read_gguf = read_gguf_file(gguf_file_path)
    data_to_save = {}
    convert_q4_gguf_to_safetensor(gguf_file_path, data_to_save=data_to_save)


    safetensor_reference = "/home/shouyud/gpt-oss-mega-kernel/python_code/gpt-oss-evaluation/hf_cache/models--openai--gpt-oss-20b/snapshots/d666cf3b67006cf8227666739edf25164aaffdeb/model-00000-of-00002.safetensors"
    safetensor_reference1 = "/home/shouyud/gpt-oss-mega-kernel/python_code/gpt-oss-evaluation/hf_cache/models--openai--gpt-oss-20b/snapshots/d666cf3b67006cf8227666739edf25164aaffdeb/model-00001-of-00002.safetensors"
    safetensor_reference2 = "/home/shouyud/gpt-oss-mega-kernel/python_code/gpt-oss-evaluation/hf_cache/models--openai--gpt-oss-20b/snapshots/d666cf3b67006cf8227666739edf25164aaffdeb/model-00002-of-00002.safetensors"
    json_reference = "/home/shouyud/gpt-oss-mega-kernel/python_code/gpt-oss-evaluation/hf_cache/models--openai--gpt-oss-20b/snapshots/d666cf3b67006cf8227666739edf25164aaffdeb/model.safetensors.index.json"
    
    
    safetensors_data = {}
    safetensors_data.update(load_file(safetensor_reference))
    safetensors_data.update(load_file(safetensor_reference1))
    safetensors_data.update(load_file(safetensor_reference2))
    
    
    
    
    #NOTE:, compare it with a safetensor file to see how far the values are off
    # # Compare other tensors inline
    # debug_dequant_tensors("lm_head.weight", data_dequant=data_to_save, data_ref=safetensors_data)
    # debug_dequant_tensors("model.embed_tokens.weight", data_dequant=data_to_save, data_ref=safetensors_data)    
    # debug_dequant_tensors("model.norm.weight", data_dequant=data_to_save, data_ref=safetensors_data)

    # # save to a certain file
    # TOTAL_LAYER= 24 # only for now 24
    # for layer_idx in range(TOTAL_LAYER): # for now
    #     safetensor_layer_common_prefix = f"model.layers.{layer_idx}."

    #     debug_dequant_tensors(safetensor_layer_common_prefix + "mlp.experts.down_proj_bias", data_dequant=data_to_save, data_ref=safetensors_data)
    #     debug_dequant_tensors(safetensor_layer_common_prefix + "mlp.experts.gate_up_proj_bias", data_dequant=data_to_save, data_ref=safetensors_data)
    #     debug_dequant_tensors(safetensor_layer_common_prefix + "self_attn.sinks", data_dequant=data_to_save, data_ref=safetensors_data)
    #     debug_dequant_tensors(safetensor_layer_common_prefix + "mlp.router.bias", data_dequant=data_to_save, data_ref=safetensors_data)
    #     debug_dequant_tensors(safetensor_layer_common_prefix + "mlp.router.weight", data_dequant=data_to_save, data_ref=safetensors_data)
        
        
    #     debug_dequant_tensors(safetensor_layer_common_prefix + "self_attn.q_proj.weight", data_dequant=data_to_save, data_ref=safetensors_data)
    #     debug_dequant_tensors(safetensor_layer_common_prefix + "self_attn.q_proj.bias", data_dequant=data_to_save, data_ref=safetensors_data)
    #     debug_dequant_tensors(safetensor_layer_common_prefix + "self_attn.k_proj.weight", data_dequant=data_to_save, data_ref=safetensors_data)
    #     debug_dequant_tensors(safetensor_layer_common_prefix + "self_attn.k_proj.bias", data_dequant=data_to_save, data_ref=safetensors_data)
    #     debug_dequant_tensors(safetensor_layer_common_prefix + "self_attn.v_proj.weight", data_dequant=data_to_save, data_ref=safetensors_data)
    #     debug_dequant_tensors(safetensor_layer_common_prefix + "self_attn.v_proj.bias", data_dequant=data_to_save, data_ref=safetensors_data)
    #     debug_dequant_tensors(safetensor_layer_common_prefix + "self_attn.o_proj.weight", data_dequant=data_to_save, data_ref=safetensors_data)
    #     debug_dequant_tensors(safetensor_layer_common_prefix + "self_attn.o_proj.bias", data_dequant=data_to_save, data_ref=safetensors_data)

        
    #     debug_dequant_tensors(safetensor_layer_common_prefix + "post_attention_layernorm.weight", data_dequant=data_to_save, data_ref=safetensors_data)
    #     debug_dequant_tensors(safetensor_layer_common_prefix + "input_layernorm.weight", data_dequant=data_to_save, data_ref=safetensors_data)

        
    
    #     debug_dequant_MXF4_dequant(
    #         safetensor_layer_common_prefix + "mlp.experts.down_proj_blocks",
    #         safetensor_layer_common_prefix + "mlp.experts.down_proj_scales",
    #         data_dequant=data_to_save,data_ref=safetensors_data
    #     )

        
    #     debug_dequant_MXF4_dequant(
    #         safetensor_layer_common_prefix  + "mlp.experts.gate_up_proj_blocks",
    #         safetensor_layer_common_prefix  + "mlp.experts.gate_up_proj_scales",
    #           data_dequant=data_to_save,data_ref=safetensors_data
    #     )
    
    # #NOTE: TODO: 
    # # do a very debug operation
    # # replace token_embd.weight in data_to_Save with the safetensors_data
    # data_to_save["model.embed_tokens.weight"] = safetensors_data["model.embed_tokens.weight"].clone()


    # also save a json file
    safetensor_filename ="model-00001-of-00001.safetensors" 
    json_filename = "model.safetensors.index.json"
    

    # Ensure all tensors are contiguous before saving
    data_to_save_contiguous = {k: v.contiguous() for k, v in data_to_save.items()}
    save_file(data_to_save_contiguous, safetensor_filename)
    
    
    # get file size
    file_size = os.path.getsize(safetensor_filename)
    json_config_data:dict[str, str] = {}
    for tensor_name in data_to_save.keys():
        json_config_data[tensor_name] = safetensor_filename
    # Open the file in write mode ('w') and use json.dump() to write the dictionary to it
    
    json_to_save = {
        "metadata": {
            "total_size": file_size  # Use actual file size
        },        
        "weight_map" :json_config_data
    }
    
    
    
    # compare if the "weight_map"  of refenrece json has same number element, and same key strings
    with open(json_reference, 'r') as f:
        json_ref_data = json.load(f)

    if sorted(json_ref_data["weight_map"].keys()) != sorted(json_config_data.keys()):
        print("Warning: Weight map keys do not match!")
    if len(json_ref_data["weight_map"]) != len(json_config_data):
        print("Warning: Weight map sizes do not match!")

    with open(json_filename, 'w') as f:
        json.dump(json_to_save, f, indent=4) # 'indent=4' adds pretty-printing for readability

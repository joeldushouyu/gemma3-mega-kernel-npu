import torch
import numpy as np

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



# Necessary to load the local gguf package
gguf_packet_dir = str(Path(__file__).parent.parent.parent.joinpath("SubModules/llama.cpp/gguf-py"))
print("gguf_packet_dir:", gguf_packet_dir)
sys.path.insert(0, gguf_packet_dir)

from gguf.constants import GGMLQuantizationType
from gguf.gguf_reader import GGUFReader
from gguf.quants import dequantize
from gguf import ReaderTensor
import gguf

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
    For every 17 byte in the array(if view as 1D array), 1 byte is the scale, and 16 byte(2 4int value) is the data block
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





# string process func
def replace_blk_with_model_layer_str(in_str:str):
    return in_str.replace("blk", "model.layers")
    




def get_relativeL2(y: np.ndarray, y_ref: np.ndarray) -> float:
    rmse = np.sqrt(np.mean((y - y_ref) ** 2))
    ref_norm = np.sqrt(np.mean(y_ref ** 2))


    return rmse / (1e-8 +  ref_norm)

def get_relativeL1(y: np.ndarray, y_ref: np.ndarray) -> float:
    l1 = np.sum(np.abs(y - y_ref))
    ref_sum = np.sum(np.abs(y_ref))

    return l1 / (1e-8 + ref_sum)

def get_rmse(y: np.ndarray, y_ref: np.ndarray) -> float:
    rmse = np.sqrt(np.mean((y - y_ref) ** 2))
    assert not np.isnan(rmse), "RMSE is NaN, cannot compute RMSE."
    return rmse 

def get_cosine_similarity(y: np.ndarray, y_ref: np.ndarray) -> float:
    # squize to 1D array
    y = y.reshape(-1)
    y_ref = y_ref.reshape(-1)
    dot_product = np.dot(y, y_ref)
    norm_y = np.linalg.norm(y)
    norm_y_ref = np.linalg.norm(y_ref)
    
    # Handle case when both arrays are zero vectors
    if norm_y == 0 and norm_y_ref == 0:
        return 1.0  # Two zero vectors are considered identical

    return dot_product / (norm_y * norm_y_ref + 1e-8)


def tensor_to_numpy(tensor: torch.Tensor, verbose: bool = False) -> np.ndarray:
    """
    Converts a PyTorch tensor to a deep-copied NumPy array.

    Supports tensors on GPU and handles bfloat16 conversion safely.
    bfloat16 will be converted to float32 since NumPy does not support bfloat16.

    Args:
        tensor (torch.Tensor): The input PyTorch tensor.
        verbose (bool, optional): Whether to print debug messages. Defaults to False.

    Returns:
        np.ndarray: The converted NumPy array (deep copy).
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input must be a torch.Tensor, but got {type(tensor)}")

    if tensor.is_cuda:
        if verbose:
            print("Tensor is on GPU, moving to CPU...")
        tensor = tensor.cpu()

    if tensor.dtype == torch.bfloat16:
        if verbose:
            print("Tensor is in bfloat16, converting to float32 for NumPy compatibility.")
        tensor = tensor.to(dtype=torch.float32)

    numpy_array = tensor.detach().numpy().copy()
    if verbose:
        print(f"Converted tensor to NumPy array. Data type: {numpy_array.dtype} (deep copy)")
    return numpy_array



def print_tensor_erros(a:torch.Tensor, b:torch.Tensor):
    
    
    # assert same shape and type
    assert a.shape == b.shape
    assert a.dtype == b.dtype

    # print shape and type
    print(f"Shape: {a.shape}")
    print(f"Type: {a.dtype}")
    
    a_np = tensor_to_numpy(a)
    b_np = tensor_to_numpy(b)
    # assert no nans in a_np and b_np
    assert not np.isnan(a_np).any(), "Array a_np contains NaN values"
    assert not np.isnan(b_np).any(), "Array b_np contains NaN values"
    
    print(f"RMSE: {get_rmse(a_np,b_np)}")
    print(f"Relative L2: {get_relativeL2(a_np,b_np)}")
    print(f"Relative L1: {get_relativeL1(a_np,b_np)}")
    print(f"Cosine Similarity: {get_cosine_similarity(a_np,b_np)}")

def print_numpy_errors(a: np.ndarray, b: np.ndarray):
    # assert same shape
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    
    # print shape and type
    print(f"Shape: {a.shape}")
    print(f"Type: {a.dtype}")
    
    # assert no nans in arrays
    assert not np.isnan(a).any(), "Array a contains NaN values"
    assert not np.isnan(b).any(), "Array b contains NaN values"
    
    print(f"RMSE: {get_rmse(a, b)}")
    print(f"Relative L2: {get_relativeL2(a, b)}")
    print(f"Relative L1: {get_relativeL1(a, b)}")
    print(f"Cosine Similarity: {get_cosine_similarity(a, b)}")
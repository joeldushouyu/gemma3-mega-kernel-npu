



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
from transformers import data
from transformers.data import data_collator
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

import einops
import torch.nn.functional as F

from util import reverse_transform_nibble_layout



def split_ggml_mxfpx_to_scale_blocks(structured_data: np.ndarray):
    """Split GGML MXFP4 data into scales and data blocks

    Format per block (17 bytes):
        - 1 byte: scale (uint8)
        - 16 bytes: 32 x 4-bit float values (2 exponent bits + 1 mantissa bit each)
    """
    assert (structured_data.dtype == np.uint8 or structured_data.dtype == np.int8), "Input must be np.uint8 or np.int8"

    original_shape = structured_data.shape
    assert original_shape[-1] % 17 == 0, "The last dimension must be a multiple of 17"
    
    # Reshape the last dimension into blocks of 17 bytes
    blocks = structured_data.reshape(*original_shape[:-1], -1, 17)
    
    # Extract scales (first byte of each block)
    scales = blocks[..., 0].astype(np.uint8)
    
    # Extract data (remaining 16 bytes, keep as uint8 for 4-bit unpacking)
    data = reverse_transform_nibble_layout( torch.from_numpy( blocks[..., 1:].astype(np.uint8))).numpy()     
    return scales, data

def split_ggml_q80_to_scale_blocks(structured_data: np.ndarray):
    """
    Split GGML Q8_0 structured data into scales and data blocks,
    preserving the original tensor shape.

    Format per block (34 bytes):
        - 2 bytes: scale (int16, little-endian)
        - 32 bytes: quantized data (int8)

    Args:
        structured_data: A NumPy array with dtype=np.uint8 where the
                         last dimension's size is a multiple of 34.

    Returns:
        A tuple (scales, data), where:
        - scales has the shape (*original_shape[:-1], num_blocks) and dtype=np.int16
        - data has the shape (*original_shape[:-1], num_blocks, 32) and dtype=np.int8
    """
    assert (structured_data.dtype == np.uint8 or structured_data.dtype == np.int8), "Input must be np.uint8 or np.int8"

    
    original_shape = structured_data.shape
    assert original_shape[-1] % 34 == 0, "The last dimension must be a multiple of 34"

    # Reshape the last dimension into blocks of 34 bytes
    # For an input of shape (..., N), this becomes (..., N // 34, 34)
    blocks = structured_data.reshape(*original_shape[:-1], -1, 34)

    # Extract the first 2 bytes of each block for the scale.
    # We use .view() to interpret the two uint8 bytes as one int16 value.
    # The .squeeze() removes the now-unnecessary final dimension of size 1.
    scales = blocks[..., :2].view(np.int16).squeeze(axis=-1)

    # Extract the remaining 32 bytes for the quantized data.
    # We use .astype() to cast the uint8 values to the correct int8 type.
    data = blocks[..., 2:].astype(np.int8)

    return scales, data


def split_ggml_q41_to_scale_zero_blocks(structured_data: np.ndarray):
    """
    Splits GGML Q4_1 structured data into scales, biases (zero-points),
    and packed 4-bit data blocks, preserving the original tensor shape.

    Format per block (20 bytes):
        - 2 bytes: scale (float16)
        - 2 bytes: bias/zero-point (float16)
        - 16 bytes: 32 x 4-bit quantized values (uint8)

    Args:
        structured_data: A NumPy array with dtype=np.uint8 where the
                         last dimension's size is a multiple of 20.

    Returns:
        A tuple (scales, biases, data), where:
        - scales has the shape (*original_shape[:-1], num_blocks) and dtype=np.int16
        - biases has the shape (*original_shape[:-1], num_blocks) and dtype=np.int16
        - data has the shape (*original_shape[:-1], num_blocks, 16) and dtype=np.uint8
    """
    assert (structured_data.dtype == np.uint8 or structured_data.dtype == np.int8), "Input must be np.uint8 or np.int8"

    original_shape = structured_data.shape
    block_size = 20
    assert original_shape[-1] % block_size == 0, f"The last dimension must be a multiple of {block_size}"

    # Reshape the last dimension into blocks of 20 bytes
    # For an input of shape (..., N), this becomes (..., N // 20, 20)
    blocks = structured_data.reshape(*original_shape[:-1], -1, block_size)

    # First 2 bytes are scales (d), interpreted as int16
    scales = blocks[..., :2].view(np.int16).squeeze(axis=-1)
    
    # Next 2 bytes are biases/zero-points (m), interpreted as int16
    biases = blocks[..., 2:4].view(np.int16).squeeze(axis=-1)

    # Remaining 16 bytes contain 32 4-bit values (packed)
    data = blocks[..., 4:]

    return scales, biases, data





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

from ggml_util import split_ggml_mxfpx_to_scale_blocks, split_ggml_q41_to_scale_zero_blocks, split_ggml_q80_to_scale_blocks
from util import print_tensor_erros



def convert_float16_np_to_bfloat16_np(scales):
    # Convert scales from int16 (representing float16 bits) to bfloat16
    # First interpret int16 as float16, then convert to float32, then to bfloat16
    scales_as_float16 = scales.view(np.float16)  # Reinterpret int16 bits as float16
    scales_as_float32 = scales_as_float16.astype(np.float32)  # Convert to float32
    scales_tensor = torch.from_numpy(scales_as_float32).to(torch.bfloat16)  # Convert to bfloat16
    scales_bfloat16 = scales_tensor.view(torch.int16).numpy()  # View bfloat16 bits as int16, then convert to numpy
    return scales_bfloat16



def convert_q41_gguf_data_to_q4nx_format(
    structured_data:npt.NDArray,
    q4nx_block_row_size:int,
    q4nx_block_col_size:int, q4nx_block_col_stride:int
):
    
    scales, biases, data = split_ggml_q41_to_scale_zero_blocks(
        structured_data
    )

    scales = convert_float16_np_to_bfloat16_np(scales)
    biases = convert_float16_np_to_bfloat16_np(biases)
    Q4_1_block_size = 32
    Q4_1_block_size_data_in_byte = 16 # because 4bit data, so 32 data is 16 byte
    
    scales = torch.from_numpy(scales)
    biases = torch.from_numpy(biases)
    data = torch.from_numpy(data)
    

    if scales.shape[-2] % q4nx_block_row_size !=0:
        scales = F.pad(scales, (0, 0, 0, q4nx_block_row_size - scales.shape[-2] % q4nx_block_row_size), "constant", 0)
        
    if (scales.shape[-1] * Q4_1_block_size) % q4nx_block_col_size !=0:
        addition_padd_size = (q4nx_block_col_size - ((scales.shape[-1] * Q4_1_block_size) % q4nx_block_col_size)) // Q4_1_block_size
        assert (q4nx_block_col_size - ((scales.shape[-1] * Q4_1_block_size) % q4nx_block_col_size)) % Q4_1_block_size == 0

        scales = F.pad(scales, (0, addition_padd_size, 0, 0), "constant", 0)

    if biases.shape[-2] % q4nx_block_row_size !=0:
        biases = F.pad(biases, (0, 0, 0, q4nx_block_row_size - biases.shape[-2] % q4nx_block_row_size), "constant", 0)
    if (biases.shape[-1] * Q4_1_block_size) % q4nx_block_col_size !=0:
        addition_padd_size = (q4nx_block_col_size - ((biases.shape[-1] * Q4_1_block_size) % q4nx_block_col_size)) // Q4_1_block_size
        assert (q4nx_block_col_size - ((biases.shape[-1] * Q4_1_block_size) % q4nx_block_col_size)) % Q4_1_block_size == 0

        biases = F.pad(biases, (0, addition_padd_size, 0, 0), "constant", 0)

    
    
    if data.shape[-3] % q4nx_block_row_size != 0:
        data = F.pad(data, (0, 0, 0,0, 0, q4nx_block_row_size - data.shape[-3] % q4nx_block_row_size), "constant", 0)
    if data.shape[-2] * Q4_1_block_size_data_in_byte % q4nx_block_col_size != 0:
        addition_padd_size = (q4nx_block_col_size - ((data.shape[-2] * Q4_1_block_size_data_in_byte) % q4nx_block_col_size)) // Q4_1_block_size_data_in_byte
        assert (q4nx_block_col_size - ((data.shape[-2] * Q4_1_block_size_data_in_byte) % q4nx_block_col_size)) % Q4_1_block_size_data_in_byte == 0

        data = F.pad(data, (0, 0, 0, addition_padd_size, 0, 0), "constant", 0)

    
    #reshape down to 2D by squashing the previous dimension
    scale_shape = scales.shape
    prev_shape = 1
    for i in range(len(scale_shape) - 2):
        prev_shape *= scale_shape[i]
    new_scale_shape = [prev_shape * scales.shape[-2], scales.shape[-1]]  # Flatten to 2D
    scales = scales.view(new_scale_shape).contiguous()
    
    biases_shape = biases.shape
    prev_shape = 1
    for i in range(len(biases_shape) - 2):
        prev_shape *= biases_shape[i]
    new_biases_shape = [prev_shape * biases.shape[-2], biases.shape[-1]]  # Flatten to 2D
    biases = biases.view(new_biases_shape).contiguous()
    
    data_shape = data.shape
    prev_shape = 1
    for i in range(len(data_shape) - 3):
        prev_shape *= data_shape[i]
    new_data_shape = [prev_shape * data.shape[-3], data.shape[-2] * data.shape[-1]]  # Flatten to 2D: shape[-1] is 16, the datablock size
    data = data.reshape(new_data_shape).contiguous()
    
    
    # calcuate dimension for scales and biases
    row_div_q4_row = scales.shape[0] // q4nx_block_row_size
    col_div_q4_col = scales.shape[1] // (q4nx_block_col_size // Q4_1_block_size)
    
    scales = einops.rearrange(
        scales, 
        "(row_div_q4_row q4_row) (col_div_q4_col q4_col_div32) -> row_div_q4_row col_div_q4_col q4_row q4_col_div32", 
        row_div_q4_row=row_div_q4_row,  
        col_div_q4_col=col_div_q4_col,
        q4_row=q4nx_block_row_size, 
        q4_col_div32=q4nx_block_col_size//Q4_1_block_size
    ).contiguous()
    
    biases = einops.rearrange(
        biases, 
        "(row_div_q4_row q4_row) (col_div_q4_col q4_col_div32) -> row_div_q4_row col_div_q4_col q4_row q4_col_div32", 
        row_div_q4_row=row_div_q4_row,  
        col_div_q4_col=col_div_q4_col,
        q4_row=q4nx_block_row_size, 
        q4_col_div32=q4nx_block_col_size//Q4_1_block_size
    ).contiguous()
    
    # Calculate dimensions for data
    data_row_div = data.shape[0] // q4nx_block_row_size
    data_col_div = data.shape[1] // (q4nx_block_col_size//2) # divide another extra 2, because 2 int4 in 1 byte
    
 

    data = einops.rearrange(
        data, 
        "(row_div_q4_row q4_row) (col_div_q4_col q4_col) -> row_div_q4_row col_div_q4_col q4_row q4_col",
        row_div_q4_row=data_row_div, 
        col_div_q4_col=data_col_div,
        q4_row=q4nx_block_row_size, 
        q4_col=(q4nx_block_col_size//2) # divide another extra 2, because 2 int4 in 1 byte
    ).contiguous()
    
    

    # at this step, both scales and data are 
        # 1. row major within the blocks
        # 2. Also row major in block level

    assert q4nx_block_col_stride == 16
    assert q4nx_block_row_size % q4nx_block_col_stride == 0
    
    
    data = einops.rearrange(
        data,
        "row_div col_div (q4_row_div_col_stride col_stride) (q4_col one) -> \
        row_div col_div (q4_row_div_col_stride q4_col) (col_stride one)",
        col_stride = q4nx_block_col_stride, # 16 element, since each data is half-byte
        one = 1,
    ).contiguous()
    
    scales = einops.rearrange(
        scales,
        "row_div col_div (q4_row_div_col_stride col_stride) (q4_col one) -> \
        row_div col_div (q4_row_div_col_stride q4_col) (col_stride one)",
        col_stride = q4nx_block_col_stride, # 16 element, since each data is half-byte
        one = 1,
    ).contiguous()
    
    biases = einops.rearrange(
        biases,
        "row_div col_div (q4_row_div_col_stride col_stride) (q4_col one) -> \
        row_div col_div (q4_row_div_col_stride q4_col) (col_stride one)",
        col_stride = q4nx_block_col_stride, # 16 element, since each data is half-byte
        one = 1,
    ).contiguous()    
    return scales,  biases, data





def convert_q80_gguf_data_to_q4nx_format(
    structured_data:npt.NDArray,
    q4nx_block_row_size:int,
    q4nx_block_col_size:int,
    q4nx_block_col_stride:int,   # The stride of each block, because each block internally is column major
):
    # first, let us extract the scale and blocks
    scales_np, data_np = split_ggml_q80_to_scale_blocks(structured_data)

    # convert scales's internal data representation from float16 to bfloat16
    scales_np = convert_float16_np_to_bfloat16_np(scales_np)
    assert len(scales_np.shape) >=2 and len(data_np.shape) >= 2

    
    Q8_block_size = 32
    scales: Tensor = torch.from_numpy(scales_np).contiguous()
    data = torch.from_numpy(data_np).contiguous()
    
    
    
    if scales.shape[-2] % q4nx_block_row_size !=0:
        scales = F.pad(scales, (0, 0, 0, q4nx_block_row_size - scales.shape[-2] % q4nx_block_row_size), "constant", 0)
    if (scales.shape[-1] *Q8_block_size) % q4nx_block_col_size !=0:
        addition_padd_size = (q4nx_block_col_size - ((scales.shape[-1]*Q8_block_size) % q4nx_block_col_size) )//Q8_block_size
        assert  (q4nx_block_col_size - ((scales.shape[-1]*Q8_block_size) % q4nx_block_col_size) ) % Q8_block_size == 0
        
        scales = F.pad(scales, (0, addition_padd_size, 0, 0), "constant", 0)


    # last data.shape[-1] is 32, the block size, so
    
    if data.shape[-3] % q4nx_block_row_size != 0:
        data = F.pad(data, (0, 0, 0,0, 0, q4nx_block_row_size - data.shape[-3] % q4nx_block_row_size), "constant", 0)
    if (data.shape[-2] * Q8_block_size) % q4nx_block_col_size != 0:
        addition_padd_size = (q4nx_block_col_size - ((data.shape[-2]*Q8_block_size) % q4nx_block_col_size) ) // Q8_block_size
        assert (q4nx_block_col_size - ((data.shape[-2]*Q8_block_size) % q4nx_block_col_size) ) % Q8_block_size == 0

        data = F.pad(data, (0,0, 0, addition_padd_size, 0, 0), "constant", 0)
    
    
    # reshape down to 2D by squashing the previous dimensions
    scale_shape = scales.shape
    prev_shape = 1  # Changed from 0 to 1 - should multiply, not add
    for i in range(len(scale_shape)-2):  # Changed from -1 to -2 to keep last 2 dimensions
        prev_shape *= scale_shape[i]  # Changed from += to *=
    new_scale_shape = [prev_shape * scales.shape[-2], scales.shape[-1]]  # Flatten to 2D

    scales = scales.view(new_scale_shape).contiguous()
    
    
    data_shape = data.shape
    prev_shape = 1  # Changed from 0 to 1 - should multiply, not add
    for i in range(len(data_shape) - 3):  # Changed to handle 3D properly
        prev_shape *= data_shape[i]  # Changed from += to *=
    
    new_data_shape = [prev_shape * data.shape[-3], data.shape[-2] * data.shape[-1]]  # Flatten to 2D: shape[-1] is 32, the datablock size

    data = data.view(new_data_shape).contiguous()
    

    # Calculate dimensions for scales
    row_div_q4_row = scales.shape[0] // q4nx_block_row_size
    col_div_q4_col = scales.shape[1] // (q4nx_block_col_size // 32)
    
    
    scales = einops.rearrange(
        scales, 
        "(row_div_q4_row q4_row) (col_div_q4_col q4_col_div32) -> row_div_q4_row col_div_q4_col q4_row q4_col_div32", 
        row_div_q4_row=row_div_q4_row,  
        col_div_q4_col=col_div_q4_col,
        q4_row=q4nx_block_row_size, 
        q4_col_div32=q4nx_block_col_size//32
    ).contiguous()

    # Calculate dimensions for data
    data_row_div = data.shape[0] // q4nx_block_row_size
    data_col_div = data.shape[1] // q4nx_block_col_size
    
 

    data = einops.rearrange(
        data, 
        "(row_div_q4_row q4_row) (col_div_q4_col q4_col) -> row_div_q4_row col_div_q4_col q4_row q4_col",
        row_div_q4_row=data_row_div, 
        col_div_q4_col=data_col_div,
        q4_row=q4nx_block_row_size, 
        q4_col=q4nx_block_col_size
    ).contiguous()
    
    
    
    # at this step, both scales and data are 
        # 1. row major within the blocks
        # 2. Also row major in block level

    assert q4nx_block_col_stride == 16
    assert q4nx_block_row_size % q4nx_block_col_stride == 0


    data = einops.rearrange(
        data,
        "row_div col_div (q4_row_div_col_stride col_stride) (q4_col one) -> \
        row_div col_div (q4_row_div_col_stride q4_col) (col_stride one)",
        q4_row_div_col_stride = q4nx_block_row_size // q4nx_block_col_stride,
        col_stride = q4nx_block_col_stride, # 16 element, since each data is 1byte
        one = 1,
    ).contiguous()
    
    scales = einops.rearrange(
        scales,
        "row_div col_div (q4_row_div_col_stride col_stride) (q4_col one) -> \
        row_div col_div (q4_row_div_col_stride q4_col) (col_stride one)",
        q4_row_div_col_stride = q4nx_block_row_size // q4nx_block_col_stride,
        col_stride = q4nx_block_col_stride, # 16 element, since each data is 1byte
        one = 1,
    ).contiguous()
    
    
    return scales, data




def convert_mxfp4_gguf_data_to_q4nx_format(structured_data:npt.NDArray,
                                            q4nx_block_row_size:int,
                                            q4nx_block_col_size:int, q4nx_block_col_stride:int
    ):
    
    scales, data = split_ggml_mxfpx_to_scale_blocks(structured_data)
    
    MXFP4_BLOCK_SIZE= 32
    MXFP4_BLOCK_SIZE_data_in_byte = 16 # because 4 bit data, so 32 data is 16 byte
    
    
    scales = torch.from_numpy(scales)
    data = torch.from_numpy(data)
    
    if scales.shape[-2] % q4nx_block_row_size !=0:
        scales = F.pad(scales, (0, 0, 0, q4nx_block_row_size - scales.shape[-2] % q4nx_block_row_size), "constant", 0)
        
    if (scales.shape[-1] * MXFP4_BLOCK_SIZE) % q4nx_block_col_size !=0:
        addition_padd_size = (q4nx_block_col_size - ((scales.shape[-1] * MXFP4_BLOCK_SIZE) % q4nx_block_col_size)) // MXFP4_BLOCK_SIZE
        assert (q4nx_block_col_size - ((scales.shape[-1] * MXFP4_BLOCK_SIZE) % q4nx_block_col_size)) % MXFP4_BLOCK_SIZE == 0

        scales: Tensor = F.pad(scales, (0, addition_padd_size, 0, 0), "constant", 0)
    
    if data.shape[-3] % q4nx_block_row_size != 0:
        data = F.pad(data, (0, 0, 0,0, 0, q4nx_block_row_size - data.shape[-3] % q4nx_block_row_size), "constant", 0)
    if data.shape[-2] * MXFP4_BLOCK_SIZE_data_in_byte % q4nx_block_col_size != 0:
        addition_padd_size = (q4nx_block_col_size - ((data.shape[-2] * MXFP4_BLOCK_SIZE_data_in_byte) % q4nx_block_col_size)) // MXFP4_BLOCK_SIZE_data_in_byte
        assert (q4nx_block_col_size - ((data.shape[-2] * MXFP4_BLOCK_SIZE_data_in_byte) % q4nx_block_col_size)) % MXFP4_BLOCK_SIZE_data_in_byte == 0

        data = F.pad(data, (0, 0, 0, addition_padd_size, 0, 0), "constant", 0)
        
    # # calcuate dimension for scales and biases

    scales= scales.contiguous()
    data = data.contiguous()
    # if len(scales.shape) == 2:
    #     # simply 2D 
    #     row_div_q4_row = scales.shape[0] // q4nx_block_row_size
    #     col_div_q4_col = scales.shape[1] // (q4nx_block_col_size // MXFP4_BLOCK_SIZE)
    #     scales = einops.rearrange(
    #         scales, 
    #         "(row_div_q4_row q4_row) (col_div_q4_col q4_col_div32) -> row_div_q4_row col_div_q4_col q4_row q4_col_div32", 
    #         row_div_q4_row=row_div_q4_row,  
    #         col_div_q4_col=col_div_q4_col,
    #         q4_row=q4nx_block_row_size, 
    #         q4_col_div32=q4nx_block_col_size//MXFP4_BLOCK_SIZE
    #     ).contiguous()
    # else:
    assert len(scales.shape) == 3

    row_div_q4_row = scales.shape[1] // q4nx_block_row_size
    col_div_q4_col = scales.shape[2] // (q4nx_block_col_size // MXFP4_BLOCK_SIZE)
    scales = einops.rearrange(
        scales, 
        "batch (row_div_q4_row q4_row) (col_div_q4_col q4_col_div32) -> batch row_div_q4_row col_div_q4_col q4_row q4_col_div32", 
        row_div_q4_row=row_div_q4_row,  
        col_div_q4_col=col_div_q4_col,
        q4_row=q4nx_block_row_size, 
        q4_col_div32=q4nx_block_col_size//MXFP4_BLOCK_SIZE
    ).contiguous()            
    


    # combine the block dim
    # Get the dimensions *before* the last two and pass the full shape to view()
    data: Tensor = data.reshape(*data.shape[:-2], -1)
    # if len(data.shape) == 2:
    #     data_row_div = data.shape[0] // q4nx_block_row_size
    #     data_col_div = data.shape[1] // (q4nx_block_col_size // 2)
    #     data = einops.rearrange(
    #         data,
    #         "(row_div_q4_row q4_row) (col_div_q4_col q4_col) -> row_div_q4_row col_div_q4_col q4_row q4_col",
    #         row_div_q4_row=data_row_div,
    #         col_div_q4_col=data_col_div,
    #         q4_row=q4nx_block_row_size,
    #         q4_col=(q4nx_block_col_size // 2)  # divide another extra 2, because 2 int4 in 1 byte            
    #     ).contiguous()
    # else:
    assert len(data.shape) == 3
    data_row_div = data.shape[1] // q4nx_block_row_size
    data_col_div = data.shape[2] // (q4nx_block_col_size // 2)
    data = einops.rearrange(
        data,
        "batch (row_div_q4_row q4_row) (col_div_q4_col q4_col) -> batch row_div_q4_row col_div_q4_col q4_row q4_col",
        row_div_q4_row=data_row_div,
        col_div_q4_col=data_col_div,
        q4_row=q4nx_block_row_size,
        q4_col=(q4nx_block_col_size // 2)
    ).contiguous()

    # at this step, both scales and data are 
        # 1. row major within the blocks
        # 2. Also row major in block level

    assert q4nx_block_col_stride == 16
    assert q4nx_block_row_size % q4nx_block_col_stride == 0
    
    
    data = einops.rearrange(
        data,
        "batch row_div col_div (q4_row_div_col_stride col_stride) (q4_col one) -> \
        batch row_div col_div (q4_row_div_col_stride q4_col) (col_stride one)",
        col_stride = q4nx_block_col_stride, # 16 element, since each data is half-byte
        one = 1,
    ).contiguous()
    
    scales = einops.rearrange(
        scales,
        "batch row_div col_div (q4_row_div_col_stride col_stride) (q4_col one) -> \
        batch row_div col_div (q4_row_div_col_stride q4_col) (col_stride one)",
        col_stride = q4nx_block_col_stride, # 16 element, since each data is half-byte
        one = 1,
    ).contiguous()
    
    
    
    return scales, data
    






# corresponding dequant q4nx
def dequant_q80_q4nx_data_format(scale_block: torch.Tensor, data_block: torch.Tensor):
    # for each scale (reinterpret int16 as bfloat16 data bits) correspond to 32 int8 in data_block
    # scale_block should be torch tensor with bfloat16 data stored as int16 bits
    # data_block should be torch tensor with int8 data
    
    # Convert scale_block from int16 bits to bfloat16 values
    scale_as_bfloat16 = scale_block.view(torch.bfloat16)
    
    # Convert data_block from int8 to float32
    data_as_float32 = data_block.float()
    
    # Apply dequantization: each scale corresponds to 32 int8 values
    # Reshape to align scales with data blocks
    scale_expanded = scale_as_bfloat16.repeat_interleave(32, dim=-1)
    
    # Multiply data by scales
    dequantized = data_as_float32 * scale_expanded
    
    return dequantized

def dequant_q41_q4nx_data_format(scale_block:torch.Tensor, biases_block:torch.Tensor, data_block:torch.Tensor):
    # for each scale(reinterpret int16 as bfloat16) and each biases (reinterpret int16 as bfloat16) correspond to 32 int4 value( 1byte has 2 int4 value, so 16 byte)
    
    # Convert scale_block and biases_block from int16 bits to bfloat16 values
    scale_as_bfloat16 = scale_block.view(torch.bfloat16)
    biases_as_bfloat16 = biases_block.view(torch.bfloat16)
    
    # Follow the exact GGML Q4_1 unpacking approach
    # Save original shape for later
    original_shape = data_block.shape
    
    # Flatten all but last dimension, then unpack following GGML logic
    # data_block shape: [..., 16] -> reshape to (-1, 16)
    data_flat = data_block.reshape(-1, 16)
    n_blocks = data_flat.shape[0]
    
    # Apply GGML unpacking: reshape to (n_blocks, 1, 1, 16) for broadcasting
    qs = data_flat.reshape(n_blocks, 1, 1, 16)
    
    # Apply bit shifts using GGML approach
    shifts = torch.tensor([0, 4], dtype=torch.uint8, device=data_block.device).reshape(1, 1, 2, 1)
    qs = torch.bitwise_right_shift(qs.unsqueeze(2), shifts.to(qs.dtype))
    
    # Mask and reshape to get final unpacked values
    qs = torch.bitwise_and(qs, 0x0F).reshape(n_blocks, 32).float()
    
    # Reshape qs back to match the scale/bias structure but with 32 values per block
    # scales/biases have shape like [row_div, col_div, q4_row, q4_col_div32]
    # we need to expand to [row_div, col_div, q4_row, q4_col] where q4_col = q4_col_div32 * 32
    scale_shape = scale_as_bfloat16.shape
    qs = qs.reshape(scale_shape + (32,))
    
    # Apply Q4_1 dequantization formula: value = scale * quantized_value + bias
    # Expand scales and biases to match the final shape
    scale_expanded = scale_as_bfloat16.unsqueeze(-1).expand_as(qs)
    biases_expanded = biases_as_bfloat16.unsqueeze(-1).expand_as(qs)
    
    # Apply dequantization: scale * data + bias
    dequantized = scale_expanded.float() * qs + biases_expanded.float()
    
    # Flatten the last two dimensions to match Q8_0 format
    # From [..., q4_col_div32, 32] to [..., q4_col_div32 * 32]
    final_shape = list(dequantized.shape[:-2]) + [dequantized.shape[-2] * dequantized.shape[-1]]
    dequantized = dequantized.reshape(final_shape)
    
    return dequantized


# see ggml_e8m0_to_fp32_half in ggml-impl.h
def e8m0_to_fp32_half(x: np.ndarray) -> np.ndarray:
    bits = np.where(x < 2, np.uint32(0x00200000) << np.uint32(x), np.uint32(x - 1) << np.uint32(23))
    return bits.view(np.float32)

def dequant_mxfp4_q4nx_data_format(scale_block, data_block, ggml_layout=True):
    KVALUES_MXFP4 = torch.tensor([
         0.0,  1.0,  2.0,  3.0,  4.0,  6.0,  8.0, 12.0,
         0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0,-12.0
    ], dtype=torch.float32, device=data_block.device)

    # 1. Decode E8M0 exponent
    # scale = e8m0_to_fp32_half(scale_block)
    scale = torch.from_numpy( e8m0_to_fp32_half(scale_block.numpy()))
    # 2. Split nibbles
    low  = data_block & 0x0F
    high = data_block >> 4

    vals_low  = KVALUES_MXFP4[low.long()]
    vals_high = KVALUES_MXFP4[high.long()]

    # 3. Arrange in GGML layout
    dequantized_block = torch.cat((vals_low, vals_high), dim=-1)

    # 4. Apply scale
    result = dequantized_block * scale.repeat_interleave(32,-1)

    return result






# Debugging 

def compare_q4nx_q80_with_ref( q4nx_q80_scale:torch.Tensor, q4nx_q80_data:torch.Tensor, reference:torch.Tensor,
                            Q4NX_PER_BLOCK_ROW_SIZE:int, Q4NX_PER_BLOCK_COL_SIZE:int, Q4NX_PER_BLOCK_COL_STRIDE:int):
    
    # first reverse the rearrange, then do normal dequant when comparing with the reference
    assert Q4NX_PER_BLOCK_ROW_SIZE%Q4NX_PER_BLOCK_COL_STRIDE == 0
    q4nx_q80_scale = einops.rearrange(
        q4nx_q80_scale,
        "row_div col_div (q4_row_div_col_stride q4_col) (col_stride one) -> \
           row_div col_div (q4_row_div_col_stride col_stride) (q4_col one)",
        one = 1,
        col_stride = Q4NX_PER_BLOCK_COL_STRIDE,
        q4_row_div_col_stride = Q4NX_PER_BLOCK_ROW_SIZE//Q4NX_PER_BLOCK_COL_STRIDE
    )

    q4nx_q80_data = einops.rearrange(
        q4nx_q80_data,
        "row_div col_div (q4_row_div_col_stride q4_col) (col_stride one) -> \
           row_div col_div (q4_row_div_col_stride col_stride) (q4_col one)",
           one = 1,
           col_stride = Q4NX_PER_BLOCK_COL_STRIDE,
           q4_row_div_col_stride = Q4NX_PER_BLOCK_ROW_SIZE//Q4NX_PER_BLOCK_COL_STRIDE
    )

    q4nx_q80_scale = einops.rearrange(
        q4nx_q80_scale,
        "row_div col_div q4_row q4_col -> (row_div q4_row) (col_div q4_col)"
    ).contiguous()
    
    q4nx_q80_data = einops.rearrange(
        q4nx_q80_data,
        "row_div col_div q4_row q4_col -> (row_div q4_row) (col_div q4_col)"
    ).contiguous()
    
    q4nx_q80_data = einops.rearrange(
        q4nx_q80_data,
        "row (col_div_group group) -> row col_div_group group",
        group= 32
    ).contiguous()
    q4nx_q80_scale = q4nx_q80_scale.reshape(q4nx_q80_scale.shape + (1,))
    dequant_q4nx_lm_head_weight = dequant_q80_q4nx_data_format(q4nx_q80_scale, q4nx_q80_data)
    # Note, the last two dimension of dequant_q4nx_lm_head_weight is being being paddd with 0 for Q4NX_PER_BLOCK_ROW_SIZE and Q4NX_PER_BLOCK_COL_SIZE multiples
    # thus, we also need to do the same on ref_lm_head_weight
    
    # Calculate padding needed for dimension -2 (rows)
    row_pad = (Q4NX_PER_BLOCK_ROW_SIZE - (reference.shape[-2] % Q4NX_PER_BLOCK_ROW_SIZE)) % Q4NX_PER_BLOCK_ROW_SIZE
    # Calculate padding needed for dimension -1 (cols)
    col_pad = (Q4NX_PER_BLOCK_COL_SIZE - (reference.shape[-1] % Q4NX_PER_BLOCK_COL_SIZE)) % Q4NX_PER_BLOCK_COL_SIZE
    
    # Apply padding: F.pad takes (left, right, top, bottom, front, back, ...)
    # For last two dimensions: (left_pad_dim-1, right_pad_dim-1, left_pad_dim-2, right_pad_dim-2)
    reference_padded = F.pad(reference, (0, col_pad, 0, row_pad), "constant", 0)
    

    print_tensor_erros(dequant_q4nx_lm_head_weight.flatten(), reference_padded.flatten().to(torch.float32))


def compare_q4nx_q41_with_ref(q4nx_q41_scale:torch.Tensor, q4nx_q41_biases:torch.Tensor, q4nx_q41_data:torch.Tensor, reference:torch.Tensor,
                               Q4NX_PER_BLOCK_ROW_SIZE:int, Q4NX_PER_BLOCK_COL_SIZE:int, Q4NX_PER_BLOCK_COL_STRIDE:int
                              ):
    
    
    # first, revert the rearrange
    assert Q4NX_PER_BLOCK_ROW_SIZE%Q4NX_PER_BLOCK_COL_STRIDE == 0
    
    
    q4nx_q41_scale = einops.rearrange(
        q4nx_q41_scale,
        "row_div col_div (q4_row_div_col_stride q4_col) (col_stride one) -> \
           row_div col_div (q4_row_div_col_stride col_stride) (q4_col one)",
        one = 1,
        q4_row_div_col_stride = Q4NX_PER_BLOCK_ROW_SIZE // Q4NX_PER_BLOCK_COL_STRIDE,
        col_stride = Q4NX_PER_BLOCK_COL_STRIDE
    ).contiguous()
    q4nx_q41_biases = einops.rearrange(
        q4nx_q41_biases,
        "row_div col_div (q4_row_div_col_stride q4_col) (col_stride one) -> \
           row_div col_div (q4_row_div_col_stride col_stride) (q4_col one)",
        one = 1,
        q4_row_div_col_stride = Q4NX_PER_BLOCK_ROW_SIZE // Q4NX_PER_BLOCK_COL_STRIDE,
        col_stride = Q4NX_PER_BLOCK_COL_STRIDE
    ).contiguous()


    q4nx_q41_data = einops.rearrange(
        q4nx_q41_data,
        "row_div col_div (q4_row_div_col_stride q4_col) (col_stride one) -> \
           row_div col_div (q4_row_div_col_stride col_stride) (q4_col one)",
           one = 1,
            q4_row_div_col_stride = Q4NX_PER_BLOCK_ROW_SIZE // Q4NX_PER_BLOCK_COL_STRIDE,           
           col_stride = Q4NX_PER_BLOCK_COL_STRIDE
        ).contiguous()
    
    
    
    q4nx_q41_scale = einops.rearrange(
        q4nx_q41_scale,
        "row_div col_div q4_row q4_col -> (row_div q4_row) (col_div q4_col)"
    ).contiguous()
    q4nx_q41_biases = einops.rearrange(
        q4nx_q41_biases,
        "row_div col_div q4_row q4_col -> (row_div q4_row) (col_div q4_col)"
    ).contiguous()
    
    q4nx_q41_data = einops.rearrange(
        q4nx_q41_data,
        "row_div col_div q4_row q4_col -> (row_div q4_row) (col_div q4_col)"
    ).contiguous()
    
    q4nx_q41_data = einops.rearrange(
        q4nx_q41_data,
        pattern="row (col_div_group group) -> row col_div_group group",
        group= 16
    ).contiguous()
    
    dequant_q4nx_weight = dequant_q41_q4nx_data_format(q4nx_q41_scale, q4nx_q41_biases, q4nx_q41_data)
    # Note, the last two dimension of dequant_q4nx_weight is being being paddd with 0 for Q4NX_PER_BLOCK_ROW_SIZE and Q4NX_PER_BLOCK_COL_SIZE multiples
    # thus, we also need to do the same on ref_lm_head_weight
    
    # Calculate padding needed for dimension -2 (rows)
    row_pad = (Q4NX_PER_BLOCK_ROW_SIZE - (reference.shape[-2] % Q4NX_PER_BLOCK_ROW_SIZE)) % Q4NX_PER_BLOCK_ROW_SIZE
    # Calculate padding needed for dimension -1 (cols)
    col_pad = (Q4NX_PER_BLOCK_COL_SIZE - (reference.shape[-1] % Q4NX_PER_BLOCK_COL_SIZE)) % Q4NX_PER_BLOCK_COL_SIZE
    
    # Apply padding: F.pad takes (left, right, top, bottom, front, back, ...)
    # For last two dimensions: (left_pad_dim-1, right_pad_dim-1, left_pad_dim-2, right_pad_dim-2)
    reference_padded = F.pad(reference, (0, col_pad, 0, row_pad), "constant", 0)
    
    # Now compare the tensors
    print_tensor_erros(dequant_q4nx_weight, reference_padded.to(torch.float32))


def compare_q4nx_mxfp4_with_ref( q4nx_mxfp4_scale:torch.Tensor, q4nx_mxfp4_data:torch.Tensor, reference:torch.Tensor,
                                Q4NX_PER_BLOCK_ROW_SIZE:int, Q4NX_PER_BLOCK_COL_SIZE:int, Q4NX_PER_BLOCK_COL_STRIDE:int):
    
    t = torch.from_numpy( e8m0_to_fp32_half(q4nx_mxfp4_scale.numpy()))
    # first, revert the rearrange
    assert Q4NX_PER_BLOCK_ROW_SIZE%Q4NX_PER_BLOCK_COL_STRIDE == 0
    
    
    q4nx_mxfp4_scale = einops.rearrange(
        q4nx_mxfp4_scale,
        "batch row_div col_div (q4_row_div_col_stride q4_col) (col_stride one) -> \
           batch row_div col_div (q4_row_div_col_stride col_stride) (q4_col one)",
        one = 1,
        q4_row_div_col_stride = Q4NX_PER_BLOCK_ROW_SIZE // Q4NX_PER_BLOCK_COL_STRIDE,
        col_stride = Q4NX_PER_BLOCK_COL_STRIDE
    ).contiguous()

    q4nx_mxfp4_data = einops.rearrange(
        q4nx_mxfp4_data,
        "batch row_div col_div (q4_row_div_col_stride q4_col) (col_stride one) -> \
           batch row_div col_div (q4_row_div_col_stride col_stride) (q4_col one)",
           one = 1,
            q4_row_div_col_stride = Q4NX_PER_BLOCK_ROW_SIZE // Q4NX_PER_BLOCK_COL_STRIDE,           
           col_stride = Q4NX_PER_BLOCK_COL_STRIDE
        ).contiguous()
    
    q4nx_mxfp4_scale = einops.rearrange(
        q4nx_mxfp4_scale,
        "batch row_div col_div q4_row q4_col -> batch (row_div q4_row) (col_div q4_col)",
        batch = q4nx_mxfp4_scale.shape[0],
        q4_row = Q4NX_PER_BLOCK_ROW_SIZE,
        q4_col = Q4NX_PER_BLOCK_COL_SIZE//32,
    ).contiguous()
    
    q4nx_mxfp4_data = einops.rearrange(
        q4nx_mxfp4_data,
        pattern="batch row_div col_div q4_row q4_col -> batch (row_div q4_row) (col_div q4_col)",
        batch = q4nx_mxfp4_data.shape[0],
        q4_row = Q4NX_PER_BLOCK_ROW_SIZE,
        q4_col = Q4NX_PER_BLOCK_COL_SIZE//2,
    ).contiguous()
    
    q4nx_mxfp4_data = einops.rearrange(
        q4nx_mxfp4_data,
        "batch rows (cols_div_group_size group_size) -> batch rows cols_div_group_size group_size",
        group_size=16 # 32 4bit 
    )
    q4nx_mxfp4_scale = q4nx_mxfp4_scale.reshape(q4nx_mxfp4_scale.shape + (1,))
    dequant_weight_padded = dequant_mxfp4_q4nx_data_format(q4nx_mxfp4_scale, q4nx_mxfp4_data)
    
    # The dequant_mxfp4_q4nx_data_format function returns a tensor with shape like [32, 2880, 2880]
    # where the last dimension is already unpacked (from 1440 bytes -> 2880 values)
    # We need to reshape it to match the original reference shape
    
    # Calculate padding needed for dimension -2 (rows)
    row_pad = (Q4NX_PER_BLOCK_ROW_SIZE - (reference.shape[-2] % Q4NX_PER_BLOCK_ROW_SIZE)) % Q4NX_PER_BLOCK_ROW_SIZE
    # Calculate padding needed for dimension -1 (cols)  
    col_pad = (Q4NX_PER_BLOCK_COL_SIZE - (reference.shape[-1] % Q4NX_PER_BLOCK_COL_SIZE)) % Q4NX_PER_BLOCK_COL_SIZE
    
    # Apply padding: F.pad takes (left, right, top, bottom, front, back, ...)
    # For last two dimensions: (left_pad_dim-1, right_pad_dim-1, left_pad_dim-2, right_pad_dim-2)
    reference_padded = F.pad(reference, (0, col_pad, 0, row_pad), "constant", 0)
    
    print_tensor_erros(dequant_weight_padded.flatten().to(torch.float32), reference_padded.flatten().to(torch.float32))


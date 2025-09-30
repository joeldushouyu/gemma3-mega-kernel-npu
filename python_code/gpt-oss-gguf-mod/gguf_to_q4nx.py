#!/usr/bin/env python3
from sympy import Q
from q4nx_util import convert_q41_gguf_data_to_q4nx_format, convert_q80_gguf_data_to_q4nx_format, convert_mxfp4_gguf_data_to_q4nx_format, dequant_mxfp4_q4nx_data_format, dequant_q41_q4nx_data_format, dequant_q80_q4nx_data_format
import torch.nn.functional as F
from q4nx_util import compare_q4nx_q80_with_ref, compare_q4nx_q41_with_ref, compare_q4nx_mxfp4_with_ref, padd_mxfp4_scale, concat_mxfp4_scale_data
from util import print_tensor_erros, print_numpy_errors, restore_from_repack_mxfp4_direct, replace_blk_with_model_layer_str
import gguf
from gguf import ReaderTensor
from gguf.quants import dequantize
from gguf.gguf_reader import GGUFReader
from gguf.constants import GGMLQuantizationType
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
gguf_packet_dir = str(Path(__file__).parent.parent.parent.joinpath(
    "SubModules/llama.cpp/gguf-py"))
print("gguf_packet_dir:", gguf_packet_dir)
sys.path.insert(0, gguf_packet_dir)

import einops
def convert_q4_gguf_to_safetensor(gguf_file_path: str,
                                  data_to_save: dict,
                                  data_dequant_debug: dict | None,
                                  TOTAL_LAYER: int,
                                  Q4NX_BLOCK_ROW_SIZE=32,
                                  Q4NX_BLOCK_COL_SIZE=256,
                                  Q4NX_BLOCK_COL_STRIDE=16
                                  ) -> None:

    gguf_reader = GGUFReader(gguf_file_path)

    # create a map of gguf_tensor_name with gguf.ReaderTensors
    gguf_name_to_tensors: dict[str, ReaderTensor] = {}

    for tensors in gguf_reader.tensors:
        gguf_name_to_tensors[tensors.name] = tensors

    # All those should be dequantized completely
    lm_head_weight_read_tensor = gguf_name_to_tensors["output.weight"]
    assert lm_head_weight_read_tensor.tensor_type.name == "Q8_0"
    # For now, let us store in Q8_0
    lm_head_scales_reshaped, lm_head_data_reshaped = convert_q80_gguf_data_to_q4nx_format(lm_head_weight_read_tensor.data,
                                                                                          Q4NX_BLOCK_ROW_SIZE, Q4NX_BLOCK_COL_SIZE,
                                                                                          Q4NX_BLOCK_COL_STRIDE)

    data_to_save["lm_head.weight_scale"] = lm_head_scales_reshaped
    data_to_save["lm_head.weight_blocks"] = lm_head_data_reshaped

    model_weight_reade_tensor: ReaderTensor = gguf_name_to_tensors["output_norm.weight"]
    data_to_save["model.norm.weight"] = torch.tensor(dequantize(
        model_weight_reade_tensor.data, model_weight_reade_tensor.tensor_type), dtype=torch.bfloat16)

    model_embed_token_weight = gguf_name_to_tensors["token_embd.weight"]
    data_to_save["model.embed_tokens.weight"] = torch.tensor(dequantize(
        model_embed_token_weight.data, model_embed_token_weight.tensor_type), dtype=torch.bfloat16)

    if data_dequant_debug is not None:
        data_dequant_debug["lm_head.weight"] = torch.tensor(dequantize(
            lm_head_weight_read_tensor.data, lm_head_weight_read_tensor.tensor_type), dtype=torch.bfloat16)

    for layer_idx in range(1):
        gguf_layer_common_prefix = f"blk.{layer_idx}."
        safetensor_layer_common_prefix = f"model.layers.{layer_idx}."

        mlp_expert_down_proj_bias = gguf_name_to_tensors[gguf_layer_common_prefix +
                                                         f"ffn_down_exps.bias"]
        data_to_save[safetensor_layer_common_prefix + "mlp.experts.down_proj_bias"] = torch.tensor(
            dequantize(mlp_expert_down_proj_bias.data, mlp_expert_down_proj_bias.tensor_type), dtype=torch.bfloat16)

        mlp_exp_up_bias = gguf_name_to_tensors[gguf_layer_common_prefix +
                                               "ffn_up_exps.bias"]
        mlp_exp_gate_bas = gguf_name_to_tensors[gguf_layer_common_prefix +
                                                "ffn_gate_exps.bias"]
        mlp_exp_up_bias = torch.tensor(dequantize(
            mlp_exp_up_bias.data, mlp_exp_up_bias.tensor_type), dtype=torch.bfloat16)
        mlp_exp_gate_bas = torch.tensor(dequantize(
            mlp_exp_gate_bas.data, mlp_exp_gate_bas.tensor_type), dtype=torch.bfloat16)

        data_to_save[safetensor_layer_common_prefix +
                     "mlp.experts.gate_proj_bias"] = mlp_exp_gate_bas
        data_to_save[safetensor_layer_common_prefix +
                     "mlp.experts.up_proj_bias"] = mlp_exp_up_bias

        attn_sinks = gguf_name_to_tensors[gguf_layer_common_prefix +
                                          "attn_sinks.weight"]
        attn_sinks = torch.tensor(dequantize(
            attn_sinks.data, attn_sinks.tensor_type), dtype=torch.bfloat16)
        data_to_save[safetensor_layer_common_prefix +
                     "self_attn.sinks"] = attn_sinks

        mlp_router_bias = gguf_name_to_tensors[gguf_layer_common_prefix +
                                               "ffn_gate_inp.bias"]
        mlp_router_bias = torch.tensor(dequantize(
            mlp_router_bias.data, mlp_router_bias.tensor_type), dtype=torch.bfloat16)
        data_to_save[safetensor_layer_common_prefix +
                     "mlp.router.bias"] = mlp_router_bias

        mlp_router_weight = gguf_name_to_tensors[gguf_layer_common_prefix +
                                                 "ffn_gate_inp.weight"]
        mlp_router_weight = torch.tensor(dequantize(
            mlp_router_weight.data, mlp_router_weight.tensor_type), dtype=torch.bfloat16)
        # At this point, do a data reorder, mlp_router_weight is [num_expert, hidden_size] in row_major order
        
        # reorder it to [num_expert/Q4NX_BLOCK_COL_STRIDE, Q4NX_BLOCK_COL_STRIDE, hidden_size] in column major order
        mlp_router_weight = einops.rearrange(
            mlp_router_weight,
            "(num_expert_div_Q4NX_BLOCK_COL_STRIDE Q4NX_BLOCK_COL_STRIDE ) (hidden_size one) -> (num_expert_div_Q4NX_BLOCK_COL_STRIDE  hidden_size) (Q4NX_BLOCK_COL_STRIDE one)",
            Q4NX_BLOCK_COL_STRIDE=Q4NX_BLOCK_COL_STRIDE,
            one=1
        ).contiguous()
        # This is still in row-major order, change to column major order
        
        
        
        data_to_save[safetensor_layer_common_prefix +
                     "mlp.router.weight"] = mlp_router_weight

        q_proj_weight = gguf_name_to_tensors[gguf_layer_common_prefix + "attn_q.weight"]
        q_proj_weight_scale, q_proj_weight_bias, q_proj_weight_blocks = convert_q41_gguf_data_to_q4nx_format(
            q_proj_weight.data, Q4NX_BLOCK_ROW_SIZE, Q4NX_BLOCK_COL_SIZE, Q4NX_BLOCK_COL_STRIDE
        )
        data_to_save[safetensor_layer_common_prefix +
                     "self_attn.q_proj.weight_scale"] = q_proj_weight_scale
        data_to_save[safetensor_layer_common_prefix +
                     "self_attn.q_proj.weight_bias"] = q_proj_weight_bias
        data_to_save[safetensor_layer_common_prefix +
                     "self_attn.q_proj.weight_blocks"] = q_proj_weight_blocks

        q_proj_bias = gguf_name_to_tensors[gguf_layer_common_prefix + "attn_q.bias"]
        q_proj_bias = torch.tensor(dequantize(
            q_proj_bias.data, q_proj_bias.tensor_type), dtype=torch.bfloat16)
        data_to_save[safetensor_layer_common_prefix +
                     "self_attn.q_proj.bias"] = q_proj_bias

        k_proj_weight = gguf_name_to_tensors[gguf_layer_common_prefix + "attn_k.weight"]
        k_proj_weight_scale, k_proj_weight_bias, k_proj_weight_blocks = convert_q41_gguf_data_to_q4nx_format(
            k_proj_weight.data, Q4NX_BLOCK_ROW_SIZE, Q4NX_BLOCK_COL_SIZE, Q4NX_BLOCK_COL_STRIDE
        )
        data_to_save[safetensor_layer_common_prefix +
                     "self_attn.k_proj.weight_scale"] = k_proj_weight_scale
        data_to_save[safetensor_layer_common_prefix +
                     "self_attn.k_proj.weight_bias"] = k_proj_weight_bias
        data_to_save[safetensor_layer_common_prefix +
                     "self_attn.k_proj.weight_blocks"] = k_proj_weight_blocks

        k_proj_bias = gguf_name_to_tensors[gguf_layer_common_prefix + "attn_k.bias"]
        k_proj_bias = torch.tensor(dequantize(
            k_proj_bias.data, k_proj_bias.tensor_type), dtype=torch.bfloat16)
        data_to_save[safetensor_layer_common_prefix +
                     "self_attn.k_proj.bias"] = k_proj_bias

        v_proj_weight = gguf_name_to_tensors[gguf_layer_common_prefix + "attn_v.weight"]
        v_proj_weight_scale, v_proj_weight_bias, v_proj_weight_blocks = convert_q41_gguf_data_to_q4nx_format(
            v_proj_weight.data, Q4NX_BLOCK_ROW_SIZE, Q4NX_BLOCK_COL_SIZE, Q4NX_BLOCK_COL_STRIDE
        )
        data_to_save[safetensor_layer_common_prefix +
                     "self_attn.v_proj.weight_scale"] = v_proj_weight_scale
        data_to_save[safetensor_layer_common_prefix +
                     "self_attn.v_proj.weight_bias"] = v_proj_weight_bias
        data_to_save[safetensor_layer_common_prefix +
                     "self_attn.v_proj.weight_blocks"] = v_proj_weight_blocks

        v_proj_bias = gguf_name_to_tensors[gguf_layer_common_prefix + "attn_v.bias"]
        v_proj_bias = torch.tensor(dequantize(
            v_proj_bias.data, v_proj_bias.tensor_type), dtype=torch.bfloat16)
        data_to_save[safetensor_layer_common_prefix +
                     "self_attn.v_proj.bias"] = v_proj_bias

        output_proj_weight = gguf_name_to_tensors[gguf_layer_common_prefix +
                                                  "attn_output.weight"]
        output_proj_weight_scale, output_proj_weight_bias, output_proj_weight_blocks = convert_q41_gguf_data_to_q4nx_format(
            output_proj_weight.data, Q4NX_BLOCK_ROW_SIZE, Q4NX_BLOCK_COL_SIZE, Q4NX_BLOCK_COL_STRIDE
        )
        data_to_save[safetensor_layer_common_prefix +
                     "self_attn.o_proj.weight_scale"] = output_proj_weight_scale
        data_to_save[safetensor_layer_common_prefix +
                     "self_attn.o_proj.weight_bias"] = output_proj_weight_bias
        data_to_save[safetensor_layer_common_prefix +
                     "self_attn.o_proj.weight_blocks"] = output_proj_weight_blocks

        if data_dequant_debug is not None:
            data_dequant_debug[safetensor_layer_common_prefix + "self_attn.q_proj.weight"] = torch.tensor(
                dequantize(q_proj_weight.data, q_proj_weight.tensor_type), dtype=torch.bfloat16)
            data_dequant_debug[safetensor_layer_common_prefix + "self_attn.k_proj.weight"] = torch.tensor(
                dequantize(k_proj_weight.data, k_proj_weight.tensor_type), dtype=torch.bfloat16)
            data_dequant_debug[safetensor_layer_common_prefix + "self_attn.v_proj.weight"] = torch.tensor(
                dequantize(v_proj_weight.data, v_proj_weight.tensor_type), dtype=torch.bfloat16)
            data_dequant_debug[safetensor_layer_common_prefix + "self_attn.o_proj.weight"] = torch.tensor(
                dequantize(output_proj_weight.data, output_proj_weight.tensor_type), dtype=torch.bfloat16)

        output_proj_bias = gguf_name_to_tensors[gguf_layer_common_prefix +
                                                "attn_output.bias"]
        output_proj_bias = torch.tensor(dequantize(
            output_proj_bias.data, output_proj_bias.tensor_type), dtype=torch.bfloat16)
        data_to_save[safetensor_layer_common_prefix +
                     "self_attn.o_proj.bias"] = output_proj_bias

        post_attention_norm_weight = gguf_name_to_tensors[gguf_layer_common_prefix +
                                                          "post_attention_norm.weight"]
        post_attention_norm_weight = torch.tensor(dequantize(
            post_attention_norm_weight.data, post_attention_norm_weight.tensor_type), dtype=torch.bfloat16)
        data_to_save[safetensor_layer_common_prefix +
                     "post_attention_layernorm.weight"] = post_attention_norm_weight

        # Add input layernorm
        input_norm_weight = gguf_name_to_tensors[gguf_layer_common_prefix +
                                                 "attn_norm.weight"]
        input_norm_weight = torch.tensor(dequantize(
            input_norm_weight.data, input_norm_weight.tensor_type), dtype=torch.bfloat16)
        data_to_save[safetensor_layer_common_prefix +
                     "input_layernorm.weight"] = input_norm_weight

        # ffn_down_exps_weights is already being rearranged into GGML type of MXFP4
        # The data is stored as (32, 2880, 1530) where 1530 = 90 * 17 bytes
        ffn_down_exps_weights = gguf_name_to_tensors[gguf_layer_common_prefix +
                                                     "ffn_down_exps.weight"]
        ffn_down_scale, ffn_down_weight = convert_mxfp4_gguf_data_to_q4nx_format(
            ffn_down_exps_weights.data, Q4NX_BLOCK_ROW_SIZE, Q4NX_BLOCK_COL_SIZE, Q4NX_BLOCK_COL_STRIDE)
        data_to_save[safetensor_layer_common_prefix +
                     "mlp.experts.down_proj_blocks"] = ffn_down_weight
        data_to_save[safetensor_layer_common_prefix +
                     "mlp.experts.down_proj_scales"] = ffn_down_scale

        ffn_up_exps_weights = gguf_name_to_tensors[gguf_layer_common_prefix +
                                                   "ffn_up_exps.weight"]
        assert ffn_up_exps_weights.tensor_type.name == "MXFP4", f"Expected MXFP4, got {ffn_up_exps_weights.tensor_type.name}"
        ffn_up_scale, ffn_up_weight = convert_mxfp4_gguf_data_to_q4nx_format(
            ffn_up_exps_weights.data, Q4NX_BLOCK_ROW_SIZE, Q4NX_BLOCK_COL_SIZE, Q4NX_BLOCK_COL_STRIDE)

        ffn_gate_exps_weights = gguf_name_to_tensors[gguf_layer_common_prefix +
                                                     "ffn_gate_exps.weight"]
        assert ffn_gate_exps_weights.tensor_type.name == "MXFP4", f"Expected MXFP4, got {ffn_gate_exps_weights.tensor_type.name}"
        ffn_gate_scale, ffn_gate_weight = convert_mxfp4_gguf_data_to_q4nx_format(
            ffn_gate_exps_weights.data, Q4NX_BLOCK_ROW_SIZE, Q4NX_BLOCK_COL_SIZE, Q4NX_BLOCK_COL_STRIDE)

        data_to_save[safetensor_layer_common_prefix +
                     "mlp.experts.gate_proj_scales"] = ffn_gate_scale
        data_to_save[safetensor_layer_common_prefix +
                     "mlp.experts.up_proj_scales"] = ffn_up_scale
        data_to_save[safetensor_layer_common_prefix +
                     "mlp.experts.gate_proj_blocks"] = ffn_gate_weight
        data_to_save[safetensor_layer_common_prefix +
                     "mlp.experts.up_proj_blocks"] = ffn_up_weight

    
        #to ensure the data matches the sizes of q4nx for q4_1, we now
        # 1. padd extra 3 byte to each MXFP4 scale data_value
        # 2. combine scale and data_blocks into one single safetensors
        ffn_down_scale_padded = padd_mxfp4_scale(ffn_down_scale)
        ffn_up_scale_padded = padd_mxfp4_scale(ffn_up_scale)
        ffn_gate_scale_padded = padd_mxfp4_scale(ffn_gate_scale)
        
        # data_to_save[safetensor_layer_common_prefix +
        #                      "mlp.experts.down_proj_comb"] =  
        
        
        

        
        
        if data_dequant_debug is not None:
            data_dequant_debug[safetensor_layer_common_prefix + "mlp.experts.down_proj"] = torch.tensor(
                dequantize(ffn_down_exps_weights.data, ffn_down_exps_weights.tensor_type), dtype=torch.bfloat16)
            data_dequant_debug[safetensor_layer_common_prefix + "mlp.experts.up_proj"] = torch.tensor(
                dequantize(ffn_up_exps_weights.data, ffn_up_exps_weights.tensor_type), dtype=torch.bfloat16)
            data_dequant_debug[safetensor_layer_common_prefix + "mlp.experts.gate_proj"] = torch.tensor(
                dequantize(ffn_gate_exps_weights.data, ffn_gate_exps_weights.tensor_type), dtype=torch.bfloat16)


def verify_q4nx_data(gguf_file_path, data_equant_debug: dict[str, torch.Tensor], Q4NX_PER_BLOCK_ROW_SIZE: int, Q4NX_PER_BLOCK_COL_SIZE, Q4NX_PER_BLOCK_COL_STRIDE):

    loaded_tensors = load_file(gguf_file_path)
    # first, dequant lm_head data
    lm_head_scale = loaded_tensors["lm_head.weight_scale"]
    lm_head_data = loaded_tensors["lm_head.weight_blocks"]
    ref_lm_head_weight = data_equant_debug["lm_head.weight"]

    compare_q4nx_q80_with_ref(
        lm_head_scale, lm_head_data, ref_lm_head_weight, Q4NX_PER_BLOCK_ROW_SIZE, Q4NX_PER_BLOCK_COL_SIZE, Q4NX_PER_BLOCK_COL_STRIDE
    )

    TOTAL_LAYER = 1  # only for now 24
    for layer_idx in range(TOTAL_LAYER):  # for now
        safetensor_layer_common_prefix = f"model.layers.{layer_idx}."
        print(f"INFO: LAYER_ID={layer_idx}")

        q_proj_scale = loaded_tensors[safetensor_layer_common_prefix +
                                      "self_attn.q_proj.weight_scale"]
        q_proj_biase = loaded_tensors[safetensor_layer_common_prefix +
                                      "self_attn.q_proj.weight_bias"]
        q_proj_block = loaded_tensors[safetensor_layer_common_prefix +
                                      "self_attn.q_proj.weight_blocks"]
        q_ref = data_dequant_debug[safetensor_layer_common_prefix +
                                   "self_attn.q_proj.weight"]
        compare_q4nx_q41_with_ref(
            q_proj_scale, q_proj_biase, q_proj_block, q_ref, Q4NX_PER_BLOCK_ROW_SIZE, Q4NX_PER_BLOCK_COL_SIZE, Q4NX_PER_BLOCK_COL_STRIDE
        )

        k_proj_scale = loaded_tensors[safetensor_layer_common_prefix +
                                      "self_attn.k_proj.weight_scale"]
        k_proj_biase = loaded_tensors[safetensor_layer_common_prefix +
                                      "self_attn.k_proj.weight_bias"]
        k_proj_block = loaded_tensors[safetensor_layer_common_prefix +
                                      "self_attn.k_proj.weight_blocks"]
        k_ref = data_dequant_debug[safetensor_layer_common_prefix +
                                   "self_attn.k_proj.weight"]
        compare_q4nx_q41_with_ref(
            k_proj_scale, k_proj_biase, k_proj_block, k_ref, Q4NX_PER_BLOCK_ROW_SIZE, Q4NX_PER_BLOCK_COL_SIZE, Q4NX_PER_BLOCK_COL_STRIDE
        )

        v_proj_scale = loaded_tensors[safetensor_layer_common_prefix +
                                      "self_attn.v_proj.weight_scale"]
        v_proj_biase = loaded_tensors[safetensor_layer_common_prefix +
                                      "self_attn.v_proj.weight_bias"]
        v_proj_block = loaded_tensors[safetensor_layer_common_prefix +
                                      "self_attn.v_proj.weight_blocks"]
        v_ref = data_dequant_debug[safetensor_layer_common_prefix +
                                   "self_attn.v_proj.weight"]
        compare_q4nx_q41_with_ref(
            v_proj_scale, v_proj_biase, v_proj_block, v_ref, Q4NX_PER_BLOCK_ROW_SIZE, Q4NX_PER_BLOCK_COL_SIZE, Q4NX_PER_BLOCK_COL_STRIDE
        )

        output_proj_scale = loaded_tensors[safetensor_layer_common_prefix +
                                           "self_attn.o_proj.weight_scale"]
        output_proj_biase = loaded_tensors[safetensor_layer_common_prefix +
                                           "self_attn.o_proj.weight_bias"]
        output_proj_block = loaded_tensors[safetensor_layer_common_prefix +
                                           "self_attn.o_proj.weight_blocks"]
        output_ref = data_dequant_debug[safetensor_layer_common_prefix +
                                        "self_attn.o_proj.weight"]
        compare_q4nx_q41_with_ref(
            output_proj_scale, output_proj_biase, output_proj_block, output_ref, Q4NX_PER_BLOCK_ROW_SIZE, Q4NX_PER_BLOCK_COL_SIZE, Q4NX_PER_BLOCK_COL_STRIDE
        )

        down_proj_block = loaded_tensors[safetensor_layer_common_prefix +
                                         "mlp.experts.down_proj_blocks"]
        down_proj_scale = loaded_tensors[safetensor_layer_common_prefix +
                                         "mlp.experts.down_proj_scales"]
        down_ref = data_equant_debug[safetensor_layer_common_prefix +
                                     "mlp.experts.down_proj"]
        compare_q4nx_mxfp4_with_ref(
            down_proj_scale, down_proj_block, down_ref, Q4NX_PER_BLOCK_ROW_SIZE, Q4NX_PER_BLOCK_COL_SIZE, Q4NX_PER_BLOCK_COL_STRIDE
        )

        up_proj_block = loaded_tensors[safetensor_layer_common_prefix +
                                       "mlp.experts.up_proj_blocks"]
        up_proj_scale = loaded_tensors[safetensor_layer_common_prefix +
                                       "mlp.experts.up_proj_scales"]
        up_ref = data_equant_debug[safetensor_layer_common_prefix +
                                   "mlp.experts.up_proj"]
        compare_q4nx_mxfp4_with_ref(
            up_proj_scale, up_proj_block, up_ref, Q4NX_PER_BLOCK_ROW_SIZE, Q4NX_PER_BLOCK_COL_SIZE, Q4NX_PER_BLOCK_COL_STRIDE
        )

        gate_proj_block = loaded_tensors[safetensor_layer_common_prefix +
                                         "mlp.experts.gate_proj_blocks"]
        gate_proj_scale = loaded_tensors[safetensor_layer_common_prefix +
                                         "mlp.experts.gate_proj_scales"]
        gate_ref = data_equant_debug[safetensor_layer_common_prefix +
                                     "mlp.experts.gate_proj"]
        compare_q4nx_mxfp4_with_ref(
            gate_proj_scale, gate_proj_block, gate_ref, Q4NX_PER_BLOCK_ROW_SIZE, Q4NX_PER_BLOCK_COL_SIZE, Q4NX_PER_BLOCK_COL_STRIDE
        )


if __name__ == '__main__':
    # if len(sys.argv) < 2:
    #     logger.info("Usage: reader.py <path_to_gguf_file>")
    #     sys.exit(1)
    # gguf_file_path = sys.argv[1]

    gguf_file_path = "/home/shouyud/gpt-oss-mega-kernel/gpt-oss/unsloth_q4_1_gguf/gpt-oss-20b-Q4_1.gguf"


    safetensor_reference = "/home/shouyud/gpt-oss-mega-kernel/python_code/gpt-oss-evaluation/hf_cache/models--openai--gpt-oss-20b/snapshots/d666cf3b67006cf8227666739edf25164aaffdeb/model-00000-of-00002.safetensors"
    safetensor_reference1 = "/home/shouyud/gpt-oss-mega-kernel/python_code/gpt-oss-evaluation/hf_cache/models--openai--gpt-oss-20b/snapshots/d666cf3b67006cf8227666739edf25164aaffdeb/model-00001-of-00002.safetensors"
    safetensor_reference2 = "/home/shouyud/gpt-oss-mega-kernel/python_code/gpt-oss-evaluation/hf_cache/models--openai--gpt-oss-20b/snapshots/d666cf3b67006cf8227666739edf25164aaffdeb/model-00002-of-00002.safetensors"
    safetensors_data = {}
    safetensors_data.update(load_file(safetensor_reference))
    safetensors_data.update(load_file(safetensor_reference1))
    safetensors_data.update(load_file(safetensor_reference2))

    
    Q4NX_BLOCK_ROW_SIZE = 32
    Q4NX_BLOCK_COL_SIZE = 256
    Q4NX_BLOCK_COL_STRIDE = 16

    data_to_save = {}
    data_dequant_debug = {}
    convert_q4_gguf_to_safetensor(gguf_file_path, data_to_save=data_to_save, data_dequant_debug=data_dequant_debug,
                                  TOTAL_LAYER=24,Q4NX_BLOCK_ROW_SIZE=Q4NX_BLOCK_ROW_SIZE, Q4NX_BLOCK_COL_SIZE=Q4NX_BLOCK_COL_SIZE,
                                  Q4NX_BLOCK_COL_STRIDE=Q4NX_BLOCK_COL_STRIDE
                                  )


    # do a very debug operation
    # NOTE DEBUGL very
    # replace token_embd.weight in data_to_Save with the safetensors_data
    data_to_save["model.embed_tokens.weight"] = safetensors_data["model.embed_tokens.weight"].clone()

    # also save a json file
    output_safetensor_filename = "model-q4nx.safetensors"
    # Ensure all tensors are contiguous before saving
    data_to_save_contiguous = {k: v.contiguous()
                               for k, v in data_to_save.items()}
    save_file(data_to_save_contiguous, output_safetensor_filename)


    if data_dequant_debug is not None:
        verify_q4nx_data(output_safetensor_filename,
                        data_dequant_debug, Q4NX_BLOCK_ROW_SIZE, Q4NX_BLOCK_COL_SIZE, Q4NX_BLOCK_COL_STRIDE)


        data_dequant_debug = {k:v.contiguous() for k,v in data_dequant_debug.items()}
        save_file(data_dequant_debug, "model-q4nx-dequant-ref.safetensors")
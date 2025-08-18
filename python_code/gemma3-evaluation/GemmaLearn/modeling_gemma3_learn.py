import copy
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Optional, Union

from numpy._typing._array_like import NDArray
import torch
import torch.nn as nn
import copy
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_masks_for_generate, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import ModelOutput, auto_docstring, can_return_tuple, is_torchdynamo_compiling, logging
from transformers.utils.deprecation import deprecate_kwarg
from transformers.models.auto import AutoModel
from transformers.models.gemma3.configuration_gemma3 import Gemma3Config, Gemma3TextConfig

from .modeling_siglip_learn import SiglipVisionModelLearn

from .common_module import assert_tensor_same_float, tensor_to_numpy, numpy_to_tensor, gelu_element_wise_np, nn_linear_numpy
from transformers.modeling_layers import GradientCheckpointingLayer
#TODO: list of replacement
from transformers.models.gemma3 import Gemma3Model, Gemma3PreTrainedModel
from transformers.models.gemma3.modeling_gemma3 import Gemma3CausalLMOutputWithPast, token_type_ids_mask_function, Gemma3MultiModalProjector,Gemma3ModelOutputWithPast
from transformers.masking_utils import create_masks_for_generate
from transformers.models.gemma3.modeling_gemma3 import eager_attention_forward, Gemma3TextScaledWordEmbedding, Gemma3Attention, Gemma3MLP, Gemma3RMSNorm, Gemma3RotaryEmbedding,apply_rotary_pos_emb

import numpy as np
import numpy.typing as npt
from typing import Tuple

from scipy.special import softmax

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


SANITY_CHECK= False
def get_relativeL2(y: np.ndarray, y_ref: np.ndarray) -> float:
    rmse = np.sqrt(np.mean((y - y_ref) ** 2))
    ref_norm = np.sqrt(np.mean(y_ref ** 2))
    return rmse / ref_norm

def get_relativeL1(y: np.ndarray, y_ref: np.ndarray) -> float:
    l1 = np.sum(np.abs(y - y_ref))
    ref_sum = np.sum(np.abs(y_ref))
    return l1 / ref_sum

def get_rmse(y: np.ndarray, y_ref: np.ndarray) -> float:
    return np.sqrt(np.mean((y - y_ref) ** 2)    ) 

def get_cosine_similarity(y: np.ndarray, y_ref: np.ndarray) -> float:
    # squize to 1D array
    y = y.reshape(-1)
    y_ref = y_ref.reshape(-1)
    dot_product = np.dot(y, y_ref)
    norm_y = np.linalg.norm(y)
    norm_y_ref = np.linalg.norm(y_ref)
    
    return dot_product / (norm_y * norm_y_ref)








class Gemma3RotaryEmbeddingLearn(nn.Module):
    def __init__(self, config: Gemma3TextConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

        
        # some cache mechanism
        self.inv_freq_cache = inv_freq
        self.prev_cos = None
        self.prev_sin = None
        self.input_cache_debug = None
    
    def np_forward(self, x, position_ids, cache=False):
        #[batch_size, seq_len, d_model/hidden_ssize]

        x_np = tensor_to_numpy(x)
        batch_size = x_np.shape[0]
        seq_len  = x_np.shape[1]
        hidden_size = x_np.shape[2]
        
        d_k_size_div_2 = self.inv_freq.shape[0]
        assert d_k_size_div_2 == getattr(self.config, "head_dim") // 2
        
        #[batch, seq_len]
        position_ids_np = tensor_to_numpy(position_ids)
        assert position_ids_np.shape[1] == seq_len
        assert seq_len <= self.original_max_seq_len  
        
        # self.inv_freq is some inverse multiple of size[d_k/2], d_k is the attention head size in multi-head attention
        inv_freq_np = tensor_to_numpy(self.inv_freq) # [d_k_size_div_2]            
        inv_freq_expand = inv_freq_np.reshape(1, d_k_size_div_2) # add a axis to be [1. d_k//2]   
        #NOTE: rope stays same for all batch (independent of batch, depend on token_id and d_k size )
        if cache is False:
            # prefill stage, calculate the whole rope size for current x_np
            #[seq_len, 1]
            position_ids_one_batch =  position_ids_np[-1, :].reshape(seq_len, 1) # should be same for all batch                
            """
            [m0_\theta_1, m0_\theta_2, .... m0\theta_d_k//2  ],
            [m1_\theta_1, m1_\theta_2, .... m1\theta_d_k//2  ],  .. repeat lines for seq_len for all         
            """
            freqs = np.matmul( position_ids_one_batch, inv_freq_expand, dtype=np.float32 ) # seq_len , d_k//2]
        
        
            # The matrix for emb is
            """
            [m0_\theta_1, m0_\theta_2, .... m0\theta_d_k//2, m0_\theta_1, m0_theta_2, ..... m0\theta_d_k///2 ],
            [m1_\theta_1, m1_\theta_2, .... m1\theta_d_k//2  ],  .. repeat lines for seq_len for all         
        
            """
            #[ seq_len, d_k]        
            emb = np.concat((freqs, freqs), axis=1)
            cos = np.cos(emb)* self.attention_scaling
            sin = np.sin(emb) * self.attention_scaling
            
            # broadcast to all batch, since shared for all batch
            
            cos = np.broadcast_to(  cos, (batch_size, seq_len, d_k_size_div_2*2) )
            sin = np.broadcast_to(sin,  (batch_size, seq_len, d_k_size_div_2*2))
            
            self.prev_cos = cos
            self.prev_sin = sin
            return numpy_to_tensor(cos, dtype=x.dtype), numpy_to_tensor(sin, dtype=x.dtype) 
        
        else:
            # only do it for the new token_id
            new_token_size = x.shape[1] -  self.prev_cos.shape[1]
            
            position_ids_one_batch = position_ids_np[0][new_token_size].reshape(1,1) # same token index for all batch (small seq_len is being padded)
            freqs_new_token =np.matmul( position_ids_one_batch , inv_freq_expand, dtype=np.float32 ) #[ 1, d_k//2]

            """
                [M_new_token\theta1, M_new_token\theta2, ... M_new_token_\theta_d_k//2, M_new_token\theta1, M_new_token\theta2, ... M_new_token_\theta_d_k//2,  ]
            """
            #[1. d_k]
            emb_new_token = np.concat((freqs_new_token, freqs_new_token), axis=1)
            new_token_cos = np.cos(emb_new_token) *self.attention_scaling
            new_token_sin = np.sin(emb_new_token) * self.attention_scaling
            
            
            batch_new_token_cos = np.broadcast_to(new_token_cos, (batch_size, new_token_size, d_k_size_div_2*2) ) # same for all batch
            batch_new_token_sin = np.broadcast_to(new_token_sin, (batch_size, new_token_size,  d_k_size_div_2*2  ))
            
            # concat it to the cache
            cos_final = np.concat((self.prev_cos, batch_new_token_cos), axis=1 )
            sin_final = np.concat((self.prev_sin, batch_new_token_sin), axis=1)
            
            self.prev_cos = cos_final
            self.prev_sin = sin_final            
            return  numpy_to_tensor(cos_final, dtype=x.dtype), numpy_to_tensor(sin_final, dtype=x.dtype)
                
        
    def forward(self, x, position_ids):
        np_cos, np_sin =self.np_forward(x, position_ids)
        if SANITY_CHECK:
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
            position_ids_expanded = position_ids[:, None, :].float()

            device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
            with torch.autocast(device_type=device_type, enabled=False):  # Force float32
                freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos() * self.attention_scaling
                sin = emb.sin() * self.attention_scaling

            tensor_cos = cos.to(dtype=x.dtype) 
            tensor_sin = sin.to(dtype=x.dtype)

            assert assert_tensor_same_float(tensor_cos,np_cos)
            assert assert_tensor_same_float(tensor_sin, np_sin)
            #return tensor_cos, tensor_sin
        return np_cos, np_sin

def rotate_half_np(x):
    """
    Rotates half the hidden dims of the input.

    This function splits the last dimension of the input array in half, negates the second half,
    and concatenates it with the first half. This is commonly used in rotary positional embeddings.

    Numerical Example:
    ------------------
    >>> import numpy as np
    >>> x = np.array([[1, 2, 3, 4]])
    >>> rotate_half_np(x)
    array([[-3, -4,  1,  2]])

    In this example, the input [1, 2, 3, 4] is split into [1, 2] and [3, 4].
    The second half is negated to [-3, -4], and then concatenated with the first half to get [-3, -4, 1, 2].
    """

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return np.concat((-x2, x1), axis=-1)



def apply_rotary_pos_emb_np(q:np.ndarray, k:np.ndarray, cos:np.ndarray, sin:np.ndarray, unsqueeze_dim=1):

    # q is [batch, num_attention_head, seq_len, d_k]
    # k is [batch, num_attention_head, seq_len, d_k]
    
    # cos, sine is [batch, seq_len, d_model]
    # but they are same for all batch
    # cos = cos.unsqueeze(unsqueeze_dim)
    # sin = sin.unsqueeze(unsqueeze_dim)
    
    """
    Note: according to the original ROPE paper
    
    The rope for q, k should be     
    $$
    q_embed =  
        \left[
            \begin{array}{c}
                w_{1}    \\
                w_{2}    \\
                w_{3}    \\
                w_{4}     \\ 
                \vdots     \\
                w_{d_k-1}  \\
                w_{d_k}     \\  
            \end{array}
        \right]
        \begin{bmatrix}
                    cos(\theta1) \\
                    cos(\theta1) \\
                    cos(\theta2) \\                        
                    cos(\theta2) \\                        
                    \vdots \\
                    cos(\theta {d_k/2}) \\
                    cos(\theta {d_k/2})                       
        \end{bmatrix}
                +
        \left[
            \begin{array}{c}
                -w_{2}    \\
                w_{1}    \\
                -w_{4}    \\
                w_{3}     \\ 
                \vdots     \\
                -w_{d_k}  \\
                w_{d_k-1}     \\  
            \end{array}
        \right]
        \begin{bmatrix}
                    sin(\theta1) \\
                    sin(\theta1) \\
                    sin(\theta2) \\                        
                    sin(\theta2) \\                        
                    \vdots \\
                    sin(\theta {d_k/2}) \\
                    sin(\theta {d_k/2})                       
        \end{bmatrix}   
    $$
    
    But for inferece, it become
    $$
        \left[
            \begin{array}{c}
                w_{1}    \\
                w_{2}    \\
                w_{3}    \\
                \vdots \\                    
                w_{d_k/2}     \\ 
                \vdots     \\
                w_{d_k-1}  \\
                w_{d_k}     \\  
            \end{array}
        \right]
        \begin{bmatrix}
                    cos(\theta1) \\
                    cos(\theta2) \\
                    cos(\theta3) \\                                                
                    \vdots \\
                    cos{\theta {d_k/2}}\\
                    cos(\theta 1) \\
                    \vdots \\
                    cos(\theta {d_k/2})                       
        \end{bmatrix}
                +
        \left[
            \begin{array}{c}
                -w_{(dk/2)+1}    \\
                -w_{(dk/2)+2}    \\
                -w_{(dk/2)+3}    \\
                \vdots \\                    
                w_{1}     \\ 
                \vdots     \\
                w_{(d_k/2)-1}  \\
                w_{(d_k/2)}     \\  
            \end{array}
        \right]
        \begin{bmatrix}
                    cos(\theta1) \\
                    cos(\theta2) \\
                    cos(\theta3) \\                                                
                    \vdots \\
                    cos{\theta {d_k/2}}\\
                    cos(\theta 1) \\
                    \vdots \\
                    cos(\theta {d_k/2})                       
        \end{bmatrix} 
    
    $$
    """
    
    
    q_embed = (q * cos) + (rotate_half_np(q) * sin)
    k_embed = (k * cos) + (rotate_half_np(k) * sin)
    return q_embed, k_embed






class Gemma3MultiModalProjectorLearn(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()

        self.mm_input_projection_weight = nn.Parameter(
            torch.zeros(config.vision_config.hidden_size, config.text_config.hidden_size)
        )

        self.mm_soft_emb_norm = Gemma3RMSNorm(
            config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps
        )

        self.patches_per_image = int(config.vision_config.image_size // config.vision_config.patch_size)
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.kernel_size)

    def forward(self, vision_outputs: torch.Tensor, tensors_to_save:None|dict=None):
        batch_size, _, seq_length = vision_outputs.shape

        if tensors_to_save is not None:
            tensors_to_save["before_average_pooling"] = vision_outputs.clone().contiguous()
            
        reshaped_vision_outputs = vision_outputs.transpose(1, 2)
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            batch_size, seq_length, self.patches_per_image, self.patches_per_image
        )
        reshaped_vision_outputs = reshaped_vision_outputs.contiguous()

        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)

    
        if tensors_to_save is not None: 
            tensors_to_save["after_average_pooling"] = pooled_vision_outputs.contiguous()
        print(f"shape after average pooling: {pooled_vision_outputs.shape}")
        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

        if tensors_to_save is not None:
            tensors_to_save["after_mm_soft_emb_norm"] = normed_vision_outputs.contiguous()
        print(f"eps of Gemma3RMSNorm is {self.mm_soft_emb_norm.eps}")    
        projected_vision_outputs = torch.matmul(normed_vision_outputs, self.mm_input_projection_weight)
        if tensors_to_save is not None:
            tensors_to_save["after_mm_input_projection"] = projected_vision_outputs.contiguous()
            tensors_to_save["mm_input_projection_weight"] = self.mm_input_projection_weight.contiguous()    
        return projected_vision_outputs.type_as(vision_outputs)



class Gemma3RMSNormMultiHeadLearn(nn.Module):
    #NOTE: has to do it in float32 precision
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

        self.cache = None
    
    def np_forward(self, x):
        #NOTE: follow implementation, will use float32 for computation
        #[batch_Size, head_count,  seq_len, head_dimension(d_k)]
        x_np = tensor_to_numpy(x)
        
        # RMS norm is applied on every hidden_size vector for all seq_len and batch
        batch_size = x_np.shape[0]
        head_count = x_np.shape[1]
        seq_len =x_np.shape[2]
        head_dim = x_np.shape[3]
        
        
        if self.cache is None :
            scalar_multiple = np.full(  (batch_size, head_count,  seq_len, 1), 1/head_dim, dtype=np.float32)
            var_approx_vector = scalar_multiple * np.sum(  np.square(x_np), axis=3, keepdims=True)
            
            # add the epsilon
            var_approx_vector += np.full(  (batch_size, head_count, seq_len, 1), self.eps, dtype=np.float32)
            rqrt_res = np.reciprocal( np.sqrt(var_approx_vector))
            
            # Now broadcast it to hidden_space_size
            #[batch, head_count, seq_len, head_Dim]
            var_approx_vector_rsqrt = np.repeat(rqrt_res, repeats=head_dim, axis=3)
            
            norm_vector = var_approx_vector_rsqrt * x_np
            
            # now is something special
            #[1,  head_dim] - >[batch, head_count, seq_len, head_dim,]
            weight_broadcast = np.broadcast_to(self.weight,  (batch_size, head_count, seq_len, head_dim)) 
            new_hidden_space = norm_vector + norm_vector*weight_broadcast
            
            self.cache = new_hidden_space  
        else:
            # use cache means only do it on the newest_sequence_length for every batch
            # the only does it on the last new_seq of every bach
            new_seq_size = x_np.shape[2] - self.cache.shape[2]
            
            x_np_last_seq = x_np[:, :, -new_seq_size:, :]
            scalar_multiple = np.full(   (batch_size, head_count,  new_seq_size, 1), 1/head_dim, dtype=np.float32)
            var_approx_vector = scalar_multiple   * np.sum(np.square(x_np_last_seq), axis=3, keepdims=True)      

            # add the epsilon
            var_approx_vector += np.full(  (batch_size, head_count , new_seq_size, 1), self.eps, dtype=np.float32)
            rqrt_res = np.reciprocal(np.sqrt(var_approx_vector)) # inverse square root
            
            
            # now broadhcast to hidden space_size
            var_approx_vector_rsqrt = np.repeat(rqrt_res, repeats=head_dim, axis=3)
            norm_vector = var_approx_vector_rsqrt * x_np_last_seq
            
            weight_broadcast = np.broadcast_to(self.weight, [batch_size, head_count , new_seq_size, head_dim])  
            new_hidden_space = norm_vector + norm_vector*weight_broadcast  #note, don't convert back to bfloat16 at this step yet

            # add the new hidden_space to it
            self.cache = np.concat( (self.cache, new_hidden_space), axis=2  )
        return numpy_to_tensor(self.cache, dtype=x.dtype)
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        
        output_np = self.np_forward(x)
        # if SANITY_CHECK:
            # #Note: it keeped in float type
            # output = self._norm(x.float())
            # # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
            # # See https://github.com/huggingface/transformers/pull/29402
            # output = output * (1.0 + self.weight.float())
            # output_tensor =  output.type_as(x)

            # assert assert_tensor_same_float(output_np, output_tensor)
        return output_np
        
    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"
class Gemma3RMSNormLearn(nn.Module):
    #NOTE: has to do it in float32 precision
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
        self.cache = None
    
    def np_forward(self, x):
        #NOTE: follow implementation, will use float32 for computation
        #[batch_Size, seq_len, hidden_size]
        x_np = tensor_to_numpy(x)
        
        # RMS norm is applied on every hidden_size vector for all seq_len and batch
        batch_size = x_np.shape[0]
        seq_len = x_np.shape[1]
        hidden_size =x_np.shape[2]
        
        if self.cache is None:
            scalar_multiple = np.full(  (batch_size, seq_len, 1), 1/hidden_size, dtype=np.float32)
            var_approx_vector = scalar_multiple * np.sum(  np.square(x_np), axis=2, keepdims=True)
            
            # add the epsilon
            var_approx_vector += np.full(  (batch_size, seq_len, 1), self.eps, dtype=np.float32)
            rqrt_res = np.reciprocal( np.sqrt(var_approx_vector))
            
            # Now broadcast it to hidden_space_size
            #[batch, seq_len, hidden_size]
            var_approx_vector_rsqrt = np.repeat(rqrt_res, repeats=hidden_size, axis=2)
            
            norm_vector = var_approx_vector_rsqrt * x_np
            
            # now is something special
            #[1,  hidden_size] - >[batchsize, seq_len,]
            weight_broadcast = np.broadcast_to(self.weight,  (batch_size, seq_len, hidden_size)) 
            self.cache = norm_vector + norm_vector*weight_broadcast

        else:
            # use cache means only do it on the newest_sequence_length for every batch
            # the only does it on the last seq_elen of every bach
            new_seq_size = x_np.shape[1] - self.cache.shape[1]
            
            x_np_last_seq = x_np[:, -new_seq_size:, :]
            scalar_multiple = np.full(   (batch_size, new_seq_size, 1), 1/hidden_size, dtype=np.float32)
            var_approx_vector = scalar_multiple   * np.sum(np.square(x_np_last_seq), axis=2, keepdims=True)      

            # add the epsilon
            var_approx_vector += np.full(  (batch_size, new_seq_size, 1), self.eps, dtype=np.float32)
            rqrt_res = np.reciprocal(np.sqrt(var_approx_vector)) # inverse square root
            
            
            # now broadhcast to hidden space_size
            var_approx_vector_rsqrt = np.repeat(rqrt_res, repeats=hidden_size, axis=2)
            norm_vector = var_approx_vector_rsqrt * x_np_last_seq
            
            weight_broadcast = np.broadcast_to(self.weight, [batch_size, new_seq_size, hidden_size])  
            new_hidden_space = norm_vector + norm_vector*weight_broadcast  #note, don't convert back to bfloat16 at this step yet
            
            # add back to cache
            self.cache = np.concat( (self.cache, new_hidden_space), axis=1)
          
        #TODO: turn cache back on later?
        return numpy_to_tensor(self.cache, dtype=x.dtype)
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        
        output_np = self.np_forward(x)
        # #Note: it keeped in float type
        if SANITY_CHECK:
            output = self._norm(x.float())
            # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
            # See https://github.com/huggingface/transformers/pull/29402
            output = output * (1.0 + self.weight.float())
            output_tensor =  output.type_as(x)

            assert assert_tensor_same_float(output_np, output_tensor)
        return output_np
        
    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"



class Gemma3TextScaledWordEmbeddingLearn(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.register_buffer("embed_scale", torch.tensor(embed_scale, dtype=self.weight.dtype), persistent=False)

    def forward(self, input_ids: torch.Tensor):
        #TODO potential cache
        print(f"embed_scalse is {self.embed_scale}")
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype) #Note:cast to same weight type!!




class Gemma3MLPLearn(nn.Module):
    def __init__(self, config: Gemma3TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]
    
        self.cache_gate_proj_res = None
        self.cache_up_proj_res = None
        self.cache_gelu_res = None
        self.cache_down_proj_res_np = None

        self.mlp_input_cache = None
    def forward(self, x, cache_np=None ):
        
        self.mlp_input_cache = x
        #[batch, seq_len, hidden_size]
        x_np = tensor_to_numpy(x)
        seq_len = x_np.shape[1]
        
        
        if self.cache_gate_proj_res is None:
        
            # #[batch, seq_len, intermediate_size]  or if use cache [batch, 1, intermediate_size ]
            self.cache_gate_proj_res = nn_linear_numpy(x_np, self.gate_proj.weight, self.gate_proj.bias, None)
            
            # #[batch, seq_len, intermediate_size] or if use cache [batch, 1, intermediate_size ]
            self.cache_up_proj_res = nn_linear_numpy(x_np, self.up_proj.weight, self.up_proj.bias, None)
            # #[batch, seq_len, intermediate_size] or if use cache [batch, len_of_offset, intermediate_size ]
            self.cache_gelu_res =   gelu_element_wise_np( self.cache_gate_proj_res) * self.cache_up_proj_res  

            self.cache_down_proj_res_np = numpy_to_tensor(  nn_linear_numpy(  self.cache_gelu_res, self.down_proj.weight, self.down_proj.bias,None),
                                            dtype=x.dtype
                                            )
        else:
            # #[batch, seq_len, intermediate_size]  or if use cache [batch, 1, intermediate_size ]
            self.cache_gate_proj_res = nn_linear_numpy(x_np, self.gate_proj.weight, self.gate_proj.bias, self.cache_gate_proj_res)
            
            # #[batch, seq_len, intermediate_size] or if use cache [batch, 1, intermediate_size ]
            self.cache_up_proj_res = nn_linear_numpy(x_np, self.up_proj.weight, self.up_proj.bias, self.cache_up_proj_res)
            # #[batch, seq_len, intermediate_size] or if use cache [batch, len_of_offset, intermediate_size ]
            
            new_seq_size = x_np.shape[1] - self.cache_down_proj_res_np.shape[1]
            
            new_gelu_res = gelu_element_wise_np( self.cache_gate_proj_res[:, -new_seq_size:, :]) 
            self.cache_gelu_res =   np.concat(  ( self.cache_gelu_res, new_gelu_res ) ,axis=1)  \
                * self.cache_up_proj_res  

            self.cache_down_proj_res_np = numpy_to_tensor(  nn_linear_numpy(  self.cache_gelu_res, self.down_proj.weight, self.down_proj.bias,self.cache_down_proj_res_np),
                                            dtype=x.dtype
                                            )  
        if SANITY_CHECK:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            
            assert assert_tensor_same_float(down_proj, self.cache_down_proj_res_np)
        return self.cache_down_proj_res_np






class Gemma3AttentionLearn(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.attn_logit_softcapping = self.config.attn_logit_softcapping
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.q_norm = Gemma3RMSNormMultiHeadLearn(dim=config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNormMultiHeadLearn(dim=config.head_dim, eps=config.rms_norm_eps)
        # some reference copieds
        self._q_norm_ref:None|Gemma3RMSNormMultiHeadLearn= None
        self._k_norm_ref:None|Gemma3RMSNormMultiHeadLearn = None #copy.deepcopy(self.k_norm)

        #[batch, head_num, seq_len, d_k]
        self.k_embed_multi_head_cache: None|npt.NDArray  = None
        self.v_embed_multi_head_cache: None|npt.NDArray  = None
        self.attention_res_cache :None | npt.NDArray = None
        
        self.attention_input_debug = None
        self.attention_output_debug = None
    def attention_np(self, batch_size, seq_len, num_attention_heads, q_emb, k_emb, v_prime_mult_heads, attention_mask) -> None:

            """
                $$
                Q^{'}\times K^{'  T}
                $$
            """
            #[batch, num_attention_head, d_k, seq_len]
            k_emb_transpose = np.transpose(a=k_emb, axes=(0,1,3,2))
            
            self.K_emb_rotated_cache  = k_emb_transpose
            self.V_cache =  v_prime_mult_heads
            
            # [batch, num_attention_head, seq_len, seq_len]
            q_mult_k_tran = np.matmul(q_emb, k_emb_transpose,)* self.scaling #Note: multiple with a scaling factor  
            if attention_mask is not None:
                # [batch, 1, seq_len, seq_len]  #Because all heads share same casual mask
                attention_mask_np = tensor_to_numpy(attention_mask)
                # At this point, the mask contain either 0/-inf
                # Thus, just simply an addition to the weight will be okay
                q_mult_k_tran +=   np.broadcast_to(attention_mask_np, (batch_size, num_attention_heads, seq_len, seq_len ))
            
            # apply softmax on it
            #[batch, num_attention_head, seq_len, seq_len]
            softmax_res = softmax(q_mult_k_tran, axis= 3) # over the  d_k
            
            # multiply with V
            #[batch, num_attention_head, seq_len, d_k]
            soft_max_mult_v = np.matmul(softmax_res, v_prime_mult_heads)
            
            # Now, it is time to concat the heads back together
            #[batch, seq_len, num_attention_head, d_k]
            soft_max_mult_v = np.transpose(soft_max_mult_v, (0, 2, 1, 3))
            #[batch, seq_len, d_model]
            soft_max_mult_v = soft_max_mult_v.reshape(batch_size, seq_len, -1)

            
            res = nn_linear_numpy( soft_max_mult_v, self.o_proj.weight, self.o_proj.bias )
            return res
            
    def flash_attention_np_single_head_singe_batch(self, Q: npt.NDArray, K: npt.NDArray, V: npt.NDArray, BR: int, BC: int, attention_mask: None | npt.NDArray):
        """
        Flash Attention 2 implementation for single head, single batch.
        
        Args:
            Q: [seq_len, d_size] - Query matrix
            K: [seq_len, d_size] - Key matrix  
            V: [seq_len, d_size] - Value matrix
            BR: Block size for rows
            BC: Block size for columns
            attention_mask: [seq_len, seq_len] - Attention mask (0 or -inf)
            
        Returns:
            O: [seq_len, d_size] - Output matrix
        """
        seq_len, d_size = Q.shape
        assert K.shape == (seq_len, d_size) and V.shape == (seq_len, d_size)
        
        # Calculate number of blocks
        num_i_block = (seq_len + BR - 1) // BR
        num_j_block = (seq_len + BC - 1) // BC
        
        O = np.zeros_like(Q, dtype=np.float32)
        
        for i in range(num_i_block):
            # init O_i
            i_start = i*BR
            i_end: int = min(seq_len, (i+1)*BR)
            BR_ac = i_end-i_start
            
            O_i = O[i_start:i_end, :] #[BR, d]
            Q_i = Q[i_start: i_end, :]  #[BR, d]
            
            prev_m_i_j = -np.inf * np.ones((BR_ac, 1), dtype=np.float32)
            prev_l_i_j = np.zeros((BR_ac, 1), dtype=np.float32)
            
            for j in range(num_j_block):
                j_start = j*BC
                j_end = min(seq_len, (j+1)*BC)
                BC_ac = j_end-j_start
                
                V_j = V[j_start: j_end, :] #[BC, d]
                K_j = K[j_start: j_end, :] #[BC, d]
                
                S_i_j = np.matmul(Q_i, np.transpose(K_j), dtype=np.float32) * self.scaling  #[BR, BC]
                
                # apply causal mask
                if attention_mask is not None:
                    S_i_j += attention_mask[i_start: i_end, j_start:j_end]  # attention mask of either 0 or -inf
                
                if j == 0:
                    # first time, no need softmax with previous
                    m_i_j = np.max(S_i_j, axis=1, keepdims=True)  #[Br, 1]
                    p_i_j = np.exp(S_i_j - m_i_j)  #[BR, BC]  # broadcast m_i_j, then element wise minus, then exp
                    l_i_j = np.sum(p_i_j, axis=1, keepdims=True)
                    O_i = np.matmul(p_i_j, V_j) #[Br, d]
                else:                 
                    m_i_j = np.maximum(prev_m_i_j, np.max(S_i_j, axis=1, keepdims=True))  #[BR,1]
                    p_i_j = np.exp(S_i_j - m_i_j)  # broadcast m_i_j, [BR, BC] element wise minus, then exp

                    # Fix: Correct l_i_j update equation for Flash Attention 2
                    l_i_j = np.exp(prev_m_i_j - m_i_j) * prev_l_i_j + np.sum(p_i_j, axis=1, keepdims=True) #[Br, 1]

                    # Fix: Correct O_i update equation for Flash Attention 2
                    O_i = O_i * np.exp(prev_m_i_j - m_i_j) + np.matmul(p_i_j, V_j) #[Br, d]
                    #O_i = O_i * np.exp(m_i_j - prev_m_i_j) + np.matmul(p_i_j, V_j)
                
                prev_m_i_j = m_i_j
                prev_l_i_j = l_i_j
                
                if j+1 == num_j_block:
                    # Fix: Add numerical stability for final normalization
                    l_i_j_safe = l_i_j #TODO: maybe eps 
                    l_i_j_inv = np.reciprocal(l_i_j_safe)
                    l_i_j_diag = np.diagflat(l_i_j_inv)
                    O_i = l_i_j_diag @ O_i
                    O[i_start:i_end, :] = O_i
        
        return O

    def forward_numpy (self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        # hidden_states is [batch, seq_len, d_model]
        # position_embedding is [batch, seq_len]
        # attention_mask is [batch, seq_len, seq_len]
        
        # split the hidden_states into multiple heads of d_k size. d_model/d_k  = num_attention_heads
        hidden_states_np = tensor_to_numpy(hidden_states)  # [batch, seq_len, d_model]
        batch_size = hidden_states_np.shape[0]
        seq_len = hidden_states_np.shape[1]
        d_model: int = hidden_states_np.shape[2]  # d_model is the hidden size
        d_k = self.head_dim  # d_k is the hidden size of each attention head
        num_attention_heads = self.config.num_attention_heads   
        num_kv_attention_heads = self.config.num_key_value_heads
        
        #NOTE: num_kv_attention_heads <= num_attention_heads
        
            
        def _interleave_repeat_axis1(arr, repeats):
            b, h, s, d = arr.shape

            # 1. Add a new dimension where repetitions will occur
            # Resulting shape: (b, h, 1, s, d)
            arr_expanded = np.expand_dims(arr, axis=2)

            # 2. Repeat along the new dimension
            # Resulting shape: (b, h, repeats, s, d)
            arr_repeated = np.repeat(arr_expanded, repeats, axis=2)

            # 3. Reshape to interleave the repetitions
            # We want to combine 'h' and 'repeats' into a single dimension of size h * repeats
            # and reorder it such that the repetitions of each 'h' element are grouped.
            # The new shape will be (b, h * repeats, s, d)
            arr_interleaved = arr_repeated.reshape(b, int(h * repeats), s, d)

            return arr_interleaved
        
        if self.k_embed_multi_head_cache is None:
            # For prefill stage
            # do a MM with w^q, w^k, w^v
            #[batch, seq_len, d_k*num_attention_head]
            q_prime = nn_linear_numpy(hidden_states_np,  self.q_proj.weight, self.q_proj.bias )
            #[batch, seq_len, d_k*num_kv_attention_heads]  where d_compres <= d_model
            k_prime = nn_linear_numpy(hidden_states_np, self.k_proj.weight, self.k_proj.bias)
            v_prime = nn_linear_numpy(hidden_states_np, self.v_proj.weight, self.v_proj.bias)
            
            # Now, split q_prime, k_prime, v_prime to multiple-heads
            #[batch, seq_len, num_attention_head, d_k]
            q_prime_mult_heads = q_prime.reshape(batch_size, seq_len, num_attention_heads, d_k)
            #[batch, seq_len, num_kv_attention_heads, d_k]        
            v_prime_mult_heads = v_prime.reshape(batch_size, seq_len, num_kv_attention_heads, d_k)
            k_prime_mult_heads = k_prime.reshape(batch_size, seq_len, num_kv_attention_heads, d_k)
            
            # swap orders for paralle operations
            #[batch, num_attention_head, seq_len, d_k]
            q_prime_mult_heads = np.transpose( q_prime_mult_heads,axes=(0,2,1,3))
            #[batch, num_kv_attention_heads, seq_len, d_k]                
            v_prime_mult_heads = np.transpose(v_prime_mult_heads, axes=(0,2,1,3))
            k_prime_mult_heads = np.transpose(k_prime_mult_heads,axes=(0,2,1,3))  

            #NOTE: something special for gemma is that it will apply norm to qk
            q_prime_mult_heads = tensor_to_numpy(self.q_norm( numpy_to_tensor( q_prime_mult_heads, dtype=hidden_states.dtype )))
            k_prime_mult_heads = tensor_to_numpy(self.k_norm( numpy_to_tensor( k_prime_mult_heads, dtype=hidden_states.dtype)))

            # Now need to extend v, k in a interleave fastion to [batch, num_attention_head, seq_len, d_k]
            # if k_prime is [k1, k2, k3, k4] -> [k1, k1, k1, k1, k2, k2, k2,k2, .... k4, k4]
            k_prime_mult_heads = _interleave_repeat_axis1(k_prime_mult_heads, num_attention_heads/num_kv_attention_heads )
            v_prime_mult_heads = _interleave_repeat_axis1(v_prime_mult_heads, num_attention_heads/num_kv_attention_heads)
            
            # apply both rope on Q_prime and K_prime
            q_emb, k_emb = apply_rotary_pos_emb_np(q_prime_mult_heads, k_prime_mult_heads, 
                                                tensor_to_numpy(position_embeddings[0]),tensor_to_numpy(position_embeddings[1])
                                                )
            
            # Cache it
            self.k_embed_multi_head_cache = k_emb.copy()
            self.v_embed_multi_head_cache = v_prime_mult_heads.copy()
            
            
            # res =  self.attention_np(batch_size, seq_len, num_attention_heads, q_emb, k_emb, 
            #                   v_prime_mult_heads, attention_mask
            #                   )
            # return numpy_to_tensor(atten_res, dtype=hidden_states.dtype)
            output = np.zeros_like(q_emb, dtype=np.float32)  # Use float32 for consistency
            # try the flash attention method(slow because not optimized)
            for batch_idx in range(batch_size):
                for head_idx in range(num_attention_heads):
                    q_bh = q_emb[batch_idx, head_idx, :, :].astype(np.float32)
                    v_bh = v_prime_mult_heads[batch_idx, head_idx, :, :].astype(np.float32)
                    k_bh = k_emb[batch_idx, head_idx, :, :].astype(np.float32)
                    # Attention mask should be same within the mask
                    atten_mask = None
                    if attention_mask is not None:
                        atten_mask = tensor_to_numpy(attention_mask[batch_idx, -1, :, :])
                    output[batch_idx, head_idx] = self.flash_attention_np_single_head_singe_batch(
                        q_bh, k_bh, v_bh, 32, 32, atten_mask
                        
                    )
                    
            # now is time to concat the head back together
            #[batch, seq_len, num_attention, d_k]
            output = np.transpose(output, (0,2,1,3))
            output = output.reshape(batch_size, seq_len, -1)
            res = nn_linear_numpy(output, self.o_proj.weight, self.o_proj.bias)
        
            self.attention_res_cache = res
            return numpy_to_tensor(res, dtype=hidden_states.dtype)
        else:
            
            new_seq_size = seq_len -  self.k_embed_multi_head_cache.shape[2] 
            
            # Decode stage, only care about the new input stuff
            fake_q_cache =np.random.rand(  hidden_states.shape[0], hidden_states.shape[1]-new_seq_size, self.q_proj.weight.shape[0]  )
            fake_kv_cache = np.random.rand(hidden_states.shape[0], hidden_states.shape[1]-new_seq_size, self.k_proj.weight.shape[0])
            q_token_prime = nn_linear_numpy(hidden_states_np, self.q_proj.weight, self.q_proj.bias, cache=fake_q_cache)
            k_token_prime = nn_linear_numpy(hidden_states_np, self.k_proj.weight, self.k_proj.bias, cache=fake_kv_cache)
            v_token_prime = nn_linear_numpy(hidden_states_np, self.v_proj.weight, self.v_proj.bias, cache=fake_kv_cache)

            #[batch, new_seq_size, d_model]
            q_new_token_prime = q_token_prime[:, -new_seq_size:, :]
            #[batch, new_seq_size, d_compress]
            k_new_token_prime = k_token_prime[:, -new_seq_size:, :]
            v_new_token_prime = v_token_prime[:, -new_seq_size:, :]
            
            # split to multiple heads
            #[batch, new_seq_size, num_attention_head, d_k]
            q_new_token_prime_mult_heads = q_new_token_prime.reshape(batch_size, new_seq_size, num_attention_heads, d_k)
            #[btach, new_seq_size, num_kv_head, d_k]
            k_new_token_prime_mult_heads = k_new_token_prime.reshape(batch_size, new_seq_size, num_kv_attention_heads, d_k)
            v_new_token_prime_mult_heads = v_new_token_prime.reshape(batch_size, new_seq_size, num_kv_attention_heads, d_k)
            
            # change shape for parallel operations
            #[batch, num_attention_head, new_seq_size, d_k]
            q_new_token_prime_mult_heads = np.transpose(q_new_token_prime_mult_heads, axes=(0, 2, 1,3))
            #[batch, num_kv_attention_head, new_seq_size, d_k]
            k_new_token_prime_mult_heads = np.transpose(k_new_token_prime_mult_heads, axes=(0, 2, 1,3))
            v_new_token_prime_mult_heads = np.transpose(v_new_token_prime_mult_heads, axes=(0,2,1,3))
            
            
            #NOTE: something special for gemma is that it will apply norm to qk
            # some fake cahce to invoke the cahce in q_norm and k_norm      
            fake_q_norm_cache = np.random.random_sample(  self.q_norm.cache.shape  )
            fake_k_norm_cache = np.random.random_sample(  self.k_norm.cache.shape  )
            fake_q_norm_cache = np.concat( (fake_q_norm_cache, q_new_token_prime_mult_heads), axis=2)
            fake_k_norm_cache = np.concat( (fake_k_norm_cache, k_new_token_prime_mult_heads), axis=2)
            
            q_new_token_prime_mult_heads = tensor_to_numpy(self.q_norm(numpy_to_tensor(fake_q_norm_cache, dtype=hidden_states.dtype)))
            k_new_token_prime_mult_heads = tensor_to_numpy(self.k_norm(numpy_to_tensor(fake_k_norm_cache,dtype=hidden_states.dtype)))
            
            q_new_token_prime_mult_heads = q_new_token_prime_mult_heads[:, :, -new_seq_size:, :]
            k_new_token_prime_mult_heads = k_new_token_prime_mult_heads[:, :, -new_seq_size:, :]
                        
            # Now need to extend v, k in a interleave fashiton to [batch, num_attention_head, new_seq_size, d_k]
            k_new_token_prime_mult_heads = _interleave_repeat_axis1(k_new_token_prime_mult_heads, repeats=num_attention_heads//num_kv_attention_heads)
            v_new_token_prime_mult_heads = _interleave_repeat_axis1(v_new_token_prime_mult_heads, repeats=num_attention_heads//num_kv_attention_heads)
            
            #[batch, num_attention_head, new_seq_size, d_k] , [batch, num_attention_head, new_seq_size, d_k]
            #NOTE: only retrieve the newest token rope embeddding
            q_emb, k_emb = apply_rotary_pos_emb_np(q_new_token_prime_mult_heads, k_new_token_prime_mult_heads,
                                       tensor_to_numpy(position_embeddings[0][:, -new_seq_size:, :] ),
                                       tensor_to_numpy(position_embeddings[1][:, -new_seq_size:, :])         
                                                )
           
            #[batch, num_atteiton_head, d_k, seq_len_prev+new_seq_size]
            self.k_embed_multi_head_cache  = np.concat( (self.k_embed_multi_head_cache, k_emb ), axis=2)
            #[batch, num_atteiton_head, seq_len_prev+new_seq_size, d_k]            
            self.v_embed_multi_head_cache = np.concat( (self.v_embed_multi_head_cache, v_new_token_prime_mult_heads), axis=2)
           
            
            #[batch, num_attention_head, d_k, new_seq_size]
            k_emb_transpose= np.transpose(a=self.k_embed_multi_head_cache, axes=(0,1,3,2)) # the whole K Cache
            #[batch, num_attention_head, new_seq_size, new_seq_size]
            q_mult_k_tran = np.matmul(q_emb, k_emb_transpose)*self.scaling
            
            #NOTE: no need casual mask at decode stage for whole attention, but need for sliding window
            if self.is_sliding:
                # apply softmax on it
                mask_window = attention_mask[:, :, -new_seq_size:, :]
                mask_window = np.broadcast_to( mask_window, q_mult_k_tran.shape )
                q_mult_k_tran += mask_window
                
            #[batch, num_attention_head, new_seq_size, seq_len_prev+1]
            softmax_res = softmax(q_mult_k_tran, axis= 3) # over the  d_k
            
            # multiply with V
            #[batch, num_attention_head, new_seq_size, d_k]
            soft_max_mult_v = np.matmul(softmax_res, self.v_embed_multi_head_cache)
            
            # Now, it is time to concat the heads back together
            #[batch, new_seq_size, num_attention_head, d_k]
            soft_max_mult_v = np.transpose(soft_max_mult_v, (0, 2, 1, 3))
            #[batch, new_seq_size, d_model]
            soft_max_mult_v = soft_max_mult_v.reshape(batch_size, new_seq_size, -1)

            res = nn_linear_numpy( soft_max_mult_v, self.o_proj.weight, self.o_proj.bias )

        
        
            self.attention_res_cache = np.concat( (self.attention_res_cache, res), axis=1 )
            return numpy_to_tensor(  self.attention_res_cache, dtype=hidden_states.dtype)




    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cache_np = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        if self._q_norm_ref is None:
            self._q_norm_ref = copy.deepcopy(self.q_norm)
            self._k_norm_ref = copy.deepcopy(self.k_norm)
        self.attention_input_debug = hidden_states
        np_attn_output = self.forward_numpy(hidden_states, position_embeddings, 
                                             attention_mask 
                                             )


        
        # # if SANITY_CHECK:
        # input_shape = hidden_states.shape[:-1]
        # hidden_shape = (*input_shape, -1, self.head_dim)

        # query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # query_states = self._q_norm_ref(query_states)
        # key_states = self._k_norm_ref(key_states)

        # cos, sin = position_embeddings
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # attention_interface: Callable = eager_attention_forward
        # if self.config._attn_implementation != "eager":
        #     attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # attn_output, attn_weights = attention_interface(
        #     self,
        #     query_states,
        #     key_states,
        #     value_states,
        #     attention_mask,
        #     dropout=self.attention_dropout if self.training else 0.0,
        #     scaling=self.scaling,
        #     sliding_window=self.sliding_window,
        #     **kwargs,
        # )

        # attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        # attn_output = self.o_proj(attn_output)
        
        
        # assert assert_tensor_same_float(np_attn_output,attn_output)
            # return np_attn_output, attn_weights
            
        self.attention_output_debug = np_attn_output
                    
        return np_attn_output, None



class Gemma3DecoderLayerLearn(GradientCheckpointingLayer):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = Gemma3AttentionLearn(config=config, layer_idx=layer_idx)
        self.mlp = Gemma3MLPLearn(config)
        self.input_layernorm = Gemma3RMSNormLearn(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNormLearn(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNormLearn(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNormLearn(self.hidden_size, eps=config.rms_norm_eps)

        
        self.decode_layer_input_hidden_states = None
        self.decode_layer_output_hidden_states = None
    @deprecate_kwarg("last_cache_position", version="4.53.0")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings_global: torch.Tensor,
        position_embeddings_local: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        self.decode_layer_input_hidden_states = hidden_states
        
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # apply global RoPE to non-sliding layer only
        if self.self_attn.is_sliding:
            position_embeddings = position_embeddings_local
        else:
            position_embeddings = position_embeddings_global

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        
        self.decode_layer_output_hidden_states  = outputs
        return outputs




class Gemma3TextModelLearn(Gemma3PreTrainedModel):
    config_class = Gemma3TextConfig

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Gemma3 downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5. See https://github.com/huggingface/transformers/pull/29402
        self.embed_tokens = Gemma3TextScaledWordEmbeddingLearn(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=self.config.hidden_size**0.5
        )
        self.layers = nn.ModuleList(
            [Gemma3DecoderLayerLearn(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma3RMSNormLearn(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbeddingLearn(config=config)
        self.gradient_checkpointing = False

        # TODO: raushan fix this after RoPE refactor. For now we hack it by reassigning thetas
        # when we want to create a local RoPE layer. Config defaults should hold values for global RoPE
        config = copy.deepcopy(config)
        config.rope_theta = config.rope_local_base_freq
        config.rope_scaling = {"rope_type": "default"}
        self.rotary_emb_local = Gemma3RotaryEmbeddingLearn(config=config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not self.training:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
        position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )



class Gemma3ModelLearn(Gemma3PreTrainedModel):
    _checkpoint_conversion_mapping = {"language_model.model": "language_model"}
    # we are filtering the logits/labels so we shouldn't divide the loss based on num_items_in_batch
    accepts_loss_kwargs = False

    def __init__(self, config: Gemma3Config):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config=config.vision_config)
        self.multi_modal_projector = Gemma3MultiModalProjectorLearn(config)
        self.vocab_size = config.text_config.vocab_size

        language_model =AutoModel.from_config(config=config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.post_init()
        
        # use my own model instead
        self.language_model  = Gemma3TextModelLearn(language_model.config) # use my own model
        self.vision_tower = SiglipVisionModelLearn(config.vision_config)

        self.input_embedding_debug_cache:None|torch.Tensor = None
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_image_features(self, pixel_values: torch.Tensor, tensors_to_save:Optional[dict] = None) -> torch.Tensor:
        """
        Projects the last hidden state from the vision model into language model space.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        vision_outputs = self.vision_tower(pixel_values=pixel_values,tensors_to_save=tensors_to_save).last_hidden_state
        image_features = self.multi_modal_projector(vision_outputs, tensors_to_save=tensors_to_save)
        return image_features

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[list[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        tensors_to_save:Optional[dict] = None,
        **lm_kwargs,
    ) -> Union[tuple, Gemma3ModelOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        >>> model = Gemma3ForConditionalGeneration.from_pretrained("google/gemma32-3b-mix-224")
        >>> processor = AutoProcessor.from_pretrained("google/gemma32-3b-mix-224")

        >>> prompt = "Where is the cat standing?"
        >>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt,  return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs,)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Where is the cat standing?\nsnow"
        ```"""
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Replace image id woth PAD if the image token if OOV, to avoid index-errors
        if input_ids is not None and self.config.image_token_id >= self.vocab_size:
            special_image_mask = input_ids == self.config.image_token_id
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # Merge text and images
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values,tensors_to_save=tensors_to_save)

            if input_ids is None:
                special_image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            else:
                special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
                special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

            if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
                image_tokens_in_text = (special_image_mask).sum(dim=1).sum(dim=0)[0]
                raise ValueError(
                    f"Number of images does not match number of special image tokens in the input text. "
                    f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
                    "tokens from image embeddings."
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config.get_text_config(),
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
            }
            if token_type_ids is not None and inputs_embeds.shape[1] != 1:
                # We need to pass an additional mask function to account for token type ids, and it needs to be an `or`
                mask_kwargs["or_mask_function"] = token_type_ids_mask_function(
                    token_type_ids.to(cache_position.device), self.config.mm_tokens_per_image
                )

            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }
        self.input_embedding_debug_cache = inputs_embeds
        outputs = self.language_model(
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **lm_kwargs,
        )

        return Gemma3ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values if use_cache else None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )




class Gemma3ForConditionalGenerationLearn(Gemma3PreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {
        "^language_model.model": "model.language_model",
        "^vision_tower": "model.vision_tower",
        "^multi_modal_projector": "model.multi_modal_projector",
        "^language_model.lm_head": "lm_head",
    }
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Gemma3Config):
        super().__init__(config)
        self.model = Gemma3ModelLearn(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_image_features(self, pixel_values):
        return self.model.get_image_features(pixel_values)

    # Make modules available throught conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def vision_tower(self):
        return self.model.vision_tower

    @property
    def multi_modal_projector(self):
        return self.model.multi_modal_projector

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[list[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        tensors_to_save:dict|None= None, 
        **lm_kwargs,
    ) -> Union[tuple, Gemma3CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        >>> model = Gemma3ForConditionalGeneration.from_pretrained("google/gemma-3-4b-it")
        >>> processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")

        >>> messages = [
        ...     {
        ...         "role": "system",
        ...         "content": [
        ...             {"type": "text", "text": "You are a helpful assistant."}
        ...         ]
        ...     },
        ...     {
        ...         "role": "user", "content": [
        ...             {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
        ...             {"type": "text", "text": "Where is the cat standing?"},
        ...         ]
        ...     },
        ... ]

        >>> inputs = processor.apply_chat_template(
        ...     messages,
        ...     tokenize=True,
        ...     return_dict=True,
        ...     return_tensors="pt",
        ...     add_generation_prompt=True
        ... )
        >>> # Generate
        >>> generate_ids = model.generate(**inputs)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "user\nYou are a helpful assistant.\n\n\n\n\n\nWhere is the cat standing?\nmodel\nBased on the image, the cat is standing in a snowy area, likely outdoors. It appears to"
        ```
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            tensors_to_save=tensors_to_save,
            **lm_kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :].to(logits.device)
                shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()

            flat_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Gemma3CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        pixel_values=None,
        attention_mask=None,
        token_type_ids=None,
        use_cache=True,
        logits_to_keep=None,
        labels=None,
        **kwargs,
    ):
        # Overwritten -- custom `position_ids` and `pixel_values` handling
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
        # Otherwise we need pixel values to be passed to model. NOTE: use_cache=False needs pixel_values always
        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values

        return model_inputs

    @staticmethod
    def create_masks_for_generate(
        config: PretrainedConfig,
        input_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        cache_position: torch.Tensor,
        past_key_values: Optional[Cache],
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        # Prepare mask arguments
        mask_kwargs = {
            "config": config.get_text_config(),
            "input_embeds": input_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
        }
        # Add the token type ids mask for generate as well
        if token_type_ids is not None and input_embeds.shape[1] != 1:
            # We need to pass an additional mask function to account for token type ids, and it needs to be an `or`
            mask_kwargs["or_mask_function"] = token_type_ids_mask_function(
                token_type_ids.to(cache_position.device), config.mm_tokens_per_image
            )

        return create_masks_for_generate(**mask_kwargs)


__all__ = [
    "Gemma3PreTrainedModel",
    "Gemma3TextModel",
    "Gemma3ForCausalLM",
    "Gemma3ForConditionalGeneration",
    "Gemma3Model",
]

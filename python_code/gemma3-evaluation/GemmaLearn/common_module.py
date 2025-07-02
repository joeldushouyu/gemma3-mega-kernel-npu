

# mypy: allow-untyped-defs
import numbers
import math
from typing import Optional, Union
import numpy as np
import numpy.typing as npt
import torch
from torch import Size, Tensor
from torch.nn import functional as F, init
from torch.nn.parameter import Parameter

from torch.nn.modules._functions import CrossMapLRN2d as _cross_map_lrn2d
from torch.nn.modules.module import Module
_shape_t = Union[int, list[int], Size]

from .learn_util import tensor_to_numpy, assert_tensor_same_float, numpy_to_tensor


def gelu_element_wise_np(x:npt.NDArray ):
    
    # x might be [batch, seq_len. hidden_size]
    # but this is applying to elementwise, so
    
    """
    Applies the Gaussian Error Linear Unit (GELU) activation function element-wise.
    Uses the approximation: 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) )).
    
    Args:
        x (np.ndarray): Input array.
        
    Returns:
        np.ndarray: Array of the same shape as `x`, with GELU applied element-wise.
    """
    # Constants
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    coeff = 0.044715
    
    # Implementation
    x_cubed = x ** 3
    tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed)
    return 0.5 * x * (1.0 + np.tanh(tanh_arg))




class LayerNormLearn(Module):
    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
            if bias:
                self.bias = Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def np_forward(self, input:Tensor, seq_len_offset:int, seq_range:int):
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        hidden_size = input.shape[2]
        
        #[hidden_size], aka unique for every hidden size element 
        bias_np = tensor_to_numpy(self.bias)
        # eps is a scalar, aka shared for every hidden size element
        
        #[batch_size, seq_len, hidden_size ]
        hidden_state_np = tensor_to_numpy(input)
        
        # The normalization is applied for all hidden_size vectors
        #[batch_size, seq_len, 1]
        mean_per_hidden_vector = np.mean( hidden_state_np , axis=2, keepdims=True)
        variance_per_hidden_vecrtor = np.var(hidden_state_np, axis=2, keepdims=True)
        
        
        # apply z-score normalization(with epsilon to prevent div-by-zero error) 
        mean_per_hidden_vector = np.broadcast_to(mean_per_hidden_vector,  (batch_size, seq_len, hidden_size) )
        variance_per_hidden_vector = np.broadcast_to(variance_per_hidden_vecrtor, (batch_size, seq_len, hidden_size))
        # the tensor is filled with all eps value
        eps_vector = np.full(  (batch_size, seq_len, hidden_size) , self.eps,  dtype=np.float32  )  
        
        # Duplicate copy for batch_size*seq_len times of the bias vector. In other words, the bias vector is shared for all seq_len and batch_size
        bias_vector = np.broadcast_to( tensor_to_numpy(self.bias), (batch_size, seq_len, hidden_size) )
        weight_vector = np.broadcast_to(tensor_to_numpy(self.weight), shape=(batch_size, seq_len, hidden_size))
        num=   hidden_state_np - mean_per_hidden_vector
        den = np.sqrt(   variance_per_hidden_vector  +  eps_vector    )

        res = num/den 
        
        res = (res *weight_vector) + bias_vector
        
        return numpy_to_tensor(res, input.dtype)
        
    def forward(self, input: Tensor, seq_len_offset=0, seq_range=0) -> Tensor:
        #NOTE: change the offset when result is cacheable(in decoder stage, maybe not in encoder stage?)
        #input  is [batch_size, seq_len, hidden_size]
        
        
        res_np = self.np_forward(input, seq_len_offset=seq_len_offset,seq_range = seq_range  )
        # res_ref =  F.layer_norm(
        #     input, self.normalized_shape, self.weight, self.bias, self.eps
        # )

        #assert assert_tensor_same_float(res_np, res_ref)
        return res_np
    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )





# class LinearLearn(Module):
#     r"""Applies an affine linear transformation to the incoming data: :math:`y = xA^T + b`.
#     """

#     __constants__ = ["in_features", "out_features"]
#     in_features: int
#     out_features: int
#     weight: Tensor

#     def __init__(
#         self,
#         in_features: int,
#         out_features: int,
#         bias: bool = True,
#         device=None,
#         dtype=None,
#     ) -> None:
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(
#             torch.empty((out_features, in_features), **factory_kwargs)
#         )
#         if bias:
#             self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
#         else:
#             self.register_parameter("bias", None)
#         self.reset_parameters()

#     def reset_parameters(self) -> None:
#         # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
#         # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
#         # https://github.com/pytorch/pytorch/issues/57109
#         init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#             init.uniform_(self.bias, -bound, bound)

#     def np_forward(self, input:Tensor, offset_seq_index:int, len_of_offset = 0) ->Tensor:
#         # input is [batch, seq_len, hidden_size]
#         # self.wight is  [proj_dimension, hidden_siz]
#         # self.bias is  [proj_dimension]
#         # When have cache, offset_seq_len is offset 
#         # implement as input * self.Weight + self.Bias
#         batch_size = input.shape[0]
#         seq_len = input.shape[1]
#         hidden_size = input.shape[2]
#         proj_dimension = self.weight.shape[0]
        
#         #[batch, seq_len, hidden_size]
  
#         input_np = tensor_to_numpy(input)


#         if len_of_offset != 0: # use of cache in linear
#             input_np = input_np[:,  offset_seq_index:offset_seq_index+len_of_offset, :]
        
#         # same weight and bias for all batch``
#         weight_batch = np.broadcast_to(tensor_to_numpy(self.weight), (batch_size, proj_dimension, hidden_size)  )
#         if self.bias is not None:
#             bias_batch  = np.broadcast_to(tensor_to_numpy(self.bias), (batch_size, proj_dimension))
        
#         #Do a tranpose on A
#         weight_batch = np.transpose(weight_batch, (0,2,1))
#         if self.bias is not None:
#             bias_batch_broad = np.broadcast_to( bias_batch, (batch_size, seq_len,  proj_dimension ))  # same [dimeison ]vector for all seq_len
#             res = np.matmul(input_np, weight_batch ) + bias_batch_broad
#         else:
#             res =  np.matmul(input_np, weight_batch )
#         return numpy_to_tensor(res, input.dtype)
        
    
#     def forward(self, input: Tensor, offset_seq_index=0, len_of_offset=0) -> Tensor:
        
#         res_np = self.np_forward(input, offset_seq_index, len_of_offset)
#         # res_ref = F.linear(input, self.weight, self.bias)

#         # assert assert_tensor_same_float(res_np, res_ref)
#         return res_np
        
#     def extra_repr(self) -> str:
#         return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"




def nn_linear_numpy( x_numpy:np.ndarray, weight:Tensor, bias:Tensor|None, cache:None| np.ndarray=None):
    # Basically is a  MV(matrix vector multiplcation) + bias vector
    # Strickly follows the torch implement of X * W_transpose + bias
    #[batch, seq_len, d_sze]

    batch_size = x_numpy.shape[0]
    seq_len = x_numpy.shape[1]
    d_size = x_numpy.shape[2]
    
    
    #  [proj_dim, d_size] where proj_dim depends on the projection layer of this nn.Linear
    weight_matrix_np = tensor_to_numpy(weight   )
    #  [d_size, proj_dim] where proj_dim depends on the projection layer of this nn.Linear
    weight_matrix_np = np.transpose(weight_matrix_np)
    
    projc_dim = weight_matrix_np.shape[1]
    
    #[batch, d_size, proj_dim]  Broadcast to mutliple batch 
    weight_matrix_np =np.broadcast_to( weight_matrix_np,  (batch_size, d_size, projc_dim))
    



    if cache is None:
        # At prefill stage, MM(matrix multilication of X*W^Transpose)
        # res is a MM(matrix multiplication of x * W^Transpose)   
        #[batch, seq_len,  proj_dim]     
        res = np.matmul( x_numpy, weight_matrix_np )  #TODO:? This could be turn to a MM operation, where the second MAtrix is seq_lenxd_size
        if bias is None:
            return res
        else:
             #[batch, d_size]
            bias_vector_numpy = np.broadcast_to(tensor_to_numpy(bias),(batch_size, d_size, projc_dim) )
            return     res + bias_vector_numpy 
    else:
        # But at the decode stage, only need to do it on the last token, [batch, seq_len_last_index, d_size]
        # AKA, do it on 1xd_size vector for every batch
        
        new_size: int = seq_len - cache.shape[1] 
        
        new_token_vector =x_numpy[:, -new_size:, :]
        # Do a MV 
        #[batch, new_size, d_size]
        res_new_token = np.matmul(new_token_vector, weight_matrix_np)

        if bias is not None:
            bias_vector_np = np.broadcast_to(tensor_to_numpy(bias),(batch_size, new_size, projc_dim) )
            res_new_token  += bias_vector_np
        return np.concat( (cache, res_new_token), axis=1)
    
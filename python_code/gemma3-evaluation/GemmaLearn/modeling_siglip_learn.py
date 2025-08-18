


import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.init import _calculate_fan_in_and_fan_out

from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils import ModelOutput, auto_docstring, can_return_tuple, logging, torch_int
from transformers.models.siglip.configuration_siglip import SiglipConfig, SiglipTextConfig, SiglipVisionConfig

import torch.nn.functional as F
from.common_module import LayerNormLearn

#TODO repalce it
from transformers.models.siglip import SiglipPreTrainedModel


from transformers.models.siglip.modeling_siglip import SiglipAttention


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->Siglip
class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
    
    
class SiglipEncoderLayerLearn(GradientCheckpointingLayer):
    def __init__(self, config: Union[SiglipVisionConfig, SiglipTextConfig]):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = SiglipAttention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
        tensors_to_save:dict|None=None,
        layer_id:int|None=None,        
    ) -> tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        if tensors_to_save is not None:
            assert layer_id is not None
        
        if tensors_to_save is not None:
            tensors_to_save[f"layer_{layer_id}_before_norm1"] = hidden_states.contiguous()
        hidden_states = self.layer_norm1(hidden_states)
        if tensors_to_save is not None:
            tensors_to_save[f"layer_{layer_id}_after_norm1"] = hidden_states.contiguous()
            
            
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        if tensors_to_save is not None:
            tensors_to_save[f"layer_{layer_id}_after_attention"] = hidden_states.contiguous()
        
        hidden_states = residual + hidden_states
        residual = hidden_states
        
        if tensors_to_save is not None:
            tensors_to_save[f"layer_{layer_id}_before_norm2"] = hidden_states.contiguous()
        
        hidden_states = self.layer_norm2(hidden_states)

        if tensors_to_save is not None:
            tensors_to_save[f"layer_{layer_id}_after_norm2"] = hidden_states.contiguous()
        
        hidden_states = self.mlp(hidden_states)

        if tensors_to_save is not None:
            tensors_to_save[f"layer_{layer_id}_after_mlp"] = hidden_states.contiguous()
                    
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        if tensors_to_save is not None:
            tensors_to_save[f"layer_{layer_id}_final"] = hidden_states.contiguous()
        return outputs


class SiglipVisionEmbeddingsLearn(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        #https://www.youtube.com/watch?v=vVaRhZXovbw
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,  # number of filters
            kernel_size=self.patch_size, # the sizes of each kernel 
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing and no class embeddings.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1]
        num_positions = self.position_embedding.weight.shape[0]

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embedding(self.position_ids)

        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward(self, pixel_values: torch.FloatTensor, interpolate_pos_encoding=False, tensors_to_save:dict[str, Any]|None = None) -> torch.Tensor:
        if tensors_to_save is not None:
            clone_v: torch.Tensor =  pixel_values.clone()
            tensors_to_save["conv2d_kernels"] = self.patch_embedding.weight.contiguous()
            tensors_to_save["before_conv2d"] = clone_v.contiguous()
            
        _, _, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        #[image_number, width/filter_size,  grid, grid]
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        
    
        # it flattens and reshape to [1, grid*grid, width/filter_size]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        if tensors_to_save is not None:
            tensors_to_save["after_conv2d_with_reordered"] = embeddings.contiguous() #[num_image, H_out*W_out, hidden_size/num_kernels]

        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            # apply a absolute position embeddings on the embedding image(similar to the "Attention is all you need", note Rope)
            #[image_number, grid*grid, width/filter_size/embed_dim]
            embeddings = embeddings + self.position_embedding(self.position_ids)  #position_ids == grid*grid (4096 for 4bit-model)
        if tensors_to_save is not None:
            tensors_to_save["after_embeddings"] = embeddings.contiguous()
        return embeddings










# Copied from transformers.models.altclip.modeling_altclip.AltCLIPEncoder with AltCLIP->Siglip
class SiglipEncoderLearn(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`SiglipEncoderLayer`].

    Args:
        config: SiglipConfig
    """

    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SiglipEncoderLayerLearn(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # Ignore copy
    @can_return_tuple
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        tensors_to_save:dict|None=None,
    ) -> BaseModelOutput:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for layer_id, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                output_attentions=output_attentions,
                tensors_to_save=tensors_to_save,
                layer_id = layer_id
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class SiglipVisionTransformerLearn(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddingsLearn(config)
        self.encoder = SiglipEncoderLearn(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head

        assert self.use_head == False # seem for gemma
        # if self.use_head:
        #     self.head = SiglipMultiheadAttentionPoolingHead(config)

    @can_return_tuple

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        tensors_to_save:dict|None=None,
    ) -> BaseModelOutputWithPooling:
        assert interpolate_pos_encoding== False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # pixel_values is [image_number, 3, 896, 896]
        
        #[image_number, grid*grid(grid = image_width_height/kernel_size), image_embd_size]  #hidden_states after applyed position embedding
        # For gemma, it is [mimage_number, 64*64, image_embed_size=1152]
        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding, tensors_to_save=tensors_to_save)

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            tensors_to_save=tensors_to_save
        )


        last_hidden_state = encoder_outputs.last_hidden_state
        
    
        if tensors_to_save is not None:
            tensors_to_save["before_post_layernorm"] = last_hidden_state.contiguous()   
            tensors_to_save["post_layernorm_weight"] = self.post_layernorm.weight.contiguous()
            tensors_to_save["post_layernorm_bias"] = self.post_layernorm.bias.contiguous()

        print(f"post_layernorm eps: {self.post_layernorm.eps}")     
        last_hidden_state = self.post_layernorm(last_hidden_state)
        
        if tensors_to_save is not None:
            tensors_to_save["after_post_layernorm"] = last_hidden_state.contiguous()

        pooler_output = self.head(last_hidden_state) if self.use_head else None

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class SiglipVisionModelLearn(SiglipPreTrainedModel):
    config_class = SiglipVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: SiglipVisionConfig):
        super().__init__(config)

        self.vision_model = SiglipVisionTransformerLearn(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    @can_return_tuple
    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        tensors_to_save:Optional[dict] = None
    ) -> BaseModelOutputWithPooling:
        r"""
        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, SiglipVisionModel

        >>> model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled features
        ```"""

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            tensors_to_save=tensors_to_save
        )
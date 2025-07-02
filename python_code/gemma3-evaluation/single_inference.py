# from transformers import AutoProcessor, Gemma3ForConditionalGeneration, Gemma3Processor, Gemma3ImageProcessor, GemmaTokenizerFast, GemmaTokenizer
# from PIL import Image
# import requests
# import torch
# import os

# model_id = "google/gemma-3-4b-it"

# # Set cache directory to the directory where this script is located
# script_dir = os.path.dirname(os.path.abspath(__file__))
# cache_dir = os.path.join(script_dir, "hf_cache")
# model_dir = os.path.join(cache_dir, "models--google--gemma-3-4b-it")

# # Check if model is downloaded
# def is_model_downloaded(model_dir):
#     for root, dirs, files in os.walk(model_dir):
#         for file in files:
#             if file.startswith("pytorch_model") or file.endswith(".safetensors"):
#                 return True
#     return False

# if not is_model_downloaded(model_dir):
#     print("Model not found locally. Downloading...")
#     Gemma3ForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_dir)
#     AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
#     print("Download complete.")

# # Load model and processor from local cache, force CPU
# model = Gemma3ForConditionalGeneration.from_pretrained(
#     model_id, local_files_only=True, cache_dir=cache_dir
# ).to("cpu").eval()

# processor = Gemma3Processor.from_pretrained(model_id, local_files_only=True, cache_dir=cache_dir) 

# # AutoProcessor.from_pretrained(model_id, local_files_only=True, cache_dir=cache_dir)

# # Download image and open with PIL
# img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
# img = Image.open(requests.get(img_url, stream=True).raw)

# messages = [
#     {
#         "role": "system",
#         "content": [{"type": "text", "text": "You are a helpful assistant."}]
#     },
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
#             {"type": "text", "text": "Describe this image in detail."}
#         ]
#     }
# ]
# inputs = processor.apply_chat_template(
#     messages, add_generation_prompt=True, tokenize=True,
#     return_dict=True, return_tensors="pt"
# ).to("cpu", dtype=torch.bfloat16)


# #inputs["input_ids"]  [batch_size, token_size]  All text/image(<image_soft_token> place_holder, id=262144 ) token
# #inputs["attention_mask"]  [batch_size, token_size]
# #Note: The image in the original input phase is replaced by 256 <image_sof_token> place holder
# #inputs["token_type_id"] [batch, token_size]  0 if is a text token, 1 if is image(<image_soft_token>) place holder token
# #inputs["pixel_values"]  [Sum of image number from all batch, 3(RGB),  image_size, image_size]

# input_len = inputs["input_ids"].shape[-1]

# with torch.inference_mode():
#     generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
#     generation = generation[0][input_len:]

# decoded = processor.decode(generation, skip_special_tokens=True)
# print(decoded)

from PIL.ImageFile import ImageFile
from GemmaLearn.common_module import numpy_to_tensor, tensor_to_numpy
from GemmaLearn.modeling_gemma3_learn import Gemma3ForConditionalGenerationLearn, Gemma3AttentionLearn, Gemma3MLPLearn 
from GemmaLearn.modeling_gemma3_learn import Gemma3RotaryEmbeddingLearn, Gemma3RMSNormMultiHeadLearn, Gemma3RMSNormLearn,Gemma3TextScaledWordEmbeddingLearn
from GemmaLearn.modeling_gemma3_learn import Gemma3DecoderLayerLearn, Gemma3TextModelLearn
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, Gemma3Processor, GemmaTokenizerFast

from PIL import Image
import requests
import torch
from torch import Tensor
import os
from typing import Dict
from safetensors.torch import safe_open
from safetensors.torch import save_file

from transformers.utils import logging
logging.set_verbosity_info()

import logging as py_logging



model_id = "google/gemma-3-4b-it"
# model_id = "google/gemma-3-27b-it"
# model_id = "google/gemma-3-12b-it"
# Set cache directory to the directory where this script is located
script_dir: str = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(script_dir, "hf_cache")
if model_id == "google/gemma-3-4b-it":
    model_dir = os.path.join(cache_dir, "models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767")
    model_dir = os.path.join(cache_dir, "models--google--gemma-3-4b-it/snapshots/dequant-q40")    
    model_dir = os.path.join(cache_dir,"models--google--gemma-3-4b-it/snapshots/dequant-q4k-unsloth")        
# `elif model_id == "google/gemma-3-12b-it":
#     model_dir = os.path.join(cache_dir, "models--google--gemma-3-12b-it")    
# elif model_id == "google/gemma-3-27b-it":
#     model_dir = os.path.join(cache_dir, "models--google--gemma-3-27b-it")`
else:
    model_dir = None


def is_model_downloaded(model_dir):
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.startswith("pytorch_model") or file.endswith(".safetensors"):
                return True
    return False

if not is_model_downloaded(model_dir=os.path.join(cache_dir, "models--google--gemma-3-4b-it")): #download from cloud if not found
    print("Model not found locally. Downloading...")
    Gemma3ForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_dir)
    AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    print("Download complete.")

# Load model and processor
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_dir, local_files_only=True
).eval()

model_learn = Gemma3ForConditionalGenerationLearn.from_pretrained(
    model_dir, local_files_only=True
).eval()



 
 
log_file_path = os.path.join(script_dir, "transformers_log.txt")
file_handler = py_logging.FileHandler(log_file_path)
file_handler.setLevel(py_logging.INFO)
logging.get_logger().addHandler(file_handler)


# # force type casting to align with the safetensor 
# if model_dir.endswith("dequant-q4k-unsloth") or model_dir.endswith("dequant-q40") :
#     # do it for all of the other dequantize modes
#     safe_tensor_path = os.path.join(model_dir, "model-00001-of-00001.safetensors")
#     tensor_dtypes = {}
#     with safe_open(safe_tensor_path, framework="pt", device="cpu") as f:
#         for key in f.keys():
#             tensor_dtypes[ key] = f.get_tensor(key).dtype
    
#     for name, param in model.named_parameters():
#         if name.startswith("model.language_model.model."):
#             remain_name = name.split("model.language_model.model.")[1]
#             param.data = param.data.to( tensor_dtypes["language_model." + remain_name]   )
#         elif name.startswith("model.vision_tower"):
#             remain_name = name[6:]
#             param.data = param.data.to( tensor_dtypes[ remain_name]   )
#         elif name.startswith("model.multi_modal_projector"):
#             remain_name = name[6:]
#             param.data = param.data.to( tensor_dtypes[ remain_name]   )
#         elif name.startswith('model.language_model.'):
#             remain_name = name[len('model.language_model.'):]
#             param.data = param.data.to( tensor_dtypes["language_model.model." +  remain_name]   )
#         else:
#             raise Exception(f"Warning: {name} not found in safetensor")

# # sanity check
# for name, param in model.named_parameters():
#     print(name, param.dtype)





processor = Gemma3Processor.from_pretrained(model_dir, local_files_only=True)
# Download image and open with PIL
img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
img2_url  =  "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"
img3_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
img4_url = "https://www.nasa.gov/wp-content/uploads/2025/06/29242855965-1388a62ecc-o.jpg"
img: ImageFile = Image.open(requests.get(url=img_url, stream=True).raw)
img2 = Image.open(requests.get(url=img2_url, stream=True).raw)
img3 = Image.open(requests.get(url=img3_url, stream=True).raw)
img4 = Image.open(requests.get(url=img4_url, stream=True).raw)
# Create the prompt
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img_url},
            {"type": "image", "image": img2_url},            
            {"type": "image", "image": img3_url},         
            {"type": "image", "image": img4_url},    
            {"type": "text", "text": "Describe both image in detail, maybe also what are commona and difference among those images? Also, could they be AI generated"}     
        ]
    }
]

# messages = [
#     {
#         "role": "system",
#         "content": [{"type": "text", "text": "You are a helpful assistant."}]
#     },
#     {
#         "role": "user",
#         "content": [
#             # {"type": "image", "image": img_url},
#             {"type": "text", "text": "Show the math eqaution of Ampere's law in latex with explanation." }     
#         ]
#     }
# ]

def save_attention_module(common_header:str, module:Gemma3AttentionLearn, output_dict):
    output_dict[common_header + "k_embed_multi_head_cache"] = numpy_to_tensor(module.k_embed_multi_head_cache, dtype=torch.bfloat16)  
    output_dict[common_header + "v_embed_multi_head_cache"] = numpy_to_tensor(module.v_embed_multi_head_cache, dtype=torch.bfloat16)
    output_dict[common_header + "attention_res_cache"] = numpy_to_tensor(module.attention_res_cache, dtype=torch.bfloat16)
    output_dict[common_header + "attention_input_debug"] = module.attention_input_debug
    output_dict[common_header + "attention_output_debug"] = module.attention_output_debug


def save_MLP_module(common_header:str, module:Gemma3MLPLearn, output_dict):
    output_dict[common_header + "mlp_input_cache"] = module.mlp_input_cache
    output_dict[common_header + "cache_gate_proj_res"] = module.cache_gate_proj_res
    output_dict[common_header + "cache_up_proj_res"] = module.cache_up_proj_res
    output_dict[common_header + "cache_gelu_res"] = module.cache_gelu_res
    output_dict[common_header + "cache_down_proj_res_np"] = module.cache_down_proj_res_np
    


def save_tensor_dict_to_safetensor(file_name, data_dict):
    save_file(data_dict, file_name)
    

# def save_prefill_data_to_file(filename:str, model:Gemma3ForConditionalGenerationLearn):
    
#     tensors_to_save :Dict[str,Tensor ] = {}
#     gemma3_model = model.model
#     # already applied the scaling and so forth
#     tensors_to_save["input_embedding"] = gemma3_model.input_embedding_debug_cache
    
#     language_model = gemma3_model.language_model
    
    
    



# debug saving

tensor_py = torch.tensor([1.0, -2.5, 3.14, -4.0, 5.5], dtype=torch.bfloat16)
save_file({"test_tensor":tensor_py}, "model.safetensors")

# Process inputs
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to("cpu", dtype=torch.bfloat16)


# #inputs["input_ids"]  [batch_size, token_size]  All text/image(<image_soft_token> place_holder, id=262144 ) token
# #inputs["attention_mask"]  [batch_size, token_size]
# #Note: The image in the original input phase is replaced by 256 <image_sof_token> place holder
# #inputs["token_type_id"] [batch, token_size]  0 if is a text token, 1 if is image(<image_soft_token>) place holder token
# #inputs["pixel_values"]  [Sum of image number from all batch, 3(RGB),  image_size, image_size], This make sense because all image is going into the visition transformer first
# Thus, "sum of image number from all batch" is ineffectively batch_size for the vision transformer


# Initialize decoding loop
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
pixel_values =   None  if "pixel_values" not in inputs else    inputs["pixel_values"]

input_len = input_ids.shape[-1]
max_new_tokens = 200



input_ids_ref = input_ids.clone()
input_ids_learn = input_ids.clone()
# Generation loop
for step in range(max_new_tokens):
    with torch.inference_mode():
        if pixel_values is None:
            outputs_learn = model_learn(
                input_ids=input_ids_learn,
                attention_mask=attention_mask,    
                # output_attentions = True,
                # output_hidden_states = True,
                # return_dict= True,                     
            )
            outputs_ref = model(
                input_ids=input_ids_ref,
                attention_mask=attention_mask,

            )
        else:
            outputs_learn = model_learn(
                input_ids=input_ids_learn,
                attention_mask=attention_mask,
                pixel_values=pixel_values,            
            )
            
            outputs_ref = model(
                input_ids=input_ids_ref,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )            
            
        if step ==0 or step == 1:
            # for now, just save the attention debug info during inference
            language_model:Gemma3TextModelLearn = model_learn.model.language_model
            sliding_window_atten_example = language_model.layers[0].self_attn
            sliding_attn_dict = {}
            save_attention_module("",sliding_window_atten_example,  sliding_attn_dict)
            save_tensor_dict_to_safetensor(f"sliding_attention_{step}.safetensors", sliding_attn_dict)
            
            
            full_atten_example = language_model.layers[5].self_attn
            full_attn_dict = {}
            save_attention_module("", full_atten_example, full_attn_dict)
            save_tensor_dict_to_safetensor(f"full_attention_{step}.safetensors",full_attn_dict )
            
            
            
        next_token_logits_ref = outputs_ref.logits[:, -1, :]  # [batch, vocab_size]
        next_token_id_ref = torch.argmax(next_token_logits_ref, dim=-1, keepdim=True)  # [batch, 1]

        next_token_logits_learn = outputs_learn.logits[:, -1, :]
        next_token_id_learn = torch.argmax(next_token_logits_learn, dim=-1, keepdim=True)  # [batch, 1]

        # assert similar for two logits and id_ref
        
        # if torch.allclose(next_token_logits_ref, next_token_logits_learn, atol=1e-1, rtol=1e-1):
        #     print("Logits match closely!")
        # else:
        #     # Compute difference stats
        #     abs_diff = (next_token_logits_ref - next_token_logits_learn).abs()
        #     max_diff = abs_diff.max().item()
        #     mean_diff = abs_diff.mean().item()
        #     print("Logits differ")
        #     print(f"Max difference: {max_diff:.6f}")
        #     print(f"Mean absolute difference: {mean_diff:.6f}")
        #     raise Exception("Out of tolerance")
        assert torch.eq(next_token_id_ref, next_token_id_learn)  # make sure they are the same token 

    # Append the new token
    input_ids_learn = torch.cat([input_ids_learn, next_token_id_learn], dim=-1)
    input_ids_ref = torch.cat([input_ids_ref, next_token_id_ref], dim=-1)
    attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id_ref)], dim=-1) # should be the same

    # Decode the newly generated part
    generated_so_far_learn = input_ids_learn[0][input_len:]  # Get only the new tokens
    generated_so_far_ref = input_ids_ref[0][input_len:]  # Get only the new tokens

    decoded_learn = processor.decode(generated_so_far_learn, skip_special_tokens=True)
    decoded_ref = processor.decode(generated_so_far_ref, skip_special_tokens=True)

    assert decoded_learn == decoded_ref
    print(f"Step {step + 1}: {decoded_learn}")
    print(f"Step {step + 1}: {decoded_ref}")

    # # Early stop if EOS token generated
    # if next_token_id_ref.item() == processor.tokenizer.eos_token_id:
    #     break

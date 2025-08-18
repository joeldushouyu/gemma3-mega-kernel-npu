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
    # model_dir = os.path.join(cache_dir, "models--google--gemma-3-4b-it/snapshots/dequant-q40")    
    # model_dir = os.path.join(cache_dir,"models--google--gemma-3-4b-it/snapshots/dequant-q4k-unsloth")        
# `elif model_id == "google/gemma-3-12b-it":
#     model_dir = os.path.join(cache_dir, "models--google--gemma-3-12b-it")    
# elif model_id == "google/gemma-3-27b-it":
#     model_dir = os.path.join(cache_dir, "models--google--gemma-3-27b-it")`
else:
    model_dir = None


# Load model and processor
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_dir, local_files_only=True
).eval()

model_learn = Gemma3ForConditionalGenerationLearn.from_pretrained(
    model_dir, local_files_only=True
).eval()



 

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
            {"type": "image", "image": "./text1.png"},                
            {"type": "text", "text": "Can you extract the text from the image"}      
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
#             # {"type": "text", "text": "Hello!" }              
#         ]
#     }
# ]



# messages = [
#     {
#         "role": "system",
#         "content": [{"type": "text", "text": "You are a helpful assistant."}]
#     },
#     {
#         "role": "user",
#         "content": [
   
#             {"type": "text", "text": "hi"}     
#         ]
#     }
# ]


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
max_new_tokens = 500



input_ids_ref = input_ids.clone()
input_ids_learn = input_ids.clone()
# Generation loop
for step in range(max_new_tokens):
    with torch.inference_mode():
        if pixel_values is None:
            outputs_ref = model(
                input_ids=input_ids_ref,
                attention_mask=attention_mask,

            )
        else:

            outputs_ref = model(
                input_ids=input_ids_ref,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )            
            

            
            
        next_token_logits_ref = outputs_ref.logits[:, -1, :]  # [batch, vocab_size]
        next_token_id_ref = torch.argmax(next_token_logits_ref, dim=-1, keepdim=True)  # [batch, 1]


    # Append the new token
    input_ids_ref = torch.cat([input_ids_ref, next_token_id_ref], dim=-1)
    attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id_ref)], dim=-1) # should be the same

    # Decode the newly generated part

    generated_so_far_ref = input_ids_ref[0][input_len:]  # Get only the new tokens

    decoded_ref = processor.decode(generated_so_far_ref, skip_special_tokens=True)


    print(f"Step {step + 1}: {decoded_ref}")

    # # Early stop if EOS token generated
    # if next_token_id_ref.item() == processor.tokenizer.eos_token_id:
    #     break

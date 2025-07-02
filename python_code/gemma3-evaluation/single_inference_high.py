from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import requests
import torch
from PIL.ImageFile import ImageFile
from GemmaLearn.modeling_gemma3_learn import Gemma3ForConditionalGenerationLearn 
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, Gemma3Processor, GemmaTokenizerFast
from PIL import Image
import requests
import torch
import os
model_id = "google/gemma-3-4b-it"

# model = Gemma3ForConditionalGeneration.from_pretrained(
#     model_id, device_map="auto"
# ).eval()

# processor = AutoProcessor.from_pretrained(model_id)





model_id = "google/gemma-3-4b-it"

# Set cache directory to the directory where this script is located
script_dir: str = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(script_dir, "hf_cache")
if model_id == "google/gemma-3-4b-it":
    model_dir = os.path.join(cache_dir, "models--google--gemma-3-4b-it")
    # model_dir = os.path.join(cache_dir, "models--google--gemma-3-4b-it-dequant-q40")    
    # model_dir = os.path.join(cache_dir, "models--google--gemma-3-4b-it-dequant-q4k-unsloth")        
else:
    model_dir = None
def is_model_downloaded(model_dir):
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.startswith("pytorch_model") or file.endswith(".safetensors"):
                return True
    return False

if not is_model_downloaded(model_dir):
    print("Model not found locally. Downloading...")
    Gemma3ForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_dir)
    AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    print("Download complete.")


# # for testing dequant model
# model_id = "google/gemma-3-4b-it"
# script_dir = os.path.dirname(os.path.abspath(__file__))
# cache_dir: str = os.path.join(script_dir, "gemma3-dequant-model")


# Load model and processor
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, local_files_only=True, cache_dir=cache_dir
).to("cpu").eval()


#TODO: our own processor
processor = Gemma3Processor.from_pretrained(model_id, local_files_only=True, cache_dir=cache_dir)


messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt", do_pan_and_scan=True
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)
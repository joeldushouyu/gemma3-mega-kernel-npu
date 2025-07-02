from transformers import AutoProcessor, Gemma3ForConditionalGeneration, Gemma3Processor, Gemma3ImageProcessor, GemmaTokenizerFast, GemmaTokenizer
from PIL import Image
import requests
import torch
import os

model_id = "google/gemma-3-4b-it"

# Set cache directory to the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(script_dir, "hf_cache")
model_dir = os.path.join(cache_dir, "models--google--gemma-3-4b-it")

# Check if model is downloaded
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
# Load model and processor from local cache, force CPU
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, local_files_only=True, cache_dir=cache_dir
).to("cpu").eval()

processor = Gemma3Processor.from_pretrained(model_id, local_files_only=True, cache_dir=cache_dir) 
    print("Download complete.")


# Download images and open with PIL
img_url_1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
img_url_2 = "https://www.nasa.gov/wp-content/uploads/2025/06/ksc-2013-3063orig.jpg" # Example second image

img1 = Image.open(requests.get(img_url_1, stream=True).raw)
img2 = Image.open(requests.get(img_url_2, stream=True).raw)


# Prepare messages for batch processing
# Each element in the outer list corresponds to a single batch item (a conversation)
batch_messages = [
    # Conversation 1
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in detail."},            
                {"type": "image", "image": img1},
                {"type": "text", "text": "What do you see?"}
            ]
        }
    ],
    # Conversation 2
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image carefully."},            
                {"type": "image", "image": img2},
                {"type": "text", "text": "Can you identify the main objects?"}
            ]
        }
    ],
    # Conversation 3 (example with multiple images in one turn, similar to your original)
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe these two images."},            
                {"type": "image", "image": img1},
                {"type": "image", "image": img2},
                {"type": "text", "text": "Also, what are the key differences?"}
            ]
        }
    ]
]


# Apply chat template for batch
# The processor automatically handles padding and batching for you
inputs = processor.apply_chat_template(
    batch_messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt", padding=True
).to("cpu", dtype=torch.bfloat16)


# Keep track of original input lengths for each item in the batch
# This is crucial for slicing the generated tokens later
input_lengths = [len(item) for item in inputs["input_ids"]]


with torch.inference_mode():
    # The model will generate for all items in the batch simultaneously
    generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)

# Decode each generated sequence
decoded_outputs = []
for i in range(generation.shape[0]): # Iterate through each item in the batch
    # Slice the generated tokens to remove the input tokens
    generated_tokens_for_item = generation[i][input_lengths[i]:]
    decoded_text = processor.decode(generated_tokens_for_item, skip_special_tokens=True)
    decoded_outputs.append(decoded_text)

# Print the results for each item in the batch
for i, output in enumerate(decoded_outputs):
    print(f"--- Output for Batch Item {i+1} ---")
    print(output)
    print("\n")

# from transformers import AutoModelForCausalLM, AutoTokenizer
 
# model_name = "openai/gpt-oss-20b"
 
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
# )
 
# messages = [
#     {"role": "system", "content": "Be concise"},
#     {"role": "user", "content": "Explain Ampere's law"},
# ]
 
# inputs = tokenizer.apply_chat_template(
#     messages,
#     add_generation_prompt=True,
#     return_tensors="pt",
#     return_dict=True,
# ).to(model.device)
 
# generated = model.generate(**inputs, max_new_tokens=500)
# print(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1] :]))


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GptOssForCausalLM
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from gptOSSLearn.modeling_gpt_oss_learn import GptOssForCausalLMLearn

import os







model_name = "openai/gpt-oss-20b"

script_dir: str = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(script_dir, "hf_cache")






# model_dir = os.path.join(cache_dir, "models--openai--gpt-oss-20b/snapshots/d666cf3b67006cf8227666739edf25164aaffdeb")
# def is_model_downloaded(model_dir):
#     for root, dirs, files in os.walk(model_dir):
#         for file in files:
#             if file.startswith("pytorch_model") or file.endswith(".safetensors"):
#                 return True
#     return False


# print("model_dir:", model_dir)
# if not is_model_downloaded(model_dir): #download from cloud if not found
#     print("Model not found locally. Downloading...")
#     model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
#     tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    
    
# Load both model and tokenizer from HuggingFace directory
# The HF directory contains all necessary files: config.json, model weights, tokenizer files, etc.
hf_model_dir = os.path.join(cache_dir, "models--openai--gpt-oss-20b/snapshots/d666cf3b67006cf8227666739edf25164aaffdeb")
gguf_model_dir = os.path.join(cache_dir, "gpt-oss-20b-Q4_1.gguf")


print("Load model from HF directory:", hf_model_dir)
print("Load tokenizer from HF directory:", hf_model_dir)

# Load tokenizer from HuggingFace directory
tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, local_files_only=True)

# Check if CUDA is available and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# # Load model from HuggingFace directory (this will work reliably)
model = GptOssForCausalLMLearn.from_pretrained(pretrained_model_name_or_path=hf_model_dir, local_files_only=True).eval()

try:
    model = model.to(device)
except NotImplementedError:
    # Handle meta tensors
    model = model.to_empty(device=device)

print(f"model device: {next(model.parameters()).device}")
# print(f"model_learn device: {next(model_learn.parameters()).device}")




# messages = [
#     {"role": "system", "content": "Be concise"},
#     {"role": "user", "content": "Explain Ampere's law"},
# ]


# and the assistant 'content' field for the final response. Do NOT include
# literal channel tags like '<|channel|>analysis...' in these strings —
# the Jinja template will handle rendering the analysis vs final text.
messages = [
    {"role": "system", "content": "Be concise"},
    {"role": "user", "content": "Explain Ampere's law"},
    {
        "role": "assistant",
        # 'thinking' is used by the template as the analysis channel (internal)
        "thinking": (
        ),
        # 'content' is the assistant's final output that will be shown to the user
        "content": (
            "Ampere's law: the circulation of the magnetic field B around a closed loop "
        ),
    },
]





inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)

# Get device from model parameters
model_device = next(model.parameters()).device
inputs = {k: v.to(model_device) for k, v in inputs.items()}

input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
# greedy decoding loop
max_new_tokens = 20
generated_ids = input_ids.clone()

for step in range(max_new_tokens):
    with torch.no_grad():
        outputs_ref = model(input_ids=generated_ids, attention_mask=torch.ones_like(generated_ids))
        logits_ref = outputs_ref.logits[:, -1, :]  # take last token logits
        next_token_id_ref = torch.argmax(logits_ref, dim=-1, keepdim=True)  # greedy pick

    
    generated_ids = torch.cat([generated_ids, next_token_id_ref], dim=-1)

    # decode all message history so far
    decoded_history = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Step {step+1}\n {decoded_history}")

    # optional: stop if EOS token
    if next_token_id_ref.item() == tokenizer.eos_token_id:
        break

print("Running the learn module")


# free model
del model


# final text
print("\nFinal output:\n")

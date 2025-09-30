import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GptOssForCausalLM
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from gptOSSLearn.modeling_gpt_oss_learn import GptOssForCausalLMLearn

import os
from safetensors.torch import save_file






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
hf_model_dir = os.path.join(cache_dir, "models--openai--gpt-oss-20b/snapshots/extract_safetensors")
gguf_model_dir = os.path.join(cache_dir, "gpt-oss-20b-Q4_1.gguf")


print("Load model from HF directory:", hf_model_dir)
print("Load tokenizer from HF directory:", hf_model_dir)

# Load tokenizer from HuggingFace directory
tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, local_files_only=True)

# Check if CUDA is available and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# # Load model from HuggingFace directory (this will work reliably)
model = GptOssForCausalLM.from_pretrained(pretrained_model_name_or_path=hf_model_dir, local_files_only=True).eval()

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
    {"role": "system", "content": "You are a helpful agent that are good at math and physics, you are a tutor to a college student major in engineering."},
    {"role": "user", "content": "Can you explain Ampere's law and give out the equation in latex format. Give as many details as possible.\n Also, give explanation regarding when the linear integral assumption fails."},

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
max_new_tokens = 50
generated_ids = input_ids.clone()

# for step in range(max_new_tokens):
#     with torch.no_grad():
#         outputs_ref = model(input_ids=generated_ids, attention_mask=torch.ones_like(generated_ids))
#         logits_ref = outputs_ref.logits[:, -1, :]  # take last token logits
#         next_token_id_ref = torch.argmax(logits_ref, dim=-1, keepdim=True)  # greedy pick

    
#     generated_ids = torch.cat([generated_ids, next_token_id_ref], dim=-1)

#     # decode all message history so far
#     decoded_history = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
#     print(f"Step {step+1}\n {decoded_history}")

#     # optional: stop if EOS token
#     if next_token_id_ref.item() == tokenizer.eos_token_id:
#         break

# print("Running the learn module")


# free model
del model


# first, open the "sample_token_id.bin" as input_ids interpret as int32
import numpy as np
input_ids_from_file = np.fromfile("sample_token_id.bin", dtype=np.int32)
input_ids_from_file = torch.from_numpy(input_ids_from_file).unsqueeze(0).to(device)  # add batch dimension and move to device
print("input_ids_from_file shape:", input_ids_from_file.shape)
print("input_ids_from_file:", input_ids_from_file)
# replace the input_ids with the file

# next, run the learn version and compare the generated_ids with 
generate_ids_learn = input_ids_from_file.clone()




model_learn = GptOssForCausalLMLearn.from_pretrained(pretrained_model_name_or_path=hf_model_dir, local_files_only=True).eval()
model_learn = model_learn.to(device)


decode_data_to_save_tensors={}
prefill_data_to_save_tensors = {}

prefill_data_to_save_tensors["input_ids"] = generate_ids_learn.clone()

for step in range(max_new_tokens):
    with torch.no_grad():
        if step == 0:
            outputs_learn = model_learn(input_ids=generate_ids_learn, attention_mask=torch.ones_like(generate_ids_learn),
                                        decode_data_to_save_tensors=None,  # only save the decode data for now
                                        prefill_data_to_save_tensors=prefill_data_to_save_tensors
                                        )
            for k, v in prefill_data_to_save_tensors.items():
                # if isinstance(v, torch.Tensor):
                prefill_data_to_save_tensors[k] = v.contiguous()
            save_file(prefill_data_to_save_tensors, "prefill_data_debug.safetensors")
        elif step == 1:
            outputs_learn = model_learn(input_ids=generate_ids_learn, attention_mask=torch.ones_like(generate_ids_learn),
                                        decode_data_to_save_tensors=decode_data_to_save_tensors  # only save the decode data for now
                                        )
            # makesure all data is contigous()
            for k, v in decode_data_to_save_tensors.items():
                if isinstance(v, torch.Tensor):
                    decode_data_to_save_tensors[k] = v.contiguous()
            save_file(decode_data_to_save_tensors, "prefill_data_debug.safetensors")   
        else:
            outputs_learn = model_learn(input_ids=generate_ids_learn, attention_mask=torch.ones_like(generate_ids_learn),
                                        decode_data_to_save_tensors=None  # only save the decode data for now
                                        )
        logits_learn = outputs_learn.logits[:, -1, :]  # take last token logits

        next_token_id_learn = torch.argmax(logits_learn, dim=-1, keepdim=True)  # greedy pick

    generate_ids_learn = torch.cat([generate_ids_learn, next_token_id_learn], dim=-1)
    # Compare the current token generated by model_learn with the corresponding token from generated_ids
    if step < generated_ids.shape[1] - input_ids.shape[1]:
        # Offset by input prompt length
        ref_token = generated_ids[0, input_ids.shape[1] + step]
        # assert torch.eq(generate_ids_learn[0, -1], ref_token), f"Mismatch at step {step}: {generate_ids_learn[0, -1]} vs {ref_token}"
    # decode all message history so far
    decoded_history_learn = tokenizer.decode(generate_ids_learn[0], skip_special_tokens=True)
    print(f"Step {step+1}\n {decoded_history_learn}")

    # optional: stop if EOS token
    if next_token_id_learn.item() == tokenizer.eos_token_id:
        break

# final text
print("\nFinal output:\n")
 


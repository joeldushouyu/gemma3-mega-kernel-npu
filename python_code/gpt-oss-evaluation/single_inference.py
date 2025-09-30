# # from transformers import AutoModelForCausalLM, AutoTokenizer
 
# # model_name = "openai/gpt-oss-20b"
 
# # tokenizer = AutoTokenizer.from_pretrained(model_name)
# # model = AutoModelForCausalLM.from_pretrained(
# #     model_name,
# # )
 
# # messages = [
# #     {"role": "system", "content": "Be concise"},
# #     {"role": "user", "content": "Explain Ampere's law"},
# # ]
 
# # inputs = tokenizer.apply_chat_template(
# #     messages,
# #     add_generation_prompt=True,
# #     return_tensors="pt",
# #     return_dict=True,
# # ).to(model.device)
 
# # generated = model.generate(**inputs, max_new_tokens=500)
# # print(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1] :]))


# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, GptOssForCausalLM
# from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
# from gptOSSLearn.modeling_gpt_oss_learn import GptOssForCausalLMLearn
# from transformers import 
# import os
# import torch.nn.functional as F






# model_name = "openai/gpt-oss-20b"

# script_dir: str = os.path.dirname(os.path.abspath(__file__))
# cache_dir = os.path.join(script_dir, "hf_cache")






# # model_dir = os.path.join(cache_dir, "models--openai--gpt-oss-20b/snapshots/d666cf3b67006cf8227666739edf25164aaffdeb")
# # def is_model_downloaded(model_dir):
# #     for root, dirs, files in os.walk(model_dir):
# #         for file in files:
# #             if file.startswith("pytorch_model") or file.endswith(".safetensors"):
# #                 return True
# #     return False


# # print("model_dir:", model_dir)
# # if not is_model_downloaded(model_dir): #download from cloud if not found
# #     print("Model not found locally. Downloading...")
# #     model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
# #     tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    
    
# # Load both model and tokenizer from HuggingFace directory
# # The HF directory contains all necessary files: config.json, model weights, tokenizer files, etc.
# # hf_model_dir = os.path.join(cache_dir, "models--openai--gpt-oss-20b/snapshots/d666cf3b67006cf8227666739edf25164aaffdeb")

# hf_model_dir = os.path.join(cache_dir, "models--openai--gpt-oss-20b/snapshots/extract_safetensors")

# print("Load model from HF directory:", hf_model_dir)
# print("Load tokenizer from HF directory:", hf_model_dir)

# # Load tokenizer from HuggingFace directory
# tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, local_files_only=True)

# # Check if CUDA is available and set device
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

# # # Load model from HuggingFace directory (this will work reliably)
# #model = GptOssForCausalLMLearn.from_pretrained(pretrained_model_name_or_path=hf_model_dir, local_files_only=True).eval()
# # for now debugging
# model = GptOssForCausalLM.from_pretrained(pretrained_model_name_or_path=hf_model_dir, local_files_only=True).eval()
# try:
#     model = model.to(device)
# except NotImplementedError:
#     # Handle meta tensors
#     model = model.to_empty(device=device)

# print(f"model device: {next(model.parameters()).device}")
# # print(f"model_learn device: {next(model_learn.parameters()).device}")




# messages = [
#     {"role": "system", "content": "You are are helpful agent"},
#     {"role": "user", "content": "Explain Ampere's law and give the equation in latex \n"},
# ]


# # # and the assistant 'content' field for the final response. Do NOT include
# # # literal channel tags like '<|channel|>analysis...' in these strings —
# # # the Jinja template will handle rendering the analysis vs final text.
# # messages = [
# #     {"role": "system", "content": "Be concise"},
# #     {"role": "user", "content": "Explain Ampere's law"},
# #     {
# #         "role": "assistant",
# #         # 'thinking' is used by the template as the analysis channel (internal)
# #         "thinking": (
# #         ),
# #         # 'content' is the assistant's final output that will be shown to the user
# #         "content": (
# #             "Ampere's law: the circulation of the magnetic field B around a closed loop "
# #         ),
# #     },
# # ]





# inputs = tokenizer.apply_chat_template(
#     messages,
#     add_generation_prompt=True,
#     return_tensors="pt",
#     return_dict=True,
# )

# # Get device from model parameters
# model_device = next(model.parameters()).device
# inputs = {k: v.to(model_device) for k, v in inputs.items()}

# input_ids = inputs["input_ids"]
# attention_mask = inputs["attention_mask"]
# # greedy decoding loop
# max_new_tokens = 100
# generated_ids = input_ids.clone()

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


# # free model
# del model


# # final text
# print("\nFinal output:\n")

# # # --- Generation Parameters ---
# # max_new_tokens = 100
# # temperature = 0.8  # Slightly increased temperature
# # top_k = 50
# # top_p = 0.9
# # repetition_penalty = 1.15 # Value > 1.0. Common values are 1.1 to 1.2

# # generated_ids = input_ids.clone()

# # for step in range(max_new_tokens):
# #     with torch.no_grad():
# #         current_attention_mask = torch.ones_like(generated_ids)
# #         outputs = model(input_ids=generated_ids, attention_mask=current_attention_mask)
# #         next_token_logits = outputs.logits[:, -1, :]

# #         # *** NEW: APPLY REPETITION PENALTY ***
# #         if repetition_penalty > 1.0:
# #             # Get the token IDs of the generated sequence
# #             prev_output_tokens = generated_ids[0]
# #             # Create a mask for the logits that correspond to previously generated tokens
# #             score_mask = torch.zeros_like(next_token_logits[0])
# #             score_mask.scatter_(0, prev_output_tokens, 1)
            
# #             # Apply the penalty
# #             # For positive logits, divide by the penalty. For negative, multiply.
# #             # This ensures the penalty always reduces the likelihood of the token.
# #             penalty_mask = torch.where(next_token_logits[0] > 0, 1.0 / repetition_penalty, repetition_penalty)
# #             penalized_logits = torch.where(score_mask.bool(), next_token_logits[0] * penalty_mask, next_token_logits[0])
# #             next_token_logits[0] = penalized_logits
# #         # ****************************************

# #         # 1. Apply Temperature
# #         scaled_logits = next_token_logits / temperature

# #         # 2. Apply Top-K Filtering
# #         if top_k > 0:
# #             # ... (code for Top-K is the same)
# #             top_k_values, top_k_indices = torch.topk(scaled_logits, top_k)
# #             mask = torch.full_like(scaled_logits, -float('Inf'))
# #             mask.scatter_(1, top_k_indices, top_k_values)
# #             scaled_logits = mask

# #         # 3. Apply Top-P (Nucleus) Filtering
# #         if top_p > 0.0:
# #             # ... (code for Top-P is the same)
# #             sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
# #             cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
# #             sorted_indices_to_remove = cumulative_probs > top_p
# #             sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
# #             sorted_indices_to_remove[..., 0] = 0
# #             indices_to_remove = torch.zeros_like(scaled_logits, dtype=torch.bool).scatter_(
# #                 dim=1, index=sorted_indices, src=sorted_indices_to_remove
# #             )
# #             scaled_logits[indices_to_remove] = -float('Inf')

# #         # 4. Sample from the filtered distribution
# #         probs = F.softmax(scaled_logits, dim=-1)
# #         next_token_id = torch.multinomial(probs, num_samples=1)

# #     generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
# #     decoded_history = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
# #     print(f"Step {step+1}\n {decoded_history}")

# #     if next_token_id.item() == tokenizer.eos_token_id:
# #         break


# # # --- Cleanup and Final Output ---
# # print("\nRunning the learn module (or other post-processing steps)...")

# # # free model
# # del model

# # # final text
# # print("\nFinal output:\n")
# # print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))


from transformers import Op
from transformers import AutoTokenizer, AutoModelForCausalLM
#from model_list import MODEL_LIST
import torch
import numpy as np
import sys
import os
from einops import rearrange

#current_model = MODEL_LIST[0]
# model_name = current_model.name
# model_tag = current_model.tag
PRINT_COLS = 64
# if name == "main":
torch.set_printoptions(threshold=256, edgeitems=5, linewidth=200)

model = AutoModelForCausalLM.from_pretrained('openbmb/MiniCPM-V-4_5', trust_remote_code=True, attn_implementation="sdpa", torch_dtype=torch.bfloat16)
# print(model)
print(type(model))
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-4_5')

# print(model)
import torch
from transformers import AutoProcessor
from Qwen3Learn.Qwen3Learn import Qwen3VLForConditionalGeneration
# =========================================================
# 1. Load model and processor
# =========================================================
model_name = "Qwen/Qwen3-VL-4B-Instruct"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",    # or "cpu"
    torch_dtype="auto",
)
model.eval()

processor = AutoProcessor.from_pretrained(model_name)

# =========================================================
# 2. Build multimodal chat input
# =========================================================
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://files.softicons.com/download/social-media-icons/simple-icons-by-dan-leech/png/512x512/google.png"},
            {"type": "image", "image": "https://m.media-amazon.com/images/I/71LCi1gwwwL._AC_UF1000,1000_QL80_.jpg"},
            {"type": "text", "text": "Describe the two images."},
        ],
    }
]
# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=1024,top_k=1)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
# # processor handles image loading, vision embeddings, and text tokenization
# inputs = processor.apply_chat_template(
#     messages,
#     tokenize=True,
#     add_generation_prompt=True,
#     return_tensors="pt",
#     return_dict=True,
# )

# # move everything to model device
# inputs = {k: v.to(model.device) for k, v in inputs.items()}

# # =========================================================
# # 3. Prepare decoding variables
# # =========================================================
# max_new_tokens = 1024
# eos_token_id = model.config.eos_token_id

# generated_ids = inputs["input_ids"].clone()

# # =========================================================
# # 4. Greedy decoding loop (preserving multimodal inputs)
# # =========================================================
# print("Starting greedy decoding...\n")

# for step in range(max_new_tokens):
#     with torch.no_grad():
#         # clone the full multimodal input dict
#         model_inputs = dict(inputs)
#         # update input_ids to include all generated tokens so far
#         model_inputs["input_ids"] = generated_ids
        
#         # update attention_mask to match the new sequence length
#         if "attention_mask" in model_inputs:
#             current_length = generated_ids.shape[1]
#             original_length = model_inputs["attention_mask"].shape[1]
#             if current_length > original_length:
#                 # extend attention mask with 1s for new tokens
#                 batch_size = model_inputs["attention_mask"].shape[0]
#                 new_mask = torch.ones(
#                     batch_size, 
#                     current_length - original_length,
#                     dtype=model_inputs["attention_mask"].dtype,
#                     device=model_inputs["attention_mask"].device
#                 )
#                 model_inputs["attention_mask"] = torch.cat([
#                     model_inputs["attention_mask"], 
#                     new_mask
#                 ], dim=1)

#         # forward pass
#         outputs = model(**model_inputs)

#         # take logits of the last generated token
#         logits = outputs.logits[:, -1, :]
#         next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

#     # append new token
#     generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

#     # decode partial text for monitoring
#     partial_text = processor.tokenizer.decode(
#         generated_ids[0],
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=False,
#     )

#     print(f"[Step {step+1}] → {processor.tokenizer.decode(next_token_id[0])}")
#     print(partial_text)
#     print("=" * 80)

#     if next_token_id.item() == eos_token_id:
#         print("Reached EOS token.")
#         break

# # =========================================================
# # 5. Final output text
# # =========================================================
# final_output = processor.tokenizer.decode(
#     generated_ids[0][inputs["input_ids"].shape[1]:],
#     skip_special_tokens=True,
#     clean_up_tokenization_spaces=False,
# )

# print("\nFinal Output:\n")
# print(final_output)

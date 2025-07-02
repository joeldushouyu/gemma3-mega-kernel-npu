# #https://github.com/huggingface/diffusers/discussions/9319#discussioncomment-10501351
from safetensors.torch import load_file, save_file 
from transformers import AutoTokenizer, AutoModel 

# import torch
# from GemmaLearn.modeling_gemma3_learn import Gemma3ForConditionalGenerationLearn 
# from transformers import AutoProcessor, Gemma3ForConditionalGeneration, Gemma3Processor, GemmaTokenizerFast
# from PIL import Image

# folder_dir  = "./hf_cache/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767"
# load1 = load_file(folder_dir + "/model-00001-of-00002.safetensors") 
# load2 = load_file(folder_dir + "/model-00002-of-00002.safetensors") 

# # Debug 
# print(load1.keys())
# print(load2.keys())

# # If here you got and issue, you may facing file corrupted issue..

# # Once everything looks right lets unpack this 
# merged_state_dict = {**load1, **load2}


# save_file(merged_state_dict, "merged_diffusion_torch.safetensors")


# your_model_name_path = "merged_diffusion_torch.safetensors" 

#  # Now, let's try loading the model to ensure it's valid 
# merged_load = load_file(your_model_name_path)
# # print(merged_load.keys())
# # try: # Load the tokenizer and model 
# #     tokenizer = AutoTokenizer.from_pretrained(your_model_name_path) 
# #     model = AutoModel.from_pretrained(your_model_name_path, from_tf=False, from_safetensors=True)
    
# #     pipeline = StableDiffusionPipeline.from_pretrained(your_model_name_path, torch_dtype=torch.float16) 
# #     print("it is successfully loaded!") 
    
# # except Exception as e: 
# #     print(f"Error: {e}")



import safetensors.torch
import os

def merge_safetensors(file_paths, output_path):
    """
    Merges multiple safetensors files into a single file.

    Args:
        file_paths: A list of paths to the safetensors files to merge.
        output_path: The path to save the merged safetensors file.
    """
    merged_state_dict = {}
    for file_path in file_paths:
        if os.path.exists(file_path):
          try:
            loaded_dict = safetensors.torch.load_file(file_path)
            merged_state_dict.update(loaded_dict)
          except Exception as e:
            print(f"Error loading or updating from {file_path}: {e}")
        else:
          print(f"Warning: File not found: {file_path}")
    if merged_state_dict:
        try:
          safetensors.torch.save_file(merged_state_dict, output_path)
          print(f"Successfully merged safetensors files into {output_path}")
        except Exception as e:
            print(f"Error saving merged file: {e}")
    else:
        print("No data loaded, nothing to save.")

folder_dir  = "./../hf_cache/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767"
load1 = folder_dir + "/model-00001-of-00002.safetensors"
load2 = folder_dir + "/model-00002-of-00002.safetensors"
# Example usage
file_paths_to_merge = [load1, load2]
output_file = "model-00001-of-00001.safetensors"
merge_safetensors(file_paths_to_merge, output_file)
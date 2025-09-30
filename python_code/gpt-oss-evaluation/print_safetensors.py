import torch
import math
from safetensors.torch import load_file

# Assuming the model file is in the root of the workspace
model_path = "/home/shouyud/gpt-oss-mega-kernel/gpt-oss-20b.safetensors"
tensor_key = "model.layers.12.self_attn.sinks.weight"

try:
    # Load the tensors from the .safetensors file
    weights = load_file(model_path)

    # Check if the specific tensor key exists
    if tensor_key in weights:
        # Get the tensor
        weight = weights[tensor_key]
        
        # Print the tensor and its properties
        print(f"Successfully loaded tensor: {tensor_key}")
        print("Tensor value:")
        print(weight)
        print(f"Tensor shape: {weight.shape}")
        print(f"Tensor dtype: {weight.dtype}")
        
        # now 
        # convert to float
        weight_float = weight.float()
        
        # then write to .bin file using numpy as pure binary format
        import numpy as np
        weight_float_np = weight_float.cpu().numpy()
        weight_float_np.tofile(f"{tensor_key}.bin")
        print(f"Tensor written to {tensor_key}.bin as float")

    else:
        print(f"Error: Tensor '{tensor_key}' not found in the file.")
        # You can uncomment the line below to see all available keys
        # print("Available keys:", list(weights.keys()))

except FileNotFoundError:
    print(f"Error: Model file not found at '{model_path}'")
except Exception as e:
    print(f"An error occurred: {e}")
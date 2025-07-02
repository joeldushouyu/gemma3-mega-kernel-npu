import numpy as np
from gguf import GGUFWriter

# Create a random tensor
random_tensor =  np.array([
                        [1.0, 1.1, 1.11, 1.111, 1.1111],
                        [3.1, 3.11, 3.12, 3.13, 3.14],
                        [5.2, 5.21, 5.22, 5.23, 5.24],
                        ],
                        dtype=np.float32
                          )
random_tensor2 =  np.array([
                        [3.1, 3.11, 3.12, 3.13, 3.14],
                        [5.2, 5.21, 5.22, 5.23, 5.24],
                        ],
                        dtype=np.float16
                          )
# Create a GGUF writer instance
gguf_writer = GGUFWriter("single_tensor.gguf", "llama") # Output file name

# Add the tensor data
tensor_name = "output_norm.weight"
gguf_writer.add_tensor(tensor_name, random_tensor)
gguf_writer.add_tensor("output_embd.weight",random_tensor2)

# --- The Missing Steps ---
# Write the header, key-value metadata, and tensor info to the file
gguf_writer.write_header_to_file()
gguf_writer.write_kv_data_to_file()
gguf_writer.write_tensors_to_file()
# --- End of Missing Steps ---

# Write the GGUF file
gguf_writer.close()

print("GGUF file 'single_tensor.gguf' created with one random tensor.")
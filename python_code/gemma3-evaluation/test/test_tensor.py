import torch
import struct

# Original bfloat16 tensor
tensor_bf16 = torch.tensor([1.0, -2.5, 3.14, -4.0, 5.5], dtype=torch.bfloat16)

# Convert to float32 for high-precision printing
tensor_f32 = tensor_bf16.to(torch.float32)

# Function to get the bfloat16 raw bits as uint16
def bf16_to_bits(bf16_tensor):
    return bf16_tensor.view(torch.uint16)

# Print each element with full precision and binary representation
for i, val in enumerate(tensor_bf16):
    val_f32 = float(tensor_f32[i])
    val_bits = bf16_to_bits(val).item()
    print(f"Element {i}:")
    print(f"  bfloat16 bits : 0x{val_bits:04x} (binary: {val_bits:016b})")
    print(f"  float32 value : {val_f32:.30f}")
    print()

import torch
import numpy as np



def get_relativeL2(y: np.ndarray, y_ref: np.ndarray) -> float:
    rmse = np.sqrt(np.mean((y - y_ref) ** 2))
    ref_norm = np.sqrt(np.mean(y_ref ** 2))
    return rmse / ref_norm

def get_relativeL1(y: np.ndarray, y_ref: np.ndarray) -> float:
    l1 = np.sum(np.abs(y - y_ref))
    ref_sum = np.sum(np.abs(y_ref))
    return l1 / ref_sum

def get_rmse(y: np.ndarray, y_ref: np.ndarray) -> float:
    return np.sqrt(np.mean((y - y_ref) ** 2)    ) 

def get_cosine_similarity(y: np.ndarray, y_ref: np.ndarray) -> float:
    # squize to 1D array
    y = y.reshape(-1)
    y_ref = y_ref.reshape(-1)
    dot_product = np.dot(y, y_ref)
    norm_y = np.linalg.norm(y)
    norm_y_ref = np.linalg.norm(y_ref)
    
    return dot_product / (norm_y * norm_y_ref)


def assert_tensor_same_float(a: torch.Tensor, b: torch.Tensor):
    assert a.dtype == b.dtype

    atol, rtol = 1e-1, 1e-2  # looser tolerance for bfloat16

    return torch.allclose(a, b, atol=atol, rtol=rtol)

def tensor_to_numpy(tensor: torch.Tensor, verbose: bool = False) -> np.ndarray:
    """
    Converts a PyTorch tensor to a deep-copied NumPy array.

    Supports tensors on GPU and handles bfloat16 conversion safely.
    bfloat16 will be converted to float32 since NumPy does not support bfloat16.

    Args:
        tensor (torch.Tensor): The input PyTorch tensor.
        verbose (bool, optional): Whether to print debug messages. Defaults to False.

    Returns:
        np.ndarray: The converted NumPy array (deep copy).
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input must be a torch.Tensor, but got {type(tensor)}")

    if tensor.is_cuda:
        if verbose:
            print("Tensor is on GPU, moving to CPU...")
        tensor = tensor.cpu()

    if tensor.dtype == torch.bfloat16:
        if verbose:
            print("Tensor is in bfloat16, converting to float32 for NumPy compatibility.")
        tensor = tensor.to(dtype=torch.float32)

    numpy_array = tensor.detach().numpy().copy()
    if verbose:
        print(f"Converted tensor to NumPy array. Data type: {numpy_array.dtype} (deep copy)")
    return numpy_array


def numpy_to_tensor(array: np.ndarray, dtype: torch.dtype,  device: str = 'cpu' , verbose: bool = False) -> torch.Tensor:
    """
    Converts a NumPy array to a deep-copied PyTorch tensor.

    The array will be copied and optionally cast to the specified dtype.
    If dtype is bfloat16, ensure your hardware supports it.

    Args:
        array (np.ndarray): The input NumPy array.
        device (str, optional): The device to load the tensor onto. Defaults to 'cpu'.
        dtype (torch.dtype, optional): Desired PyTorch dtype (e.g. torch.float32, torch.bfloat16).
        verbose (bool, optional): Whether to print debug messages. Defaults to False.

    Returns:
        torch.Tensor: The converted PyTorch tensor (deep copy).
    """
    
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Input must be a np.ndarray, but got {type(array)}")

    device = device.lower()
    if not (device == 'cpu' or device.startswith('cuda')):
        raise ValueError(f"Invalid device specified: {device}. Must be 'cpu' or start with 'cuda'.")

    torch_tensor = torch.tensor(array.copy(), dtype=dtype if dtype else None)
    if verbose:
        print(f"Converted NumPy array to PyTorch tensor. Data type: {torch_tensor.dtype} (deep copy)")

    if dtype == torch.bfloat16:
        if verbose:
            print("Target dtype is bfloat16.")
        if not torch.cuda.is_available() and device.startswith('cuda'):
            if verbose:
                print("Warning: CUDA device requested for bfloat16, but not available. Tensor will remain on CPU.")
        elif torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bfloat16 is not supported on this CUDA hardware.")

    if device.startswith('cuda'):
        if torch.cuda.is_available():
            if verbose:
                print(f"Moving tensor to device: {device}")
            torch_tensor = torch_tensor.to(device)
        else:
            if verbose:
                print("Warning: CUDA device requested, but not available. Tensor will remain on CPU.")
    else:
        if verbose:
            print("Tensor is on CPU.")

    return torch_tensor

# # --- Example Usage ---
# if __name__ == '__main__':
#     print("--- Testing tensor_to_numpy ---")
#     # Create a tensor on CPU
#     cpu_tensor = torch.randn(3, 3, dtype=torch.float32)
#     print(f"Original CPU tensor dtype: {cpu_tensor.dtype}, device: {cpu_tensor.device}")
#     numpy_arr_from_cpu = tensor_to_numpy(cpu_tensor)
#     print(f"NumPy array dtype: {numpy_arr_from_cpu.dtype}")
#     print("NumPy array:\n", numpy_arr_from_cpu)
#     print("-" * 30)

#     if torch.cuda.is_available():
#         # Create a tensor on GPU with bfloat16 to test conversion
#         if torch.cuda.is_bf16_supported():
#             gpu_tensor_bf16 = torch.randn(2, 4, dtype=torch.bfloat16).cuda()
#             print(f"Original GPU bfloat16 tensor dtype: {gpu_tensor_bf16.dtype}, device: {gpu_tensor_bf16.device}")
#             numpy_arr_from_gpu_bf16 = tensor_to_numpy(gpu_tensor_bf16)
#             print(f"NumPy array dtype: {numpy_arr_from_gpu_bf16.dtype}")
#             print("NumPy array:\n", numpy_arr_from_gpu_bf16)
#         else:
#             print("bfloat16 not supported on this GPU, skipping bfloat16 tensor to NumPy test.")

#         gpu_tensor_int64 = torch.randn(2, 4, dtype=torch.int64).cuda()
#         print(f"Original GPU int64 tensor dtype: {gpu_tensor_int64.dtype}, device: {gpu_tensor_int64.device}")
#         numpy_arr_from_gpu_int64 = tensor_to_numpy(gpu_tensor_int64)
#         print(f"NumPy array dtype: {numpy_arr_from_gpu_int64.dtype}")
#         print("NumPy array:\n", numpy_arr_from_gpu_int64)
#     else:
#         print("CUDA is not available, skipping GPU tensor to NumPy test.")
#     print("-" * 30)

#     print("\n--- Testing numpy_to_tensor ---")
#     # Create a NumPy array
#     numpy_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
#     print(f"Original NumPy array dtype: {numpy_arr.dtype}")
#     cpu_tensor_from_numpy = numpy_to_tensor(numpy_arr, device='cpu')
#     print(f"PyTorch tensor dtype: {cpu_tensor_from_numpy.dtype}, device: {cpu_tensor_from_numpy.device}")
#     print("PyTorch tensor:\n", cpu_tensor_from_numpy)
#     print("-" * 30)

#     if torch.cuda.is_available():
#         gpu_tensor_from_numpy = numpy_to_tensor(numpy_arr, device='cuda')
#         print(f"PyTorch tensor dtype: {gpu_tensor_from_numpy.dtype}, device: {gpu_tensor_from_numpy.device}")
#         print("PyTorch tensor:\n", gpu_tensor_from_numpy)

#         # Test bfloat16 conversion if supported
#         if torch.cuda.is_bf16_supported():
#             print("\n--- Testing NumPy to bfloat16 Tensor on GPU ---")
#             numpy_arr_float32 = np.array([[7.0, 8.0], [9.0, 10.0]], dtype=np.float32)
#             gpu_tensor_bf16_from_numpy = numpy_to_tensor(numpy_arr_float32, device='cuda', dtype=torch.bfloat16)
#             print(f"PyTorch tensor dtype: {gpu_tensor_bf16_from_numpy.dtype}, device: {gpu_tensor_bf16_from_numpy.device}")
#             print("PyTorch tensor:\n", gpu_tensor_bf16_from_numpy)
#         else:
#             print("bfloat16 not supported on this GPU, skipping NumPy to bfloat16 tensor test.")
#     else:
#         print("CUDA is not available, skipping NumPy to GPU tensor tests.")
#     print("-" * 30)

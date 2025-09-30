import torch
import math
import torch.nn.functional as F

# Define the dimensions and block sizes as per the user's request.
# Q (query) has a sequence length of 1 for the decode stage.
# K/V cache has a sequence length of 512.
# We process the cache in blocks of 64.
# DQ is the dimension of the query.
# DK, DV are the dimensions of key and value.
# N_HEADS_Q is the number of heads for query.
# N_HEADS_KV is the number of heads for key/value (for grouped-query attention).

B, H_Q, H_KV, SEQ_LEN_Q, SEQ_LEN_KV, DQ, DK, DV = 1, 8, 1, 1, 1024, 4096, 512, 512
D_HEAD_Q = DQ // H_Q
D_HEAD_KV = DK // H_KV
BLOCK_SIZE = 64


def flash_attention_decode(q, k_cache, v_cache):
    """
    Implements Flash Attention for the decode stage for a group of query heads
    against a single (or broadcastable) key/value head.

    Args:
        q (torch.Tensor): The query tensor of shape (B, H_Q_GROUP, SEQ_LEN_Q, D_HEAD_Q).
        k_cache (torch.Tensor): The key cache tensor of shape (B, 1, SEQ_LEN_KV, D_HEAD_KV).
        v_cache (torch.Tensor): The value cache tensor of shape (B, 1, SEQ_LEN_KV, D_HEAD_KV).

    Returns:
        torch.Tensor: The output tensor of shape (B, H_Q_GROUP, SEQ_LEN_Q, D_HEAD_KV).
    """
    # Ensure SEQ_LEN_Q is 1 for decoding
    if q.shape[2] != 1:
        raise ValueError("Query sequence length must be 1 for decode stage.")

    # This version assumes Q heads and K/V heads are compatible for broadcasting,
    # removing the explicit repeat for GQA. The looping is handled by the caller.

    # Initialize output, running max, and sum of exponents for softmax
    # o: output tensor
    # m_i: row-wise max of the scores (for stable softmax)
    # l_i: sum of exp(scores - max) for each row
    o = torch.zeros((q.shape[0], q.shape[1], q.shape[2], D_HEAD_KV), device=q.device, dtype=q.dtype)
    m_i = torch.full((q.shape[0], q.shape[1], SEQ_LEN_Q, 1), -float('inf'), device=q.device, dtype=q.dtype)
    l_i = torch.zeros((q.shape[0], q.shape[1], SEQ_LEN_Q, 1), device=q.device, dtype=q.dtype)
    
    # Scale factor for dot-product attention
    scale = 1.0 / math.sqrt(D_HEAD_Q)
    q = q * scale

    # Iterate over the key-value cache in blocks
    num_blocks = k_cache.shape[2] // BLOCK_SIZE
    for j in range(num_blocks):
        # --- Step 1: Load a block of K and V from the cache ---
        start_idx = j * BLOCK_SIZE
        end_idx = (j + 1) * BLOCK_SIZE
        k_j = k_cache[:, :, start_idx:end_idx, :]
        v_j = v_cache[:, :, start_idx:end_idx, :]

        # --- Step 2: Compute attention scores for the block ---
        # S_ij = Q @ K_j^T
        # Broadcasting will happen on the head dimension if k_j has 1 head
        s_ij = torch.matmul(q, k_j.transpose(-2, -1))

        # --- Step 3: Update softmax statistics (m_i and l_i) ---
        # Find the new max score for the row
        m_ij = torch.max(s_ij, dim=-1, keepdim=True)[0]
        m_i_new = torch.maximum(m_i, m_ij)

        # Calculate softmax weights for the current block (P_ij)
        # using the updated max for numerical stability.
        p_ij = torch.exp(s_ij - m_i_new)

        # Rescale the running sum of exponents (l_i) with the old max
        l_i_rescaled = torch.exp(m_i - m_i_new) * l_i
        
        # Calculate the sum of exponents for the current block
        l_ij = torch.sum(p_ij, dim=-1, keepdim=True)
        
        # Update the total sum of exponents
        l_i_new = l_i_rescaled + l_ij

        # --- Step 4: Update the output ---
        # The previous 'o' is normalized by the old 'l_i'. We un-normalize it,
        # rescale it with the change in max, add the new block's contribution,
        # and then re-normalize by the new denominator 'l_i_new'.
        
        # Un-normalize and rescale the old output 'o'
        o_unnormalized_rescaled = o * l_i * torch.exp(m_i - m_i_new)
        
        # Calculate the output for the current block (un-normalized)
        o_j = torch.matmul(p_ij, v_j)
        
        # Add the new block's output and re-normalize with the new denominator
        o = (o_unnormalized_rescaled + o_j) / (l_i_new + 1e-9)

        # Update the running statistics for the next iteration
        m_i = m_i_new
        l_i = l_i_new
        
    return o


if __name__ == '__main__':
    # Set a seed for reproducibility
    torch.manual_seed(42)

    # Create dummy tensors with the specified dimensions
    # Using float16 as it's common for inference
    dtype = torch.float16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    q = torch.randn(B, H_Q, SEQ_LEN_Q, D_HEAD_Q, device=device, dtype=dtype)
    k_cache = torch.randn(B, H_KV, SEQ_LEN_KV, D_HEAD_KV, device=device, dtype=dtype)
    v_cache = torch.randn(B, H_KV, SEQ_LEN_KV, D_HEAD_KV, device=device, dtype=dtype)

    print("--- Input Shapes ---")
    print(f"Query (Q):      {q.shape}")
    print(f"Key Cache (K):    {k_cache.shape}")
    print(f"Value Cache (V):  {v_cache.shape}")
    print("-" * 20)

    # --- Call flash_attention_decode multiple times for GQA ---
    # This simulates processing each K/V head group separately.
    
    # Calculate how many Q heads map to one K/V head
    q_heads_per_kv_group = H_Q // H_KV
    
    output_parts = []
    # Iterate over each K/V head
    for i in range(H_KV):
        # Get the i-th head from K and V caches
        k_head = k_cache[:, i:i+1, :, :]
        v_head = v_cache[:, i:i+1, :, :]
        
        # Get the corresponding group of Q heads
        q_start_index = i * q_heads_per_kv_group
        q_end_index = (i + 1) * q_heads_per_kv_group
        q_group = q[:, q_start_index:q_end_index, :, :]
        
        # Call the attention function for the current group
        output_part = flash_attention_decode(q_group, k_head, v_head)
        output_parts.append(output_part)
        
    # Concatenate the results from all groups along the head dimension
    output = torch.cat(output_parts, dim=1)


    print("--- Output Shape ---")
    print(f"Output (O):     {output.shape}")
    print("-" * 20)

    # For verification, let's compare with a standard attention implementation
    # This will consume more memory but is good for checking correctness.
    if H_Q != H_KV:
        num_repeats = H_Q // H_KV
        k_full = k_cache.repeat_interleave(num_repeats, dim=1)
        v_full = v_cache.repeat_interleave(num_repeats, dim=1)
    else:
        k_full = k_cache
        v_full = v_cache
        
    scale = 1.0 / math.sqrt(D_HEAD_Q)
    attn_scores = torch.matmul(q * scale, k_full.transpose(-2, -1))
    attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(dtype)
    expected_output = torch.matmul(attn_weights, v_full)

    print("--- Verification ---")
    
    # Flatten tensors for metric calculations
    output_flat = output.flatten()
    expected_flat = expected_output.flatten()

    # Cosine Similarity
    cosine_sim = F.cosine_similarity(output_flat, expected_flat, dim=0)
    print(f"Cosine Similarity: {cosine_sim.item():.6f}")

    # L1 (Mean Absolute Error)
    l1_error = torch.mean(torch.abs(output - expected_output))
    print(f"L1 (Mean Absolute Error): {l1_error.item():.6f}")

    # L2 (Mean Squared Error)
    l2_error = torch.mean((output - expected_output)**2)
    print(f"L2 (Mean Squared Error): {l2_error.item():.6f}")

    # RMSE (Root Mean Squared Error)
    rmse_error = torch.sqrt(l2_error)
    print(f"RMSE (Root Mean Squared Error): {rmse_error.item():.6f}")

    # A high cosine similarity and low errors indicate correctness.
    assert cosine_sim > 0.999, "Cosine similarity is too low!"
    assert rmse_error < 1e-2, "RMSE is too high!"
    print("\nVerification successful: Metrics are within acceptable thresholds.")


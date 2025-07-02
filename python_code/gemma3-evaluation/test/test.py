import numpy as np
import numpy.typing as npt

def flash_attention_np_single_head_single_batch(Q: npt.NDArray, K: npt.NDArray, V: npt.NDArray, 
                                               BR: int, BC: int, attention_mask: None|npt.NDArray,
                                               scaling: float = None):
    """
    Improved Flash Attention implementation with better numerical stability.
    
    Args:
        Q, K, V: [seq_len, d_size] attention matrices
        BR, BC: block sizes for rows and columns
        attention_mask: [seq_len, seq_len] mask with 0 or -inf values
        scaling: attention scaling factor (default: 1/sqrt(d_size))
    """
    # Input validation
    seq_len = Q.shape[0]
    d_size = Q.shape[1]
    assert K.shape[0] == seq_len and V.shape[0] == seq_len
    assert K.shape[1] == d_size and V.shape[1] == d_size
    
    if scaling is None:
        scaling = 1.0 / np.sqrt(d_size)
    
    # Use higher precision for better numerical stability
    dtype = np.float64
    
    # Initialize output matrices
    O = np.zeros_like(Q, dtype=dtype)
    L = np.zeros((seq_len, 1), dtype=dtype)  # Row sums (denominators)
    M = np.full((seq_len, 1), -np.inf, dtype=dtype)  # Row max values
    
    # Calculate number of blocks
    num_i_blocks = int((seq_len + BR - 1) // BR)
    num_j_blocks = int((seq_len + BC - 1) // BC)
    
    # Numerical stability constants
    eps = 1e-12
    max_exp = 50.0  # Prevent overflow in exp
    
    # Main computation loop
    for j in range(num_j_blocks):
        j_start = j * BC
        j_end = min((j + 1) * BC, seq_len)
        
        # Load K_j, V_j blocks
        K_j = K[j_start:j_end, :].astype(dtype)  # [BC, d_size]
        V_j = V[j_start:j_end, :].astype(dtype)  # [BC, d_size]
        
        for i in range(num_i_blocks):
            i_start = i * BR
            i_end = min((i + 1) * BR, seq_len)
            
            # Load Q_i block and current state
            Q_i = Q[i_start:i_end, :].astype(dtype)  # [BR, d_size]
            O_i = O[i_start:i_end, :].copy()  # [BR, d_size]
            l_i = L[i_start:i_end, :]  # [BR, 1]
            m_i = M[i_start:i_end, :]  # [BR, 1]
            
            # Compute attention scores
            s_i_j = np.matmul(Q_i, K_j.T) * scaling  # [BR, BC]
            
            # Apply attention mask if provided
            if attention_mask is not None:
                mask_block = attention_mask[i_start:i_end, j_start:j_end]
                s_i_j = s_i_j + mask_block
            
            # Compute block-wise maximum (with clipping for stability)
            m_i_j = np.max(s_i_j, axis=1, keepdims=True)  # [BR, 1]
            m_i_j = np.clip(m_i_j, -max_exp, max_exp)
            
            # Compute softmax numerator
            s_i_j_stable = s_i_j - m_i_j  # Broadcast subtraction
            s_i_j_stable = np.clip(s_i_j_stable, -max_exp, max_exp)
            p_i_j = np.exp(s_i_j_stable)  # [BR, BC]
            
            # Compute block-wise sum
            l_i_j = np.sum(p_i_j, axis=1, keepdims=True)  # [BR, 1]
            l_i_j = np.maximum(l_i_j, eps)  # Prevent division by zero
            
            # Update global statistics
            m_i_new = np.maximum(m_i, m_i_j)  # [BR, 1]
            
            # Compute exponential differences (clipped for stability)
            exp_diff_old = np.exp(np.clip(m_i - m_i_new, -max_exp, max_exp))  # [BR, 1]
            exp_diff_new = np.exp(np.clip(m_i_j - m_i_new, -max_exp, max_exp))  # [BR, 1]
            
            # Update row sums
            l_i_new = exp_diff_old * l_i + exp_diff_new * l_i_j  # [BR, 1]
            l_i_new = np.maximum(l_i_new, eps)  # Ensure non-zero
            
            # Compute weighted contributions
            if np.any(l_i > eps):  # Only if we have previous contributions
                # Weight for existing output
                w_old = (exp_diff_old * l_i) / l_i_new  # [BR, 1]
                # Weight for new contribution  
                w_new = (exp_diff_new * l_i_j) / l_i_new  # [BR, 1]
                
                # Update output with weighted average
                O_new = w_old * O_i + w_new * np.matmul(p_i_j, V_j)
            else:
                # First contribution to this block
                O_new = np.matmul(p_i_j, V_j) / l_i_j
                l_i_new = l_i_j
            
            # Store updates
            O[i_start:i_end, :] = O_new
            L[i_start:i_end, :] = l_i_new  
            M[i_start:i_end, :] = m_i_new
    
    # Convert back to original dtype
    if Q.dtype != dtype:
        O = O.astype(Q.dtype)
    
    return O


def flash_attention_reference(Q: npt.NDArray, K: npt.NDArray, V: npt.NDArray, 
                            attention_mask: None|npt.NDArray = None,
                            scaling: float = None):
    """
    Reference implementation of standard attention for comparison.
    """
    if scaling is None:
        scaling = 1.0 / np.sqrt(Q.shape[1])
    
    # Compute attention scores
    scores = np.matmul(Q, K.T) * scaling
    
    # Apply mask
    if attention_mask is not None:
        scores = scores + attention_mask
    
    # Apply softmax
    scores_max = np.max(scores, axis=1, keepdims=True)
    scores_exp = np.exp(scores - scores_max)
    attention_weights = scores_exp / np.sum(scores_exp, axis=1, keepdims=True)
    
    # Compute output
    output = np.matmul(attention_weights, V)
    
    return output


# Test function to compare implementations
def test_flash_attention():
    """Test the Flash Attention implementation against reference."""
    
    # Test parameters
    seq_len = 128
    d_size = 64
    
    # Generate random inputs
    np.random.seed(42)
    Q = np.random.randn(seq_len, d_size).astype(np.float32)
    K = np.random.randn(seq_len, d_size).astype(np.float32)  
    V = np.random.randn(seq_len, d_size).astype(np.float32)
    
    # Create causal mask
    mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
    
    scaling = 1.0 / np.sqrt(d_size)
    
    # Reference implementation
    ref_output = flash_attention_reference(Q, K, V, mask, scaling)
    
    # Test different block sizes
    block_sizes = [seq_len, seq_len//2, seq_len//4, 32, 16]
    
    print("Testing Flash Attention with different block sizes:")
    print(f"Sequence length: {seq_len}, d_size: {d_size}")
    print("-" * 60)
    
    for br in block_sizes:
        bc = br  # Use same block size for both dimensions
        if br > seq_len:
            continue
            
        flash_output = flash_attention_np_single_head_single_batch(
            Q, K, V, br, bc, mask, scaling
        )
        
        # Compute error metrics
        abs_error = np.abs(flash_output - ref_output)
        max_error = np.max(abs_error)
        mean_error = np.mean(abs_error)
        rel_error = max_error / (np.max(np.abs(ref_output)) + 1e-8)
        
        print(f"BR=BC={br:3d}: Max Error={max_error:.2e}, Mean Error={mean_error:.2e}, Rel Error={rel_error:.2e}")
    
    return ref_output, flash_output


if __name__ == "__main__":
    ref_out, flash_out = test_flash_attention()
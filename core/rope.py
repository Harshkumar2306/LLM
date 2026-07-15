import torch

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (RoPE).
    Args:
        dim: Dimension of the attention head
        end: Maximum sequence length to precompute for
        theta: The base scale for the frequencies
    Returns:
        freqs_cis: Complex tensor of shape (end, dim // 2)
    """
    # 1. Calculate the frequencies: theta^(-2i/dim)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # 2. Calculate the positions: [0, 1, 2, ..., end - 1]
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    
    # 3. Outer product: (end, dim // 2)
    freqs = torch.outer(t, freqs)
    
    # 4. Convert to complex numbers: cos(freqs) + i * sin(freqs)
    # Using torch.polar(abs, angle), abs=1 for pure rotations
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    Apply rotary embeddings to queries and keys.
    Args:
        xq: Query tensor of shape (Batch, Seq_Len, N_Heads, Head_Dim)
        xk: Key tensor of shape (Batch, Seq_Len, N_Heads, Head_Dim)
        freqs_cis: Complex frequencies of shape (Seq_Len, Head_Dim // 2)
    Returns:
        xq_out, xk_out: Rotated query and key tensors of the same shape as inputs.
    """
    # 1. Reshape inputs as complex numbers
    # (Batch, Seq_Len, N_Heads, Head_Dim) -> (Batch, Seq_Len, N_Heads, Head_Dim // 2, 2)
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    # Turn into complex tensors: (Batch, Seq_Len, N_Heads, Head_Dim // 2)
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)
    
    # 2. Reshape freqs_cis for broadcasting
    # (Seq_Len, Head_Dim // 2) -> (1, Seq_Len, 1, Head_Dim // 2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    
    # 3. Apply rotation (complex multiplication)
    xq_out = xq_ * freqs_cis
    xk_out = xk_ * freqs_cis
    
    # 4. Convert back to real and flatten
    # (Batch, Seq_Len, N_Heads, Head_Dim // 2) -> (Batch, Seq_Len, N_Heads, Head_Dim // 2, 2) -> (Batch, Seq_Len, N_Heads, Head_Dim)
    xq_out = torch.view_as_real(xq_out).flatten(3)
    xk_out = torch.view_as_real(xk_out).flatten(3)
    
    # Return as original dtype
    return xq_out.type_as(xq), xk_out.type_as(xk)

import torch
from core.rope import precompute_freqs_cis, apply_rotary_emb

def test_precompute_freqs_cis():
    dim = 64
    end = 128
    freqs_cis = precompute_freqs_cis(dim, end)
    
    assert freqs_cis.shape == (end, dim // 2)
    assert freqs_cis.is_complex()
    
    # Absolute value of complex rotations should be 1.0 (pure rotation)
    assert torch.allclose(torch.abs(freqs_cis), torch.ones_like(torch.abs(freqs_cis)))

def test_apply_rotary_emb():
    batch, seq_len, n_heads, head_dim = 2, 64, 4, 32
    q = torch.randn(batch, seq_len, n_heads, head_dim)
    k = torch.randn(batch, seq_len, n_heads, head_dim)
    
    freqs_cis = precompute_freqs_cis(head_dim, seq_len * 2)
    
    # Slice freqs_cis to current seq_len
    freqs_cis = freqs_cis[:seq_len]
    
    q_out, k_out = apply_rotary_emb(q, k, freqs_cis)
    
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    
    # RoPE preserves the L2 norm of the vectors since it's a pure rotation
    assert torch.allclose(torch.norm(q, dim=-1), torch.norm(q_out, dim=-1), atol=1e-5)
    assert torch.allclose(torch.norm(k, dim=-1), torch.norm(k_out, dim=-1), atol=1e-5)

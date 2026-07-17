import torch
import math
import sys

def apply_rotary_emb_complex(xq, xk, freqs_cis):
    # As implemented currently
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    xq_out = xq_ * freqs_cis
    xk_out = xk_ * freqs_cis
    xq_out = torch.view_as_real(xq_out).flatten(3)
    xk_out = torch.view_as_real(xk_out).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def apply_rotary_emb_real(xq, xk, freqs_cos, freqs_sin):
    # Using real operations (Llama style without complex casting)
    # xq: (B, T, NH, HD)
    xq_r, xq_i = xq.float().reshape(*xq.shape[:-1], -1, 2).unbind(-1)
    xk_r, xk_i = xk.float().reshape(*xk.shape[:-1], -1, 2).unbind(-1)
    
    freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(2)
    freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(2)
    
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos
    
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def run_profile(device):
    B, T, NH, HD = 16, 256, 8, 32  # 256 / 8 = 32
    print(f"Profiling on {device}...")
    xq = torch.randn(B, T, NH, HD, device=device, requires_grad=True)
    xk = torch.randn(B, T, NH, HD, device=device, requires_grad=True)
    
    # Precompute freqs
    freqs = torch.exp(-math.log(10000.0) / HD * torch.arange(0, HD, 2).float())
    t = torch.arange(T, device=device).float()
    freqs = torch.outer(t, freqs.to(device))
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    freqs_cos = freqs.cos()
    freqs_sin = freqs.sin()
    
    if device == 'mps':
        torch.mps.empty_cache()
        mem_func = torch.mps.current_allocated_memory
    elif device == 'cuda':
        torch.cuda.empty_cache()
        mem_func = torch.cuda.memory_allocated
    else:
        mem_func = lambda: 0
        
    m0 = mem_func()
    out_q, out_k = apply_rotary_emb_complex(xq, xk, freqs_cis)
    loss = out_q.sum() + out_k.sum()
    loss.backward()
    m1 = mem_func()
    if mem_func != (lambda: 0):
        print(f"Complex Implementation Peak Memory Delta: {(m1 - m0) / 1024**2:.2f} MB")
    
    if device in ['mps', 'cuda']:
        torch.mps.empty_cache() if device == 'mps' else torch.cuda.empty_cache()
        
    xq.grad = None
    xk.grad = None
    
    m0 = mem_func()
    out_q, out_k = apply_rotary_emb_real(xq, xk, freqs_cos, freqs_sin)
    loss = out_q.sum() + out_k.sum()
    loss.backward()
    m1 = mem_func()
    if mem_func != (lambda: 0):
        print(f"Real Implementation Peak Memory Delta: {(m1 - m0) / 1024**2:.2f} MB")

run_profile('cpu')
run_profile('mps')

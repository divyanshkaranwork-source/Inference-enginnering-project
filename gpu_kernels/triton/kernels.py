"""Triton kernels for LLM inference (requires an NVIDIA/AMD GPU, torch and triton).

    pip install torch triton
    python gpu_kernels/triton/kernels.py        # runs self-checks vs PyTorch

Triton lets you write tile-level GPU code in Python: you describe what one
*program* (thread block) does on a BLOCK of data and the compiler handles
threads, shared memory and vectorisation. vLLM, SGLang and Unsloth ship many
Triton kernels; Liger-Kernel is a good library to read next.

Kernels: fused softmax, RMSNorm, causal FlashAttention forward.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_stride, out_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(in_ptr + row * in_stride + offs, mask=mask, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)  # whole row lives in registers: one HBM read, one write
    num = tl.exp(x)
    tl.store(out_ptr + row * out_stride + offs, num / tl.sum(num, axis=0), mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    out = torch.empty_like(x)
    softmax_kernel[(rows,)](out, x, x.stride(0), out.stride(0), cols, BLOCK=triton.next_power_of_2(cols))
    return out


# ---------------------------------------------------------------------------
@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, stride, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * stride + offs, mask=mask, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * stride + offs, (x * rstd * w).to(y_ptr.dtype.element_ty), mask=mask)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    rows, cols = x.shape
    y = torch.empty_like(x)
    rmsnorm_kernel[(rows,)](x, w, y, x.stride(0), cols, eps, BLOCK=triton.next_power_of_2(cols))
    return y


# ---------------------------------------------------------------------------
@triton.jit
def flash_attn_fwd_kernel(
    Q, K, V, O, sm_scale, T,
    stride_h, stride_t,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    """One program = BLOCK_M query rows of one head. Tensors are [H, T, D] contiguous."""
    start_m = tl.program_id(0)
    h = tl.program_id(1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    base = h * stride_h
    q = tl.load(Q + base + offs_m[:, None] * stride_t + offs_d[None, :], mask=offs_m[:, None] < T, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    hi = tl.minimum((start_m + 1) * BLOCK_M, T)  # causal: skip tiles above the diagonal
    for start_n in range(0, hi, BLOCK_N):
        n = start_n + offs_n
        kv_mask = n[:, None] < T
        k = tl.load(K + base + n[:, None] * stride_t + offs_d[None, :], mask=kv_mask, other=0.0)
        v = tl.load(V + base + n[:, None] * stride_t + offs_d[None, :], mask=kv_mask, other=0.0)
        s = tl.dot(q, tl.trans(k)) * sm_scale  # tensor cores
        s = tl.where((offs_m[:, None] >= n[None, :]) & (n[None, :] < T), s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(O + base + offs_m[:, None] * stride_t + offs_d[None, :], acc.to(O.dtype.element_ty),
             mask=offs_m[:, None] < T)


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Causal attention. q, k, v: [H, T, D] (D in 16/32/64/128)."""
    H, T, D = q.shape
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    o = torch.empty_like(q)
    BLOCK = 64
    grid = (triton.cdiv(T, BLOCK), H)
    flash_attn_fwd_kernel[grid](q, k, v, o, 1.0 / math.sqrt(D), T, q.stride(0), q.stride(1),
                                BLOCK_M=BLOCK, BLOCK_N=BLOCK, HEAD_DIM=D)
    return o


# ---------------------------------------------------------------------------
def _self_test() -> None:
    torch.manual_seed(0)
    dev = "cuda"
    x = torch.randn(64, 32000, device=dev)
    torch.testing.assert_close(softmax(x), torch.softmax(x, dim=-1), atol=1e-5, rtol=1e-4)
    w = torch.randn(4096, device=dev)
    xs = torch.randn(16, 4096, device=dev)
    ref = xs * torch.rsqrt(xs.pow(2).mean(-1, keepdim=True) + 1e-5) * w
    torch.testing.assert_close(rmsnorm(xs, w), ref, atol=1e-4, rtol=1e-4)
    H, T, D = 8, 1000, 64
    q, k, v = (torch.randn(H, T, D, device=dev, dtype=torch.float16) for _ in range(3))
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(flash_attention(q, k, v), ref, atol=2e-2, rtol=2e-2)
    print("all Triton kernels match PyTorch")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA (or ROCm) GPU")
    _self_test()

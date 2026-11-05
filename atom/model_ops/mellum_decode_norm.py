"""Parallel rows with PyTorch ROCm's single-row FP32 mean reduction order.

Reduction order follows installed ATen/native/cuda/Reduce.cuh:
four vector accumulators; inter-warp offsets 256,128,64; intra-warp
shuffle-down offsets 1,2,4,8,16,32. Decode only, width 2304.
"""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.hip import libdevice


@tr.jit
def _norm(X, W, Y, V, STRIDE: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0)
    idx = tl.arange(0, 512)
    a = tl.full((512,), 0, tl.float32)
    b = tl.full((512,), 0, tl.float32)
    c = tl.full((512,), 0, tl.float32)
    d = tl.full((512,), 0, tl.float32)
    for step in tl.static_range(2):
        col = (idx + step * 512) * 4
        x0 = tl.load(X + row * STRIDE + col, col < 2304, 0).to(tl.float32)
        x1 = tl.load(X + row * STRIDE + col + 1, col + 1 < 2304, 0).to(tl.float32)
        x2 = tl.load(X + row * STRIDE + col + 2, col + 2 < 2304, 0).to(tl.float32)
        x3 = tl.load(X + row * STRIDE + col + 3, col + 3 < 2304, 0).to(tl.float32)
        a = a + x0 * x0
        b = b + x1 * x1
        c = c + x2 * x2
        d = d + x3 * x3
    value = ((a + b) + c) + d
    for offset in tl.static_range(3):
        distance = 256 >> offset
        other = tl.gather(value, (idx + distance) % 512, axis=0)
        value = tl.where(idx < distance, value + other, value)
    for shift in tl.static_range(6):
        distance = 1 << shift
        source = idx // 64 * 64 + tl.minimum(idx % 64 + distance, 63)
        other = tl.gather(value, source, axis=0)
        value = value + other
    total = tl.sum(tl.where(idx == 0, value, 0), axis=0)
    variance = total * (1.0 / 2304.0)
    tl.store(V + row, variance)
    inverse = libdevice.rsqrt((variance + EPS).to(tl.float64)).to(tl.float32)
    cols = tl.arange(0, 4096)
    x = tl.load(X + row * STRIDE + cols, cols < 2304, 0).to(tl.float32)
    weight = tl.load(W + cols, cols < 2304, 0).to(tl.float32)
    normalized = (x * inverse).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * 2304 + cols, weight * normalized, cols < 2304)


def rmsnorm(x, weight, eps, *, return_variance=False):
    assert x.ndim == 2 and x.shape[1] == 2304 and x.stride(1) == 1
    assert x.dtype == weight.dtype == torch.bfloat16
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    variance = torch.empty(x.shape[0],device=x.device,dtype=torch.float32)
    _norm[(x.shape[0],)](x,weight,out,variance,x.stride(0),eps,num_warps=8,enable_fp_fusion=False)
    return (out,variance) if return_variance else out

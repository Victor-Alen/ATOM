"""Request metadata adapter for the original AITER fused HIP RMSNorm kernel."""

import aiter
import torch

from atom.model_ops.mellum_request_gemm import ranges
from atom.utils.forward_context import get_forward_context


def rmsnorm(x, weight, eps, residual=None):
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    factor = flat.shape[0] // shape[0]
    parts = ranges(x)
    ctx = get_forward_context()
    cache = getattr(ctx, "_mellum_native_norm_offsets", None)
    if cache is None:
        cache = {}
        ctx._mellum_native_norm_offsets = cache
    key = (tuple(parts), factor, x.device)
    entry = cache.get(key)
    if entry is None:
        offsets = [parts[0][0] * factor] + [stop * factor for _, stop in parts]
        maximum = max(stop - start for start, stop in parts) * factor
        entry = (torch.tensor(offsets, device=x.device, dtype=torch.int32), maximum)
        cache[key] = entry
    offsets, maximum = entry
    out = torch.empty_like(flat)
    if residual is None:
        aiter.rmsnorm(
            out, flat, weight, eps, hf_row_offsets=offsets, hf_max_rows=maximum
        )
        return out.reshape(shape)
    residual_flat = residual.reshape_as(flat)
    residual_out = torch.empty_like(flat)
    aiter.add_rmsnorm(
        out,
        flat,
        residual_flat,
        residual_out,
        weight,
        eps,
        hf_row_offsets=offsets,
        hf_max_rows=maximum,
    )
    return out.reshape(shape), residual_out.reshape(shape)

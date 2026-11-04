"""Request-shaped GEMMs for Mellum parity under continuous batching."""
import itertools

import torch

from atom.utils.forward_context import get_forward_context


def cpu_metadata(metadata, *, with_lengths=False):
    """Copy request boundaries once per forward, shared by all 28 layers.

    ForwardContext is recreated by set_forward_context for every model step.
    Keep both the metadata and tensor references here so neither recycled
    object identities nor another layer's metadata can reuse stale offsets.
    """
    ctx = get_forward_context()
    enabled = __import__("os").environ.get("ATOM_MELLUM_CACHE_METADATA", "1") == "1"
    cache = getattr(ctx, "_mellum_cpu_metadata", None)
    if cache is None:
        cache = {}
        ctx._mellum_cpu_metadata = cache
    key = id(metadata)
    item = cache.get(key) if enabled else None
    cu = metadata.cu_seqlens_q
    lengths = metadata.context_lens
    if (
        item is None
        or item[0] is not metadata
        or item[1] is not cu
        or item[2] is not lengths
    ):
        offsets = (
            tuple(cu.tolist()) if cu is not None else tuple(range(len(lengths) + 1))
        )
        item = [metadata, cu, lengths, offsets, None]
        if enabled:
            cache[key] = item
    if with_lengths and item[4] is None:
        item[4] = tuple(lengths.tolist())
    return item[3], item[4]


def ranges(x):
    meta = get_forward_context().attn_metadata
    if isinstance(meta, dict):
        if not meta:
            return [(0, x.shape[0])]
        meta = next(iter(meta.values()))
    if meta is None or meta.context_lens is None:
        return [(0, x.shape[0])]
    offsets, _ = cpu_metadata(meta)
    assert offsets[-1] == x.shape[0], (offsets, x.shape)
    return list(itertools.pairwise(offsets))


def linear(x, weight, bias=None):
    parts = ranges(x)
    from atom.model_ops.mellum_piecewise import request_region

    return request_region(
        x, weight, bias, parts, not get_forward_context().context.is_prefill
    )


def rmsnorm(x, weight, eps):
    if (
        __import__("os").environ.get("ATOM_MELLUM_BATCH_DECODE_QKNORM", "0") == "1"
        and not get_forward_context().context.is_prefill
        and x.shape[-1] == 128
        and x.shape[0] <= 64
    ):
        from atom.model_ops.mellum_piecewise import norm_tensor

        return norm_tensor(x, weight, eps)

    def one(values):
        floats = values.float()
        variance = floats.pow(2).mean(-1, keepdim=True)
        normalized = (floats * torch.rsqrt(variance + eps)).to(values.dtype)
        return weight * normalized

    parts = ranges(x)
    if not get_forward_context().context.is_prefill:
        from atom.model_ops.mellum_piecewise import request_region

        return request_region(x, weight, None, parts, True, norm_eps=eps)
    if len(parts) == 1:
        return one(x)
    return torch.cat([one(x[a:b]) for a, b in parts], dim=0)

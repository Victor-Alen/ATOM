"""Opt-in HF-order native paged decode with one logical sequence per query.

Speculative target verification must crop each query's KV extent exactly as
sequential decode does. Expand paged metadata, never the KV cache itself.
"""
import os
import torch
from atom.model_ops.mellum_request_gemm import cpu_metadata
from atom.utils.forward_context import get_forward_context


def _legacy_metadata(q, fwd_ctx):
    meta = fwd_ctx.attn_metadata
    offsets, lengths = cpu_metadata(meta, with_lengths=True)
    assert offsets[-1] == q.shape[0] and not fwd_ctx.context.is_prefill
    ctx = get_forward_context()
    cache = getattr(ctx, '_mellum_mtp_native_metadata', None)
    if cache is None:
        cache = {}
        ctx._mellum_mtp_native_metadata = cache
    key = (id(meta), offsets, lengths, q.device)
    entry = cache.get(key)
    if entry is None:
        owners, contexts = [], []
        for seq, length in enumerate(lengths):
            count = offsets[seq + 1] - offsets[seq]
            owners.extend([seq] * count)
            contexts.extend(range(length - count + 1, length + 1))
        assert len(owners) == q.shape[0] and min(contexts) > 0
        indices = torch.tensor(owners, dtype=torch.long, device=q.device)
        context_lens = torch.tensor(contexts, dtype=meta.context_lens.dtype, device=q.device)
        tables = meta.block_tables.index_select(0, indices)
        entry = (context_lens, tables, meta)
        cache[key] = entry
    return entry[0], entry[1]

# Static indexing tensors are warmed before capture; context lengths and block
# tables must be computed inside the captured graph on every replay.
_UNIFORM_INDICES = {}

def _uniform_metadata(q, fwd_ctx):
    meta = fwd_ctx.attn_metadata
    q_len = int(meta.max_seqlen_q)
    rows = q.shape[0]
    if q_len < 1 or rows != meta.context_lens.numel() * q_len:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("MTP graph metadata requires uniform decode query lengths")
        return _legacy_metadata(q, fwd_ctx)
    ctx = get_forward_context()
    capture = torch.cuda.is_current_stream_capturing()
    cache = getattr(ctx, '_mellum_mtp_uniform_metadata', None)
    if cache is None:
        cache = {}; ctx._mellum_mtp_uniform_metadata = cache
    key = (id(meta), rows, q_len, q.device, capture)
    entry = cache.get(key)
    if entry is None:
        index_key = (rows, q_len, q.device, meta.context_lens.dtype)
        indices = _UNIFORM_INDICES.get(index_key)
        if indices is None:
            if capture:
                raise RuntimeError("MTP graph indices must be warmed before capture")
            arange = torch.arange(rows, device=q.device, dtype=torch.long)
            owners = torch.div(arange, q_len, rounding_mode='floor')
            relative = (arange.remainder(q_len) - q_len + 1).to(meta.context_lens.dtype)
            indices = (owners, relative); _UNIFORM_INDICES[index_key] = indices
        owners, relative = indices
        # Padded graph rows have zero context; clamp the first speculative
        # query to zero rather than exposing a negative KV extent.
        lengths = (meta.context_lens.index_select(0, owners) + relative).clamp_min(0)
        tables = meta.block_tables.index_select(0, owners)
        entry = (lengths, tables, meta); cache[key] = entry
    return entry[0], entry[1]


def attention(layer, q, k, v, kc, vc, k_scale, v_scale, fwd_ctx):
    assert q.dtype == torch.bfloat16 and kc.dtype == vc.dtype == torch.bfloat16
    assert not fwd_ctx.context.is_prefill
    if os.environ.get('ATOM_MELLUM_MTP_GRAPH_METADATA') == '1':
        context_lens, tables = _uniform_metadata(q, fwd_ctx)
    else:
        context_lens, tables = _legacy_metadata(q, fwd_ctx)
    blocks, kvheads, _, block_size, _ = kc.shape
    if layer._cache_format == 'SHUFFLE':
        pack = 16 // vc.element_size()
        vc = vc.view(blocks, kvheads, block_size // pack, q.shape[-1], pack)
    else:
        assert layer._cache_format == 'NHD'
    heads = q.shape[1]
    assert heads % kvheads == 0
    shape = (q.shape[0], kvheads, 1, heads // kvheads)
    exp_sums = torch.empty(shape, device=q.device, dtype=torch.float32)
    max_logits = torch.empty_like(exp_sums)
    temporary = torch.empty((*shape, q.shape[-1]), device=q.device, dtype=q.dtype)
    out = torch.empty_like(q)
    # The existing AITER head-split knob also affects full attention, whose
    # GMEM path requires unsplit heads. Scope this experiment to window layers
    # while leaving full attention untouched. Only single-worker TP/DP=1 is
    # supported here; restore the process option immediately after dispatch.
    window_splits = os.environ.get('ATOM_MELLUM_MTP_WINDOW_HEAD_SPLITS', '1')
    if window_splits not in ('1', '2'):
        raise ValueError('MTP window head splits must be 1 or 2')
    old_splits = os.environ.get('AITER_MELLUM_HF_HEAD_SPLITS')
    change = layer.sliding_window > 0 and window_splits != '1'
    if change:
        os.environ['AITER_MELLUM_HF_HEAD_SPLITS'] = window_splits
    try:
        torch.ops.aiter.pa_decode_gluon(
            out, q, kc, vc, context_lens, tables, layer.scale, 1, 1, 128,
            torch.bfloat16, None, None, None, exp_sums=exp_sums,
            max_logits=max_logits, temporary_output=temporary, alibi_slopes=None,
            sinks=None, sliding_window=layer.sliding_window, ps=True,
        )
    finally:
        if change:
            if old_splits is None:
                os.environ.pop('AITER_MELLUM_HF_HEAD_SPLITS', None)
            else:
                os.environ['AITER_MELLUM_HF_HEAD_SPLITS'] = old_splits
    return out

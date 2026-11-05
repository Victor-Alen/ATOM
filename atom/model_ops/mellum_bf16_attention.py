"""HF-order BF16 attention over the production request and paged KV cache.

No model reference state or saved tensors. Uses BF16 score/PV GEMMs and FP32
softmax with the same materialization boundaries as native HF eager attention.
"""
import torch
import torch.nn.functional as F


def attention(q, k, v, kc, vc, metadata, scale, window, cache_format='NHD'):
    assert q.ndim == 3 and q.shape[-1] == 128
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    from atom.model_ops.mellum_request_gemm import cpu_metadata
    offsets, lengths = cpu_metadata(metadata, with_lengths=True)
    assert len(offsets) == len(lengths) + 1 and offsets[-1] == q.shape[0]
    if (__import__('os').environ.get('ATOM_MELLUM_PAGED_DECODE') == '1'
            and window == 1024
            and all(stop - start == 1 for start, stop in zip(offsets, offsets[1:]))
            and all(length >= 512 for length in lengths)):
        from atom.model_ops.mellum_paged_decode import attention as paged_decode
        return paged_decode(q, kc, vc, metadata.block_tables, metadata.context_lens, scale, cache_format, host_lengths=lengths)
    if (__import__('os').environ.get('ATOM_MELLUM_BATCH_ATTENTION') == '1'
            and all(stop - start == 1 for start, stop in zip(offsets, offsets[1:]))
            and all(length > 1 for length in lengths)):
        from atom.model_ops.mellum_batch_decode_attention import attention as batched
        return batched(q, kc, vc, metadata, lengths, scale, window, cache_format)
    heads, kvheads, dim = q.shape[1], k.shape[1], q.shape[2]
    assert heads % kvheads == 0
    repeat = heads // kvheads
    out = torch.empty_like(q)
    for b, context_length in enumerate(lengths):
        start, stop = offsets[b], offsets[b+1]
        count = stop-start
        if count == context_length:
            key, value = k[start:stop], v[start:stop]
        else:
            assert kc.ndim == 5 and vc.ndim == 4
            block_size = kc.shape[3]
            assert vc.shape[-1] == block_size
            # Read only blocks participating in this attention GEMM. The
            # original implementation gathered the complete history before
            # discarding the old sliding-window prefix.
            first_token = 0
            if window is not None and window > 0:
                first_token = max(0, context_length - (window - 1 + count))
            first_block, skip = divmod(first_token, block_size)
            block_stop = (context_length + block_size - 1) // block_size
            blocks = metadata.block_tables[b, first_block:block_stop].long()
            wanted = context_length - first_token
            key = kc.index_select(0,blocks).permute(0,3,1,2,4).reshape(-1,kvheads,dim)[skip:skip+wanted]
            cached_value=vc.index_select(0,blocks)
            if cache_format == 'SHUFFLE':
                pack=16//vc.element_size()
                value=cached_value.view(len(blocks),kvheads,block_size//pack,dim,pack).permute(0,2,4,1,3).reshape(-1,kvheads,dim)[skip:skip+wanted]
            else:
                assert cache_format == 'NHD',cache_format
                value=cached_value.permute(0,3,1,2).reshape(-1,kvheads,dim)[skip:skip+wanted]
        # HF DynamicSlidingWindowLayer retains window-1 past positions and
        # concatenates the current query block before returning attention KV.
        # Match that GEMM extent, rather than multiplying over masked old KV.
        if count < context_length and window is not None and window > 0:
            keep=min(context_length,window-1+count)
            key,value=key[-keep:],value[-keep:]
        key_length=key.shape[0]
        query = q[start:stop].transpose(0,1).unsqueeze(0)
        key = key.transpose(0,1).unsqueeze(0)
        value = value.transpose(0,1).unsqueeze(0)
        key = key[:,:,None,:,:].expand(1,kvheads,repeat,key_length,dim).reshape(1,heads,key_length,dim)
        value = value[:,:,None,:,:].expand(1,kvheads,repeat,key_length,dim).reshape(1,heads,key_length,dim)
        scores = torch.matmul(query,key.transpose(2,3))
        if (__import__('os').environ.get('ATOM_MELLUM_FUSED_PREFILL_SOFTMAX') == '1'
                and count == context_length and 512 <= key_length <= 16384):
            variant = __import__('os').environ.get('ATOM_MELLUM_PREFILL_SOFTMAX_VARIANT', 'legacy')
            if variant == 'hybrid':
                from atom.model_ops.mellum_prefill_softmax_hybrid import softmax as fused_softmax
            else:
                assert variant == 'legacy', variant
                from atom.model_ops.mellum_prefill_softmax import softmax as fused_softmax
            probabilities = fused_softmax(scores, count, context_length, window, scale)
        else:
            scores = scores * scale
            mask = attention_mask(q, count, key_length, context_length, window)
            scores = scores + mask[None,None,:,:]
            probabilities = softmax_bf16(scores)
        del scores
        result = torch.matmul(probabilities,value)
        out[start:stop] = result.transpose(1,2).reshape(count,heads,dim)
    return out


def attention_mask(q, count, key_length, context_length, window):
    """Reuse immutable masks only inside the current model forward."""
    from atom.utils.forward_context import get_forward_context
    ctx = get_forward_context()
    cache = getattr(ctx, '_mellum_attention_masks', None)
    if cache is None:
        cache = {}
        ctx._mellum_attention_masks = cache
    key = (count, key_length, context_length, window, q.dtype, q.device)
    mask = cache.get(key)
    if mask is None:
        qi = torch.arange(context_length-count,context_length,device=q.device)[:,None]
        ki = torch.arange(context_length-key_length,context_length,device=q.device)[None,:]
        allowed = ki <= qi
        if window is not None and window > 0:
            allowed = allowed & (ki > qi-window)
        mask = torch.zeros((count,key_length),dtype=q.dtype,device=q.device)
        mask.masked_fill_(~allowed,torch.finfo(q.dtype).min)
        cache[key] = mask
    return mask


def softmax_bf16(scores):
    # Preserve QK and PV GEMM extents. Only partition independent softmax rows.
    if scores.numel() <= (1 << 28):
        return F.softmax(scores,dim=-1,dtype=torch.float32).to(scores.dtype)
    output=torch.empty_like(scores)
    for start in range(0,scores.shape[1],4):
        output[:,start:start+4]=F.softmax(scores[:,start:start+4],dim=-1,dtype=torch.float32).to(scores.dtype)
    return output

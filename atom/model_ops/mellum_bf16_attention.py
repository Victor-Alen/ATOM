"""HF-order BF16 attention over the production request and paged KV cache.

No model reference state or saved tensors. Uses BF16 score/PV GEMMs and FP32
softmax with the same materialization boundaries as native HF eager attention.
"""
import torch
import torch.nn.functional as F


def attention(q, k, v, kc, vc, metadata, scale, window, cache_format='NHD'):
    assert q.ndim == 3 and q.shape[-1] == 128
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    lengths = metadata.context_lens.tolist()
    cu = metadata.cu_seqlens_q
    offsets = cu.tolist() if cu is not None else list(range(len(lengths) + 1))
    assert len(offsets) == len(lengths) + 1 and offsets[-1] == q.shape[0]
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
            blocks = metadata.block_tables[b, :(context_length+block_size-1)//block_size].long()
            key = kc.index_select(0,blocks).permute(0,3,1,2,4).reshape(-1,kvheads,dim)[:context_length]
            cached_value=vc.index_select(0,blocks)
            if cache_format == 'SHUFFLE':
                pack=16//vc.element_size()
                value=cached_value.view(len(blocks),kvheads,block_size//pack,dim,pack).permute(0,2,4,1,3).reshape(-1,kvheads,dim)[:context_length]
            else:
                assert cache_format == 'NHD',cache_format
                value=cached_value.permute(0,3,1,2).reshape(-1,kvheads,dim)[:context_length]
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
        scores = torch.matmul(query,key.transpose(2,3)) * scale
        qi = torch.arange(context_length-count,context_length,device=q.device)[:,None]
        ki = torch.arange(context_length-key_length,context_length,device=q.device)[None,:]
        allowed = ki <= qi
        if window is not None and window > 0:
            allowed = allowed & (ki > qi-window)
        mask = torch.zeros((count,key_length),dtype=q.dtype,device=q.device)
        mask.masked_fill_(~allowed,torch.finfo(q.dtype).min)
        scores = scores + mask[None,None,:,:]
        probabilities = F.softmax(scores,dim=-1,dtype=torch.float32).to(q.dtype)
        del scores
        result = torch.matmul(probabilities,value)
        out[start:stop] = result.transpose(1,2).reshape(count,heads,dim)
    return out

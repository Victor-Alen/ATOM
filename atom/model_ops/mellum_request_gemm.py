"""Request-shaped GEMMs for Mellum parity under continuous batching."""
import torch
import torch.nn.functional as F
from atom.utils.forward_context import get_forward_context

def ranges(x):
 ctx=get_forward_context();meta=ctx.attn_metadata
 if isinstance(meta,dict):
  if not meta:return [(0,x.shape[0])]
  meta=next(iter(meta.values()))
 if meta is None or meta.context_lens is None:return [(0,x.shape[0])]
 cu=meta.cu_seqlens_q
 offsets=cu.tolist() if cu is not None else list(range(len(meta.context_lens)+1))
 assert offsets[-1]==x.shape[0],(offsets,x.shape)
 return list(zip(offsets,offsets[1:]))

def linear(x,weight,bias=None):
 parts=ranges(x)
 if len(parts)==1:return F.linear(x,weight,bias)
 return torch.cat([F.linear(x[a:b],weight,bias) for a,b in parts],dim=0)

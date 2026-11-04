"""HF-order RoPE/YaRN from model parameters, shared across production layers."""
import json
import torch
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

_CACHE={}

def rotate(config,layer_index,positions,q,k):
    kind=config.layer_types[layer_index]
    params=config.rope_parameters[kind]
    dim=getattr(config,'head_dim',None) or config.hidden_size//config.num_attention_heads
    assert params.get('partial_rotary_factor',1.0)==1.0
    key=(str(q.device),str(q.dtype),dim,config.max_position_embeddings,json.dumps(params,sort_keys=True))
    if key not in _CACHE:
        if params['rope_type']=='default':
            inv=1.0/(params['rope_theta']**(torch.arange(0,dim,2,dtype=torch.float32,device='cpu')/dim))
            factor=1.0
        else:
            inv,factor=ROPE_INIT_FUNCTIONS[params['rope_type']](config,device=torch.device('cpu'),layer_type=kind)
        pos=torch.arange(config.max_position_embeddings,device=q.device,dtype=torch.float32)
        freqs=pos[:,None]*inv.to(q.device)[None,:]
        emb=torch.cat((freqs,freqs),-1)
        _CACHE[key]=((emb.cos()*factor).to(q.dtype),(emb.sin()*factor).to(q.dtype))
    cos,sin=_CACHE[key]
    positions=positions.reshape(-1).long()
    cos=cos.index_select(0,positions)[:,None,:]
    sin=sin.index_select(0,positions)[:,None,:]
    def apply(x):
        shape=x.shape
        x=x.reshape(len(positions),-1,dim)
        a,b=x.chunk(2,-1)
        return (x*cos+torch.cat((-b,a),-1)*sin).reshape(shape)
    return apply(q),apply(k)

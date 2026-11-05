"""Paged sliding decode with explicit HF BF16 boundaries and MFMA16/KPack8."""
import torch
import torch.nn.functional as F
import triton as tr
import triton.language as tl

@tr.jit
def _qk(Q,K,BT,L,S,Q0:tl.constexpr,Q1:tl.constexpr,K0:tl.constexpr,K1:tl.constexpr,K2:tl.constexpr,K3:tl.constexpr,
        BT0:tl.constexpr,BS:tl.constexpr,G:tl.constexpr,H:tl.constexpr,W:tl.constexpr,SCALE:tl.constexpr):
    seq=tl.program_id(0);kv=tl.program_id(1);tile=tl.program_id(2)
    t=tile*64+tl.arange(0,64);d=tl.arange(0,128);h=tl.arange(0,16)
    length=tl.load(L+seq);pos=tl.maximum(0,length-W)+t
    blocks=tl.load(BT+seq*BT0+pos//BS,pos<length,0)
    a=tl.load(K+blocks[:,None]*K0+kv*K1+(d[None,:]//8)*K2+(pos[:,None]%BS)*K3+d[None,:]%8,pos[:,None]<length,0)
    b=tl.load(Q+seq*Q0+(kv*G+h[None,:])*Q1+d[:,None],h[None,:]<G,0)
    dot=tl.dot(a,b).to(tl.bfloat16).to(tl.float32)
    score=(dot*SCALE).to(tl.bfloat16)
    score=tl.where(pos[:,None]<length,score,float('-inf'))
    tl.store(S+(seq*H+kv*G+h[None,:])*W+t[:,None],score,h[None,:]<G)

@tr.jit
def _pv(P,V,BT,L,O,V0:tl.constexpr,V1:tl.constexpr,V2:tl.constexpr,V3:tl.constexpr,
        BT0:tl.constexpr,BS:tl.constexpr,G:tl.constexpr,H:tl.constexpr,W:tl.constexpr,SHUFFLE:tl.constexpr):
    seq=tl.program_id(0);kv=tl.program_id(1);tile=tl.program_id(2)
    d=tile*64+tl.arange(0,64);t=tl.arange(0,64);h=tl.arange(0,16)
    length=tl.load(L+seq);start=tl.maximum(0,length-W)
    acc=tl.full((64,16),0,tl.float32)
    for step in range(W//64):
        ti=step*64+t;pos=start+ti
        blocks=tl.load(BT+seq*BT0+pos//BS,pos<length,0)
        if SHUFFLE:
            addr=blocks[None,:]*V0+kv*V1+(pos[None,:]%BS//8)*128*8+d[:,None]*8+pos[None,:]%8
        else:
            addr=blocks[None,:]*V0+kv*V1+d[:,None]*V2+(pos[None,:]%BS)*V3
        a=tl.load(V+addr,pos[None,:]<length,0)
        b=tl.load(P+(seq*H+kv*G+h[None,:])*W+ti[:,None],h[None,:]<G,0)
        acc=tl.dot(a,b,acc)
    tl.store(O+(seq*H+kv*G+h[None,:])*128+d[:,None],acc,h[None,:]<G)

def attention(q,kc,vc,bt,lengths,scale,cache_format,return_intermediates=False,host_lengths=None):
    n,heads,dim=q.shape;kv=kc.shape[1];g=heads//kv
    assert dim==128 and g<=16 and kc.shape[4]==8
    scores=torch.empty((n,heads,1024),device=q.device,dtype=q.dtype)
    out=torch.empty_like(q)
    options=dict(num_warps=4,matrix_instr_nonkdim=16,kpack=2,enable_fp_fusion=False)
    _qk[(n,kv,16)](q,kc,bt,lengths,scores,*q.stride()[:2],*kc.stride()[:4],bt.stride(0),kc.shape[3],g,heads,1024,scale,**options)
    probabilities=F.softmax(scores,dim=-1,dtype=torch.float32).to(q.dtype)
    # Retain the exact native PV dispatch while the fused PV reduction is unverified.
    if host_lengths is None:
        host_lengths=lengths.tolist()
    block_size=kc.shape[3]
    for row,length in enumerate(host_lengths):
        effective=min(length,1024)
        first=length-effective;first_block,skip=divmod(first,block_size)
        blocks=bt[row,first_block:(length+block_size-1)//block_size].long()
        values=vc.index_select(0,blocks)
        if cache_format=='SHUFFLE':
            values=values.view(len(blocks),kv,block_size//8,128,8).permute(0,2,4,1,3).reshape(-1,kv,128)
        else:
            assert cache_format=='NHD'
            values=values.permute(0,3,1,2).reshape(-1,kv,128)
        values=values[skip:skip+effective].transpose(0,1).unsqueeze(0)
        values=values[:,:,None].expand(1,kv,g,effective,128).reshape(1,heads,effective,128)
        p=probabilities[row:row+1,:,:effective].unsqueeze(2).contiguous()
        out[row:row+1]=torch.matmul(p,values).squeeze(2)
    return (out,scores,probabilities) if return_intermediates else out

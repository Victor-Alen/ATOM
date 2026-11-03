"""Experimental BF16 paged sliding decode with HF materialization boundaries.

Three GPU kernels: tiled BF16 scores, full-window softmax, then BF16 P @ V.
Only batch decode with one query, D128, window1024, BF16 unquantized caches.
"""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _qk(Q,K,BT,L,S,
    Q0:tl.constexpr,Q1:tl.constexpr,K0:tl.constexpr,K1:tl.constexpr,K2:tl.constexpr,K3:tl.constexpr,K4:tl.constexpr,
    BT0:tl.constexpr,BS:tl.constexpr,X:tl.constexpr,G:tl.constexpr,NKV:tl.constexpr,W:tl.constexpr,SCALE:tl.constexpr):
    seq=tl.program_id(0);kv=tl.program_id(1);tile=tl.program_id(2)
    h=tl.arange(0,16);d=tl.arange(0,128);t=tile*128+tl.arange(0,128)
    n=tl.load(L+seq);pos=tl.maximum(0,n-W)+t
    blocks=tl.load(BT+seq*BT0+pos//BS,mask=pos<n,other=0)
    q=tl.load(Q+seq*Q0+(kv*G+h[:,None])*Q1+d[None,:],mask=h[:,None]<G,other=0)
    k=tl.load(K+blocks[None,:]*K0+kv*K1+(d[:,None]//X)*K2+(pos[None,:]%BS)*K3+(d[:,None]%X)*K4,mask=pos[None,:]<n,other=0)
    qk=tl.dot(q,k).to(tl.bfloat16).to(tl.float32)
    logits=(qk*SCALE).to(tl.bfloat16).to(tl.float32)
    logits=tl.where(pos[None,:]<n,logits,float('-inf'))
    tl.store(S+((seq*NKV+kv)*16+h[:,None])*W+t[None,:],logits.to(tl.bfloat16))

@tr.jit
def _softmax(S,P,W:tl.constexpr):
    row=tl.program_id(0);t=tl.arange(0,W)
    logits=tl.load(S+row*W+t).to(tl.float32)
    e=tl.exp(logits-tl.max(logits,0));p=e/tl.sum(e,0)
    tl.store(P+row*W+t,p.to(tl.bfloat16))

@tr.jit
def _pv(P,V,BT,L,O,
    V0:tl.constexpr,V1:tl.constexpr,V2:tl.constexpr,V3:tl.constexpr,
    O0:tl.constexpr,O1:tl.constexpr,BT0:tl.constexpr,BS:tl.constexpr,G:tl.constexpr,NKV:tl.constexpr,W:tl.constexpr):
    seq=tl.program_id(0);kv=tl.program_id(1)
    h=tl.arange(0,16);d=tl.arange(0,128);t=tl.arange(0,128)
    n=tl.load(L+seq);start=tl.maximum(0,n-W)
    out=tl.zeros((16,128),tl.float32)
    for tile in range(W//128):
        tt=tile*128+t;pos=start+tt
        blocks=tl.load(BT+seq*BT0+pos//BS,mask=pos<n,other=0)
        p=tl.load(P+((seq*NKV+kv)*16+h[:,None])*W+tt[None,:])
        v=tl.load(V+blocks[:,None]*V0+kv*V1+d[None,:]*V2+(pos[:,None]%BS)*V3,mask=pos[:,None]<n,other=0)
        out=tl.dot(p,v,out)
    tl.store(O+seq*O0+(kv*G+h[:,None])*O1+d[None,:],out,mask=h[:,None]<G)

def paged_decode_hf_rounding(q,kc,vc,bt,lens,scale,window=1024,*,return_intermediates=False):
    assert q.dtype==kc.dtype==vc.dtype==torch.bfloat16
    assert q.ndim==3 and q.shape[-1]==128 and kc.ndim==5 and vc.ndim==4
    assert window==1024 and vc.shape[-1]==kc.shape[3]
    n,heads,d=q.shape;kv=kc.shape[1];g=heads//kv
    assert heads%kv==0 and g<=16 and bt.shape[0]==n
    p=torch.empty((n,kv,16,window),dtype=q.dtype,device=q.device);scores=torch.empty_like(p);out=torch.empty_like(q)
    _qk[(n,kv,window//128)](q,kc,bt,lens,scores,*q.stride()[:2],*kc.stride(),bt.stride(0),kc.shape[3],kc.shape[4],g,kv,window,scale,num_warps=4,enable_fp_fusion=False)
    _softmax[(n*kv*16,)](scores,p,window,num_warps=4,enable_fp_fusion=False)
    _pv[(n,kv)](p,vc,bt,lens,out,*vc.stride(),*out.stride()[:2],bt.stride(0),kc.shape[3],g,kv,window,num_warps=4,enable_fp_fusion=False)
    return (out,p,scores) if return_intermediates else out

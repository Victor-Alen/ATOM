"""Experimental BF16 sliding attention with HF materialization boundaries.

Three GPU kernels: tiled BF16 scores, full-window softmax, then BF16 P @ V.
Decode: one query per sequence. Dense prefill: verified M=N=990 profile only.
D128, window1024, BF16 unquantized caches; three actual Triton GPU kernels.
"""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.hip import libdevice

@tr.jit
def _qk(Q,K,BT,L,S,
    Q0:tl.constexpr,Q1:tl.constexpr,K0:tl.constexpr,K1:tl.constexpr,K2:tl.constexpr,K3:tl.constexpr,K4:tl.constexpr,
    BT0:tl.constexpr,BS:tl.constexpr,X:tl.constexpr,G:tl.constexpr,NKV:tl.constexpr,W:tl.constexpr,SCALE:tl.constexpr,DENSE:tl.constexpr=False):
    seq=tl.program_id(0);kv=tl.program_id(1);tile=tl.program_id(2)
    h=tl.arange(0,16);d=tl.arange(0,128);t=tile*128+tl.arange(0,128)
    n=tl.load(L+seq);pos=tl.maximum(0,n-W)+t
    if not DENSE:
        blocks=tl.load(BT+seq*BT0+pos//BS,mask=pos<n,other=0)
    q=tl.load(Q+seq*Q0+(kv*G+h[:,None])*Q1+d[None,:],mask=h[:,None]<G,other=0)
    if DENSE:
        k=tl.load(K+pos[None,:]*K0+kv*K1+d[:,None]*K2,mask=pos[None,:]<n,other=0)
    else:
        k=tl.load(K+blocks[None,:]*K0+kv*K1+(d[:,None]//X)*K2+(pos[None,:]%BS)*K3+(d[:,None]%X)*K4,mask=pos[None,:]<n,other=0)
    if DENSE:
        acc=tl.zeros((128,16),tl.float32)
        for chunk in tl.static_range(8):
            # HF's profiled M=N=990 Tensile kernel: query tile320, DepthU32,
            # StaggerU16 reduced to four K iterations, mapping to query WG.
            dd=(chunk*16+((seq//320)%4)*32+tl.arange(0,16))%128
            qa=tl.load(Q+seq*Q0+(kv*G+h[:,None])*Q1+dd[None,:],mask=h[:,None]<G,other=0)
            kb=tl.load(K+pos[None,:]*K0+kv*K1+dd[:,None]*K2,mask=pos[None,:]<n,other=0)
            acc=tl.dot(tl.trans(kb),tl.trans(qa),acc)
        qk=tl.trans(acc).to(tl.bfloat16).to(tl.float32)
    else:
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
def _softmax_hf_tree(S,P,W:tl.constexpr):
    row=tl.program_id(0);lane=tl.arange(0,64)
    mx=tl.full((64,),float('-inf'),tl.float32)
    for it in tl.static_range(W//64):
        x=tl.load(S+row*W+it*64+lane).to(tl.float32)
        mx=tl.maximum(mx,x)
    mx=tl.max(mx,0)
    values=()
    acc=tl.full((64,),0,tl.float32)
    for it in tl.static_range(W//64):
        x=tl.load(S+row*W+it*64+lane).to(tl.float32)
        e=libdevice.exp(x-mx)
        values+=(e,)
        acc=acc+e
    # PyTorch PersistentSoftmax: XOR 32,16,8,4,2,1, not Triton's tree.
    for level in tl.static_range(6):
        acc=acc+tl.gather(acc,lane^(32>>level),0)
    den=tl.sum(tl.where(lane==0,acc,0),0)
    for it in tl.static_range(W//64):
        value=tl.div_rn(values[it],den)
        tl.store(P+row*W+it*64+lane,value.to(tl.bfloat16))

@tr.jit
def _pv(P,V,BT,L,O,
    V0:tl.constexpr,V1:tl.constexpr,V2:tl.constexpr,V3:tl.constexpr,
    O0:tl.constexpr,O1:tl.constexpr,BT0:tl.constexpr,BS:tl.constexpr,G:tl.constexpr,NKV:tl.constexpr,W:tl.constexpr,DENSE:tl.constexpr=False):
    seq=tl.program_id(0);kv=tl.program_id(1)
    h=tl.arange(0,16);d=tl.arange(0,128);t=tl.arange(0,128)
    n=tl.load(L+seq);start=tl.maximum(0,n-W)
    out=tl.zeros((16,128),tl.float32)
    for tile in range(W//128):
        tt=tile*128+t;pos=start+tt
        if not DENSE:
            blocks=tl.load(BT+seq*BT0+pos//BS,mask=pos<n,other=0)
        p=tl.load(P+((seq*NKV+kv)*16+h[:,None])*W+tt[None,:])
        if DENSE:
            v=tl.load(V+pos[:,None]*V0+kv*V1+d[None,:]*V2,mask=pos[:,None]<n,other=0)
        else:
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


def dense_prefill_hf_rounding(q,k,v,scale,window=1024,*,return_intermediates=False):
    assert q.dtype==k.dtype==v.dtype==torch.bfloat16
    assert q.ndim==k.ndim==v.ndim==3 and q.shape[-1]==128 and window==1024
    n,heads,d=q.shape;kv=k.shape[1];g=heads//kv
    assert n==990 and heads==32 and kv==4, "unverified HF GEMM profile"
    assert n==k.shape[0]==v.shape[0] and g<=16
    lens=torch.arange(1,n+1,dtype=torch.int32,device=q.device)
    p=torch.empty((n,kv,16,window),dtype=q.dtype,device=q.device);scores=torch.empty_like(p);out=torch.empty_like(q)
    _qk[(n,kv,window//128)](q,k,lens,lens,scores,*q.stride()[:2],*k.stride(),0,0,0,16,8,g,kv,window,scale,DENSE=True,num_warps=4,enable_fp_fusion=False)
    _softmax_hf_tree[(n*kv*16,)](scores,p,window,num_warps=1,enable_fp_fusion=False)
    _pv[(n,kv)](p,v,lens,lens,out,*v.stride(),0,*out.stride()[:2],0,16,g,kv,window,DENSE=True,num_warps=4,enable_fp_fusion=False)
    return (out,p,scores) if return_intermediates else out

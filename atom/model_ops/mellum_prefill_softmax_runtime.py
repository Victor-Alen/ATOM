"""Fused BF16 scale/mask/FP32 softmax with installed ATen ROCm reduction order.

Matches ROCm/pytorch 3d3aa833 SoftMax.cu and PersistentSoftmax.cuh.
Keeps native QK and PV GEMMs unchanged.
"""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.hip import libdevice

@tr.jit
def _large_exp_ftz(x):
    # ATen large-row fast exp flushes only exponential underflow. Keep the
    # subsequent normalization product in IEEE mode, including subnormals.
    e = tl.exp(x)
    return tl.where(e < 1.1754943508222875e-38, 0.0, e)

@tr.jit
def _at(values,idx,N,B:tl.constexpr):
    value=tl.gather(values,tl.minimum(tl.maximum(idx,0),B-1),axis=0)
    return tl.where((idx>=0)&(idx<N),value,0.0)

@tr.jit
def _softmax(S,P,N,B:tl.constexpr,Q,CTX,W:tl.constexpr,SCALE:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,B)
    base=row.to(tl.int64)*N
    raw=tl.load(S+base+col,col<N,0).to(tl.float32)
    scaled=(raw*SCALE).to(tl.bfloat16).to(tl.float32)
    qpos=row%Q+CTX-Q;kpos=col+CTX-N
    valid=(col<N)&(kpos<=qpos)
    if W>0:valid=valid&(kpos>qpos-W)
    logits=tl.where(valid,scaled,float('-inf'))
    maximum=tl.max(logits,axis=0)
    if B<=2048:
        e=libdevice.exp(logits-maximum)
        lane=tl.arange(0,64);acc=tl.full((64,),0,tl.float32)
        for i in tl.static_range(tr.cdiv(B,64)):
            acc=acc+_at(e,lane+i*64,N,B)
        for i in tl.static_range(6):
            acc=acc+tl.gather(acc,lane^(32>>i),axis=0)
        den=tl.sum(tl.where(lane==0,acc,0),axis=0)
        result=tl.div_rn(e,den)
    else:
        e=_large_exp_ftz(logits-maximum)
        lane=tl.arange(0,512);acc=tl.full((512,),0,tl.float32)
        if N%4==0:
            for block in tl.static_range(tr.cdiv(B,2048)):
                for j in tl.static_range(4):
                    acc=acc+_at(e,block*2048+lane*4+j,N,B)
        else:
            shift=((row%4)*(N%4))%4
            if shift>0:
                acc=acc+_at(e,lane-shift,N,B)
            start=tl.where(shift>0,512-shift,0)
            remaining=N-start;bulk=remaining-remaining%2048
            for block in tl.static_range(tr.cdiv(B,2048)):
                for j in tl.static_range(4):
                    part=_at(e,start+block*2048+lane*4+j,N,B)
                    acc=acc+tl.where(block*2048<bulk,part,0.0)
            for j in tl.static_range(4):
                acc=acc+_at(e,start+bulk+lane+j*512,N,B)
        for i in tl.static_range(6):
            offset=32>>i
            source=lane//64*64+tl.minimum(lane%64+offset,63)
            acc=acc+tl.gather(acc,source,axis=0)
        first=tl.arange(0,64)
        warp=tl.gather(acc,tl.minimum(first,7)*64,axis=0)
        warp=tl.where(first<8,warp,0.0)
        for i in tl.static_range(6):
            warp=warp+tl.gather(warp,tl.minimum(first+(32>>i),63),axis=0)
        den=tl.sum(tl.where(first==0,warp,0),axis=0)
        inverse=tl.div_rn(1.0,den)
        result=e*inverse
    tl.store(P+base+col,result,col<N)

def softmax(scores,query_count,context_length,window,scale):
    assert scores.dtype==torch.bfloat16 and scores.is_contiguous()
    n=scores.shape[-1]
    output=torch.empty_like(scores)
    _softmax[(scores.numel()//n,)](scores,output,n,tr.next_power_of_2(n),query_count,context_length,window or -1,scale,
        num_warps=8,enable_fp_fusion=False)
    return output

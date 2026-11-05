"""Isolated Gluon row-parallel register cache; preserves ATen small-row order."""
import torch
import triton as tr
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.extra.hip import libdevice

@gluon.jit
def _at_rows(values,idx,N,B:gl.constexpr,cache_start):
    idx=idx-cache_start
    value=gl.gather(values,gl.minimum(gl.maximum(idx,0),B-1),axis=1)
    return gl.where((idx>=0)&(idx+cache_start<N)&(idx<B),value,0.0)

@gluon.jit
def _cached_block(S,P,N,Q,CTX,W:gl.constexpr,SCALE:gl.constexpr,B:gl.constexpr,R:gl.constexpr,FULL:gl.constexpr,SPARSE:gl.constexpr):
    layout:gl.constexpr=gl.BlockedLayout([1,1],[1,64],[R,2],[1,0])
    local=gl.program_id(0)*R+gl.arange(0,R,layout=gl.SliceLayout(1,layout))
    rr=gl.program_id(1)*Q+local
    cc=gl.arange(0,B,layout=gl.SliceLayout(0,layout))
    base=rr.to(gl.int64)*N
    qpos=rr%Q+CTX-Q
    cache_start=gl.maximum(0,gl.minimum(N,qpos+1-CTX+N)-1024)
    global_cc=cache_start[:,None]+cc[None,:]
    allowed=(local[:,None]<Q)&(global_cc<N)&(global_cc+CTX-N<=qpos[:,None])
    if W>0:allowed=allowed&(global_cc+CTX-N>qpos[:,None]-W)
    raw=gl.load(S+base[:,None]+global_cc,allowed,0).to(gl.float32)
    logits=gl.where(allowed,(raw*SCALE).to(gl.bfloat16).to(gl.float32),float('-inf'))
    maximum=gl.max(logits,1)
    if FULL<=2048:
        values=libdevice.exp(logits-maximum[:,None])
        lane=gl.arange(0,64,layout=gl.SliceLayout(0,layout))
        acc=gl.full((R,64),0.0,gl.float32,layout)
        for i in gl.static_range(tr.cdiv(B,64)):
            idx=gl.full((R,64),0,gl.int32,layout)+(lane+i*64)[None,:]
            part=gl.gather(values,gl.minimum(idx,B-1),axis=1)
            acc=acc+gl.where(idx<N,part,0.0)
        for i in gl.static_range(6):
            idx=gl.full((R,64),0,gl.int32,layout)+(lane^(32>>i))[None,:]
            acc=acc+gl.gather(acc,idx,axis=1)
        den=gl.sum(gl.where(lane[None,:]==0,acc,0.0),1)
        out=gl.div_rn(values,den[:,None]).to(gl.bfloat16)
    else:
        values=gl.exp(logits-maximum[:,None])
        values=gl.where(values<1.1754943508222875e-38,0.0,values)
        lanes=gl.arange(0,512,layout=gl.SliceLayout(0,layout))
        zero=gl.full((R,512),0,gl.int32,layout)
        ids=zero+lanes[None,:]
        acc=gl.full((R,512),0.0,gl.float32,layout)
        if N%4==0:
            first_block=gl.min(cache_start,0)//2048
            last_block=gl.minimum((gl.max(cache_start,0)+B+2047)//2048,(N+2047)//2048)
            for block in range(first_block,last_block):
                for j in gl.static_range(4):
                    acc+=_at_rows(values,block*2048+ids*4+j,N,B,cache_start[:,None])
        else:
            shift=((rr%4)*(N%4))%4
            start=gl.where(shift>0,512-shift,0)
            bulk=(N-start)-((N-start)%2048)
            prefix=_at_rows(values,ids-shift[:,None],N,B,cache_start[:,None])
            acc+=gl.where(shift[:,None]>0,prefix,0.0)
            first_block=gl.maximum(0,gl.min(cache_start-start,0)//2048)
            last_block=gl.minimum((gl.max(cache_start-start,0)+B+2047)//2048,gl.max(bulk,0)//2048)
            for block in range(first_block,last_block):
                for j in gl.static_range(4):
                    part=_at_rows(values,start[:,None]+block*2048+ids*4+j,N,B,cache_start[:,None])
                    acc+=gl.where(block*2048<bulk[:,None],part,0.0)
            for j in gl.static_range(4):
                acc+=_at_rows(values,start[:,None]+bulk[:,None]+ids+j*512,N,B,cache_start[:,None])
        for step in gl.static_range(6):
            src=ids//64*64+gl.minimum(ids%64+(32>>step),63)
            acc+=gl.gather(acc,src,axis=1)
        warp=gl.gather(acc,gl.minimum(ids,7)*64,axis=1)
        warp=gl.where(ids<8,warp,0.0)
        for step in gl.static_range(6):
            warp+=gl.gather(warp,gl.minimum(ids+(32>>step),63),axis=1)
        denominator=gl.sum(gl.where(ids==0,warp,0.0),1)
        out=(values*gl.div_rn(1.0,denominator[:,None])).to(gl.bfloat16)
    gl.store(P+base[:,None]+global_cc,out,(local[:,None]<Q)&(global_cc<N))
    if B<FULL and not SPARSE:
        tail=gl.arange(0,FULL,layout=gl.SliceLayout(0,layout))
        gl.store(P+base[:,None]+tail[None,:],0.0,(local[:,None]<Q)&((tail[None,:]<cache_start[:,None])|(tail[None,:]>=cache_start[:,None]+B))&(tail[None,:]<N))

@gluon.jit
def _small_cached_rows(S,P,N,Q,CTX,W:gl.constexpr,SCALE:gl.constexpr,FULL:gl.constexpr,R:gl.constexpr,SPARSE:gl.constexpr):
    _cached_block(S,P,N,Q,CTX,W,SCALE,1024,R,FULL,SPARSE)


def softmax(scores,query_count,context_length,window,scale,rows=4,sparse_output=False):
    assert scores.is_contiguous() and scores.dtype==torch.bfloat16
    n=scores.shape[-1];assert 2048<n<=16384 and window==1024
    out=torch.empty_like(scores);total=scores.numel()//n
    _small_cached_rows[(tr.cdiv(query_count,rows),total//query_count)](scores,out,n,query_count,context_length,window or -1,scale,max(256,tr.next_power_of_2(n)),rows,sparse_output,num_warps=rows*2,enable_fp_fusion=False)
    return out

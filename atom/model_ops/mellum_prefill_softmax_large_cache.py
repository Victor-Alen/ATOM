"""Isolated Gluon row-parallel register cache; preserves ATen small-row order."""
import torch
import triton as tr
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.extra.hip import libdevice

@gluon.jit
def _at_rows(values,idx,N,B:gl.constexpr):
    value=gl.gather(values,gl.minimum(gl.maximum(idx,0),B-1),axis=1)
    return gl.where((idx>=0)&(idx<N)&(idx<B),value,0.0)

@gluon.jit
def _cached_block(S,P,N,Q,CTX,W:gl.constexpr,SCALE:gl.constexpr,B:gl.constexpr,R:gl.constexpr,FULL:gl.constexpr):
    layout:gl.constexpr=gl.BlockedLayout([1,1],[1,64],[R,8],[1,0])
    local=gl.program_id(0)*R+gl.arange(0,R,layout=gl.SliceLayout(1,layout))
    rr=gl.program_id(1)*Q+local
    cc=gl.arange(0,B,layout=gl.SliceLayout(0,layout))
    base=rr.to(gl.int64)*N
    qpos=rr%Q+CTX-Q
    allowed=(local[:,None]<Q)&(cc[None,:]<N)&(cc[None,:]+CTX-N<=qpos[:,None])
    if W>0:allowed=allowed&(cc[None,:]+CTX-N>qpos[:,None]-W)
    raw=gl.load(S+base[:,None]+cc[None,:],allowed,0).to(gl.float32)
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
            for block in gl.static_range(tr.cdiv(FULL,2048)):
                for j in gl.static_range(4):
                    acc+=_at_rows(values,block*2048+ids*4+j,N,B)
        else:
            shift=((rr%4)*(N%4))%4
            start=gl.where(shift>0,512-shift,0)
            bulk=(N-start)-((N-start)%2048)
            prefix=_at_rows(values,ids-shift[:,None],N,B)
            acc+=gl.where(shift[:,None]>0,prefix,0.0)
            for block in gl.static_range(tr.cdiv(FULL,2048)):
                for j in gl.static_range(4):
                    part=_at_rows(values,start[:,None]+block*2048+ids*4+j,N,B)
                    acc+=gl.where(block*2048<bulk[:,None],part,0.0)
            for j in gl.static_range(4):
                acc+=_at_rows(values,start[:,None]+bulk[:,None]+ids+j*512,N,B)
        for step in gl.static_range(6):
            src=ids//64*64+gl.minimum(ids%64+(32>>step),63)
            acc+=gl.gather(acc,src,axis=1)
        warp=gl.gather(acc,gl.minimum(ids,7)*64,axis=1)
        warp=gl.where(ids<8,warp,0.0)
        for step in gl.static_range(6):
            warp+=gl.gather(warp,gl.minimum(ids+(32>>step),63),axis=1)
        denominator=gl.sum(gl.where(ids==0,warp,0.0),1)
        out=(values*gl.div_rn(1.0,denominator[:,None])).to(gl.bfloat16)
    gl.store(P+base[:,None]+cc[None,:],out,(local[:,None]<Q)&(cc[None,:]<N))
    if B<FULL:
        tail=gl.arange(0,FULL,layout=gl.SliceLayout(0,layout))
        gl.store(P+base[:,None]+tail[None,:],0.0,(local[:,None]<Q)&(tail[None,:]>=B)&(tail[None,:]<N))

@gluon.jit
def _small_cached_rows(S,P,N,Q,CTX,W:gl.constexpr,SCALE:gl.constexpr,FULL:gl.constexpr,R:gl.constexpr):
    end=gl.minimum(N,(gl.program_id(0)+1)*R+N-Q)
    if end<=256:
        _cached_block(S,P,N,Q,CTX,W,SCALE,256,R,FULL)
    elif end<=512:
        _cached_block(S,P,N,Q,CTX,W,SCALE,512,R,FULL)
    elif end<=1024:
        _cached_block(S,P,N,Q,CTX,W,SCALE,1024,R,FULL)
    else:
        if FULL<=2048:
            _cached_block(S,P,N,Q,CTX,W,SCALE,2048,R,FULL)
        else:
            if end<=2048:
                _cached_block(S,P,N,Q,CTX,W,SCALE,2048,R,FULL)
            else:
                _cached_block(S,P,N,Q,CTX,W,SCALE,4096,R,FULL)


def softmax(scores,query_count,context_length,window,scale,rows=4):
    assert scores.is_contiguous() and scores.dtype==torch.bfloat16
    n=scores.shape[-1];assert 2048<n<=4096
    out=torch.empty_like(scores);total=scores.numel()//n
    _small_cached_rows[(tr.cdiv(query_count,rows),total//query_count)](scores,out,n,query_count,context_length,window or -1,scale,max(256,tr.next_power_of_2(n)),rows,num_warps=rows*8,enable_fp_fusion=False)
    return out

"""Isolated Gluon row-parallel register cache; preserves ATen small-row order."""
import torch
import triton as tr
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.extra.hip import libdevice

@gluon.jit
def _small_cached_block(S,P,N,Q,CTX,W:gl.constexpr,SCALE:gl.constexpr,B:gl.constexpr,R:gl.constexpr,FULL:gl.constexpr):
    layout:gl.constexpr=gl.BlockedLayout([1,1],[1,64],[R,1],[1,0])
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
    gl.store(P+base[:,None]+cc[None,:],out,(local[:,None]<Q)&(cc[None,:]<N))
    if B<FULL:
        tail=gl.arange(0,FULL,layout=gl.SliceLayout(0,layout))
        gl.store(P+base[:,None]+tail[None,:],0.0,(local[:,None]<Q)&(tail[None,:]>=B)&(tail[None,:]<N))

@gluon.jit
def _small_cached_rows(S,P,N,Q,CTX,W:gl.constexpr,SCALE:gl.constexpr,FULL:gl.constexpr,R:gl.constexpr):
    end=gl.minimum(N,(gl.program_id(0)+1)*R+N-Q)
    if end<=256:
        _small_cached_block(S,P,N,Q,CTX,W,SCALE,256,R,FULL)
    elif end<=512:
        _small_cached_block(S,P,N,Q,CTX,W,SCALE,512,R,FULL)
    elif end<=1024:
        _small_cached_block(S,P,N,Q,CTX,W,SCALE,1024,R,FULL)
    else:
        _small_cached_block(S,P,N,Q,CTX,W,SCALE,2048,R,FULL)


def softmax(scores,query_count,context_length,window,scale,rows=4):
    assert scores.is_contiguous() and scores.dtype==torch.bfloat16
    n=scores.shape[-1];assert 1<=n<=2048
    out=torch.empty_like(scores);total=scores.numel()//n
    _small_cached_rows[(tr.cdiv(query_count,rows),total//query_count)](scores,out,n,query_count,context_length,window or -1,scale,max(256,tr.next_power_of_2(n)),rows,num_warps=rows,enable_fp_fusion=False)
    return out

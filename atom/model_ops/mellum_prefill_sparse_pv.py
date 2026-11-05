"""Experimental SWA PV preserving the frozen HF MFMA reduction order."""
import os
import json
from functools import lru_cache
import torch
import triton as tr
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
def _pv(P,V,O,N,MI:gl.constexpr,KW:gl.constexpr,ROW_TILE:gl.constexpr,DEPTH:gl.constexpr,ROTATE:gl.constexpr,BM:gl.constexpr,BN:gl.constexpr):
    qb=gl.program_id(0);head=gl.program_id(1);qbase=qb*BM
    pl:gl.constexpr=gl.BlockedLayout([1,8],[4,16],[4,1],[1,0])
    vl:gl.constexpr=gl.BlockedLayout([1,8],[4,16],[4,1],[1,0])
    rows=qbase+gl.arange(0,BM,layout=gl.SliceLayout(1,pl))
    pk=gl.arange(0,DEPTH,layout=gl.SliceLayout(0,pl))
    vk=gl.arange(0,DEPTH,layout=gl.SliceLayout(1,vl))
    cols=gl.arange(0,BN,layout=gl.SliceLayout(0,vl))
    mfma:gl.constexpr=gl.amd.AMDMFMALayout(version=3,instr_shape=[MI,MI,16 if MI==16 else 8],transposed=True,warps_per_cta=[1,4])
    lhs:gl.constexpr=gl.DotOperandLayout(0,mfma,k_width=KW)
    rhs:gl.constexpr=gl.DotOperandLayout(1,mfma,k_width=KW)
    acc=gl.zeros((BM,BN),gl.float32,mfma)
    first=gl.maximum(0,(qbase-1023)//DEPTH)
    last=gl.minimum(N//DEPTH,(qbase+BM+DEPTH-1)//DEPTH)
    blocks=last-first
    if ROTATE:
        rotation=(qbase//ROW_TILE)%(N//DEPTH)
        begin=gl.where((rotation>=first)&(rotation<last),rotation,first)
    else:
        begin=first
    for i in range(blocks):
        block=first+(begin-first+i)%blocks
        kk=block*DEPTH
        p=gl.load(P+(head.to(gl.int64)*N+rows[:,None])*N+kk+pk[None,:],(rows[:,None]<N)&(kk+pk[None,:]<=rows[:,None])&(kk+pk[None,:]>rows[:,None]-1024),0)
        v=gl.load(V+(head.to(gl.int64)*N+kk+vk[:,None])*128+cols[None,:])
        acc=gl.amd.cdna3.mfma(gl.convert_layout(p,lhs),gl.convert_layout(v,rhs),acc)
    if N%DEPTH!=0 and qbase+BM>N//DEPTH*DEPTH:
        kk=N//DEPTH*DEPTH
        p=gl.load(P+(head.to(gl.int64)*N+rows[:,None])*N+kk+pk[None,:],(rows[:,None]<N)&(kk+pk[None,:]<N)&(kk+pk[None,:]<=rows[:,None])&(kk+pk[None,:]>rows[:,None]-1024),0)
        v=gl.load(V+(head.to(gl.int64)*N+kk+vk[:,None])*128+cols[None,:],kk+vk[:,None]<N,0)
        tail_lhs:gl.constexpr=gl.DotOperandLayout(0,mfma,k_width=4)
        tail_rhs:gl.constexpr=gl.DotOperandLayout(1,mfma,k_width=4)
        acc=gl.amd.cdna3.mfma(gl.convert_layout(p,tail_lhs),gl.convert_layout(v,tail_rhs),acc)
    ol:gl.constexpr=gl.BlockedLayout([1,4],[4,16],[4,1],[1,0])
    result=gl.convert_layout(acc,ol)
    rr=qbase+gl.arange(0,BM,layout=gl.SliceLayout(1,ol))
    cc=gl.arange(0,BN,layout=gl.SliceLayout(0,ol))
    gl.store(O+(head*N+rr[:,None])*128+cc[None,:],result.to(gl.bfloat16),rr[:,None]<N)

def pv(p,v,mi=16,kw=8,row_tile=288,depth=64,rotate=False):
    n=p.shape[-1]
    assert p.shape==(1,32,n,n) and p.is_contiguous()
    assert v.shape==(1,32,n,128) and v.is_contiguous()
    out=torch.empty_like(v)
    _pv[(tr.cdiv(n,32),32)](p,v,out,n,mi,kw,row_tile,depth,rotate,32,128,num_warps=4,enable_fp_fusion=False)
    return out

@lru_cache(maxsize=1)
def _profiles():
    path=os.environ.get('ATOM_MELLUM_PREFILL_PV_CONFIG')
    if not path:return {}
    data=json.load(open(path))
    if data['torch_version']!=torch.__version__:return {}
    if not torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.startswith('gfx942'):return {}
    return {int(n):p for n,p in data['profiles'].items()}

def try_pv(p,v):
    n=p.shape[-1];profile=_profiles().get(n)
    if profile is None or p.shape!=(1,32,n,n) or v.shape!=(1,32,n,128):return None
    if p.dtype!=torch.bfloat16 or v.dtype!=torch.bfloat16 or p.device!=v.device:return None
    if not p.is_contiguous() or not v.is_contiguous():return None
    return pv(p,v,**profile)


def try_window_attention(scores, value, scale):
    n=scores.shape[-1]
    profile=_profiles().get(n)
    if profile is None or scores.shape!=(1,32,n,n) or value.shape!=(1,32,n,128):return None
    if scores.dtype!=torch.bfloat16 or value.dtype!=torch.bfloat16 or scores.device!=value.device:return None
    if not scores.is_contiguous() or not value.is_contiguous():return None
    from atom.model_ops.mellum_prefill_softmax_window_sparse import softmax
    # Outside-window probabilities are uninitialized. pv() masks every such
    # load; this allocation is never exposed to a dense fallback consumer.
    probabilities=softmax(scores,n,n,1024,scale,rows=1,sparse_output=True)
    return pv(probabilities,value,**profile)

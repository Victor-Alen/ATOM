"""Prefill QK tiles derived from AITER pa_decode_gluon MFMA arithmetic.

Compute only causal/SWA tiles. Unwritten score entries MUST be consumed by the
accepted softmax's masked loads; the caller must retain that masked-load contract.
"""
import os
import json
from functools import lru_cache
import torch
import triton as tr
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
def _qk(Q,K,S,N,QS0,QS1,KS0,KS1,W:gl.constexpr,MI:gl.constexpr,KW:gl.constexpr,SPLIT:gl.constexpr,ROW_TILE:gl.constexpr,DEPTH:gl.constexpr,ORDER:gl.constexpr,BM:gl.constexpr,BN:gl.constexpr):
    if ORDER == 0:
        qb=gl.program_id(0);kb=gl.program_id(1);head=gl.program_id(2)
    elif ORDER == 1:
        head=gl.program_id(0);kb=gl.program_id(1);qb=gl.program_id(2)
    else:
        kb=gl.program_id(0);head=gl.program_id(1);qb=gl.program_id(2)
    qbase=(qb // gl.cdiv(ROW_TILE, BM)) * ROW_TILE + (qb % gl.cdiv(ROW_TILE, BM)) * BM
    qend=gl.minimum(N,(qbase//ROW_TILE+1)*ROW_TILE)
    kbase=gl.maximum(0,((qbase-W+1)//BN)*BN)+kb*BN
    if kbase<N and kbase<qbase+BM:
        qload:gl.constexpr=gl.BlockedLayout([1,8],[4,16],[4,1],[1,0])
        kload:gl.constexpr=gl.BlockedLayout([8,1],[16,4],[1,4],[0,1])
        qr=qbase+gl.arange(0,BM,layout=gl.SliceLayout(1,qload))
        qd=gl.arange(0,128,layout=gl.SliceLayout(0,qload))
        kd=gl.arange(0,128,layout=gl.SliceLayout(1,kload))
        kt=kbase+gl.arange(0,BN,layout=gl.SliceLayout(0,kload))
        rotation=((qbase//ROW_TILE)%(128//DEPTH))*DEPTH
        qd=(qd+rotation)%128
        kd=(kd+rotation)%128
        q=gl.load(Q+qr[:,None]*QS0+head*QS1+qd[None,:],qr[:,None]<qend,0)
        k=gl.load(K+kt[None,:]*KS0+(head//8)*KS1+kd[:,None],kt[None,:]<N,0)
        mfma:gl.constexpr=gl.amd.AMDMFMALayout(version=3,instr_shape=[MI,MI,16 if MI==16 else 8],transposed=True,warps_per_cta=[1,4])
        lhs:gl.constexpr=gl.DotOperandLayout(0,mfma,k_width=KW)
        rhs:gl.constexpr=gl.DotOperandLayout(1,mfma,k_width=KW)
        q=gl.convert_layout(q,lhs)
        zero=gl.zeros((BM,BN),gl.float32,mfma)
        if SPLIT:
            k0=gl.convert_layout(gl.where(kd[:,None]<64,k,0.0).to(gl.bfloat16),rhs)
            k1=gl.convert_layout(gl.where(kd[:,None]>=64,k,0.0).to(gl.bfloat16),rhs)
            score=gl.amd.cdna3.mfma(q,k0,zero)+gl.amd.cdna3.mfma(q,k1,zero)
        else:
            score=gl.amd.cdna3.mfma(q,gl.convert_layout(k,rhs),zero)
        output_layout:gl.constexpr=gl.BlockedLayout([1,4],[4,16],[4,1],[1,0])
        score=gl.convert_layout(score,output_layout)
        rows=qbase+gl.arange(0,BM,layout=gl.SliceLayout(1,output_layout))
        cols=kbase+gl.arange(0,BN,layout=gl.SliceLayout(0,output_layout))
        live=(rows[:,None]<qend)&(cols[None,:]<N)&(cols[None,:]<=rows[:,None])&(cols[None,:]>rows[:,None]-W)
        gl.store(S+(head.to(gl.int64)*N+rows[:,None])*N+cols[None,:],score.to(gl.bfloat16),live)

def qk(q,k,mi=16,kw=4,split=False,row_tile=160,bn=128,order=1,depth=32,bm=64):
    n=q.shape[0];assert q.shape==(n,32,128) and k.shape==(n,4,128)
    assert q.dtype==k.dtype==torch.bfloat16 and q.stride(-1)==k.stride(-1)==1
    out=torch.empty((1,32,n,n),device=q.device,dtype=q.dtype)
    qtiles=tr.cdiv(n,row_tile)*tr.cdiv(row_tile,bm)
    ktiles=tr.cdiv(n,bn)
    grid=[(qtiles,ktiles,32),(32,ktiles,qtiles),(ktiles,32,qtiles)][order]
    _qk[grid](q,k,out,n,q.stride(0),q.stride(1),k.stride(0),k.stride(1),n,mi,kw,split,row_tile,depth,order,bm,bn,num_warps=4,enable_fp_fusion=False)
    return out


def try_qk(q, k):
    # Reuse the frozen HF GEMM arithmetic profile. Unsupported shapes fall back.
    from atom.model_ops.mellum_prefill_sparse_qk import _profiles
    n = q.shape[0]
    profile = _profiles().get(n)
    if profile is None or q.shape != (n, 32, 128) or k.shape != (n, 4, 128):
        return None
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or q.device != k.device:
        return None
    if q.stride(-1) != 1 or k.stride(-1) != 1:
        return None
    if n > 4096 and (q.stride(0) != 4096 or q.stride(1) != 128):
        return None
    # General length bands control occupancy; no prompt contents are consulted.
    if n <= 4096:
        bm, bn = 32, 64
    elif n <= 8192:
        bm, bn = 128, 128
    elif n <= 12288:
        bm, bn = 64, 256
    else:
        bm, bn = 32, 128
    return qk(q, k, **dict(profile, bm=bm, bn=bn))

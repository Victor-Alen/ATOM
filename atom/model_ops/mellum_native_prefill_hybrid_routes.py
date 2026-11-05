"""Fused GPU route metadata and packing for the unchanged native CK stage1."""
import torch
import triton
import triton.language as tl
from atom.model_ops.mellum_native_prefill_hybrid_core import _plan, expert_counts

@triton.jit
def _pack_one(Si, Sw, Logical, Counts, Source, Plan, OutSi, OutSw, OutSe, M, LOOKUP:tl.constexpr, P:tl.constexpr, SP:tl.constexpr, e, block):
    count=tl.load(Counts+e);rows=tl.load(Logical+(count if LOOKUP else e))
    if block<tl.cdiv(count,rows):
        lane=tl.arange(0,256 if P==192 else P);src=tl.load(Source+e)*SP+block*rows+lane
        live=(lane<rows)&(block*rows+lane<count)
        values=tl.load(Si+src,live,M);weights=tl.load(Sw+src,live,0.)
        dst_block=tl.load(Plan+66+e)+block
        tl.store(OutSi+dst_block*P+lane,values,lane<P);tl.store(OutSw+dst_block*P+lane,weights,lane<P)
        tl.store(OutSe+dst_block,e)

@triton.jit
def _plan_groups(Ids, Flags, Logical, Nv, Plans, Source, Counts, Given, N):
    pid=tl.program_id(0)
    P:tl.constexpr=(64,128,64,32,128,256,192)
    Groups:tl.constexpr=(0,1,4,3,5,6,7)
    for g in tl.static_range(7):
        if pid==g:
            _plan(Ids,Flags,Logical[g],Nv,Plans+g*130,Source+g*64,Counts+g*64,Given,N,64,True,P[g],64,Groups[g],True)

@triton.jit
def _pack_groups(Si, Sw, Logical, Counts, Source, Plans, OutSi, OutSw, OutSe, M, MaxBlocks):
    pid=tl.program_id(0)
    P:tl.constexpr=(64,128,64,32,128,256,192)
    start=0
    for g in tl.static_range(7):
        end=start+64*MaxBlocks[g]
        if (pid>=start)&(pid<end):
            e=(pid-start)//MaxBlocks[g];block=(pid-start)%MaxBlocks[g]
            _pack_one(Si,Sw,Logical[g],Counts+g*64,Source+g*64,Plans+g*130,OutSi[g],OutSw[g],OutSe[g],M,True,P[g],64,e,block)
        start=end

def prepare_all(ids,si,sw,nv,flags,logical,logical_mi32,logical_lsu4,logical_wide,logical_wide32,bounds,counts):
    m=ids.shape[0];physical=(64,128,64,32,128,256,192)
    max_blocks=tuple(bounds[g][m][0] for g in range(7))
    capacities=tuple(bounds[g][m][1] for g in range(7))
    logicals=(logical,logical_mi32,logical,logical_lsu4,logical_mi32,logical_wide,logical_wide32)
    plan=torch.empty((7,130),device=ids.device,dtype=torch.int32)
    source=torch.empty((7,64),device=ids.device,dtype=torch.int32);kept=torch.empty_like(source)
    total_slots=sum(c*p for c,p in zip(capacities,physical))
    all_si=torch.empty((total_slots,),device=ids.device,dtype=torch.int32)
    all_sw=torch.empty((total_slots,),device=ids.device,dtype=torch.float32)
    all_se=torch.empty((sum(capacities),),device=ids.device,dtype=torch.int32)
    out_si=[];out_sw=[];out_se=[];a=b=0
    for c,p in zip(capacities,physical):
        out_si.append(all_si[a:a+c*p]);out_sw.append(all_sw[a:a+c*p]);out_se.append(all_se[b:b+c]);a+=c*p;b+=c
    _plan_groups[(7,)](ids,flags,logicals,nv,plan,source,kept,counts,ids.numel())
    _pack_groups[(64*sum(max_blocks),)](si,sw,logicals,kept,source,plan,tuple(out_si),tuple(out_sw),tuple(out_se),m,max_blocks)
    return [(out_si[g],out_sw[g],out_se[g],plan[g]) if capacities[g] else None for g in range(7)]

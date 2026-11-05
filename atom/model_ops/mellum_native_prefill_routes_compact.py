"""Experimental GPU-only native-CK route padding for profiled HF row tiles."""
import torch
import triton
import triton.language as tl

@triton.jit
def _plan(Ids, Flags, Logical, Nv, Plan, Source, Counts, Given, N, B:tl.constexpr, LOOKUP:tl.constexpr, P:tl.constexpr, SP:tl.constexpr, GROUP:tl.constexpr, GIVEN):
    e=tl.arange(0,64)
    if GIVEN:
        counts=tl.load(Given+e)
    else:
        i=tl.arange(0,B)
        expert=tl.load(Ids+i,i<N,0)
        counts=tl.histogram(expert,64)-tl.where(e==0,B-N,0)
    rows=tl.load(Logical+(counts if LOOKUP else e))
    source_blocks=tl.cdiv(counts,SP)
    kind=tl.load(Flags+(counts if LOOKUP else e))
    if GROUP>=0:
        split=(kind==2)|(kind==3)|(kind==10)|(kind==11)|(kind==12)|(kind==16)|(kind==17)
        mi32=(kind==1)|(kind==4)|(kind==9)|(kind==19)|(kind==23)
        if GROUP==0:keep=~split&~mi32&(kind!=22)
        elif GROUP==1:keep=mi32
        elif GROUP==2:keep=split
        elif GROUP==3:keep=kind==2
        elif GROUP==4:keep=split&(kind!=2)
        else:keep=kind==22
        counts=tl.where(keep,counts,0)
    target_blocks=tl.cdiv(counts,rows)
    source_prefix=tl.cumsum(source_blocks)-source_blocks
    target_prefix=tl.cumsum(target_blocks)-target_blocks
    tl.store(Counts+e,counts);tl.store(Source+e,source_prefix)
    tl.store(Plan+2+e,kind)
    tl.store(Plan+66+e,tl.where(counts>0,target_prefix,0))
    tl.store(Plan,tl.sum(target_blocks)*P);tl.store(Plan+1,tl.load(Nv+1)|0x40000000)

@triton.jit
def _pack(Si, Sw, Logical, Counts, Source, Plan, OutSi, OutSw, OutSe, M, LOOKUP:tl.constexpr, P:tl.constexpr, SP:tl.constexpr):
    e=tl.program_id(0);block=tl.program_id(1)
    count=tl.load(Counts+e);rows=tl.load(Logical+(count if LOOKUP else e))
    if block<tl.cdiv(count,rows):
        lane=tl.arange(0,P);src=tl.load(Source+e)*SP+block*rows+lane
        live=(lane<rows)&(block*rows+lane<count)
        values=tl.load(Si+src,live,M);weights=tl.load(Sw+src,live,0.)
        dst_block=tl.load(Plan+66+e)+block
        tl.store(OutSi+dst_block*P+lane,values);tl.store(OutSw+dst_block*P+lane,weights)
        tl.store(OutSe+dst_block,e)

def repack(ids,si,sw,nv,flags,logical,lookup=False,physical=64,source_physical=None,group=-1,given_counts=None,max_blocks_per_expert=None,capacity_blocks=None):
    source_physical=physical if source_physical is None else source_physical
    m=ids.shape[0]
    capacity=(capacity_blocks*physical if capacity_blocks is not None else triton.cdiv((physical//32)*ids.numel()+64*(physical-1),physical)*physical)
    plan=torch.empty((130,),device=ids.device,dtype=torch.int32)
    source=torch.empty((64,),device=ids.device,dtype=torch.int32);counts=torch.empty_like(source)
    out_si=torch.empty((capacity,),device=ids.device,dtype=torch.int32)
    out_sw=torch.empty((capacity,),device=ids.device,dtype=torch.float32)
    out_se=torch.empty((capacity//physical,),device=ids.device,dtype=torch.int32)
    _plan[(1,)](ids,flags,logical,nv,plan,source,counts,counts if given_counts is None else given_counts,ids.numel(),triton.next_power_of_2(ids.numel()) if given_counts is None else 64,lookup,physical,source_physical,group,given_counts is not None)
    _pack[(64,max_blocks_per_expert if max_blocks_per_expert is not None else triton.cdiv(m,32))](si,sw,logical,counts,source,plan,out_si,out_sw,out_se,m,lookup,physical,source_physical)
    return out_si,out_sw,out_se,plan

@triton.jit
def _hist(Ids, Partial, N, B:tl.constexpr):
    pid=tl.program_id(0);i=pid*B+tl.arange(0,B);valid=i<N
    values=tl.load(Ids+i,valid,0)
    c=tl.histogram(values,64)-tl.where(tl.arange(0,64)==0,tl.sum((~valid).to(tl.int32)),0)
    tl.store(Partial+pid*64+tl.arange(0,64),c)

@triton.jit
def _hist_sum(Partial, Counts, R, B:tl.constexpr):
    i=tl.arange(0,B)[:,None];e=tl.arange(0,64)[None,:]
    v=tl.load(Partial+i*64+e,i<R,0)
    tl.store(Counts+tl.arange(0,64),tl.sum(v,axis=0))

def expert_counts(ids):
    n=ids.numel();block=min(triton.next_power_of_2(n),4096);parts=triton.cdiv(n,block)
    partial=torch.empty((parts,64),device=ids.device,dtype=torch.int32)
    _hist[(parts,)](ids,partial,n,block)
    if parts==1:return partial.view(64)
    counts=torch.empty((64,),device=ids.device,dtype=torch.int32)
    _hist_sum[(1,)](partial,counts,parts,triton.next_power_of_2(parts))
    return counts

"""Expert-grouped BF16 decode using HF's MFMA16/KPack8 accumulation order."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _sort(IDS, PAIRS, COUNTS, TOKENS:tl.constexpr, TOPK:tl.constexpr, S:tl.constexpr):
    expert=tl.program_id(0)
    pair=tl.arange(0,S)
    ids=tl.load(IDS+pair,pair<TOKENS*TOPK,-1)
    selected=(ids==expert).to(tl.int32)
    ranks=tl.cumsum(selected,axis=0)-1
    tl.store(PAIRS+expert*TOKENS+ranks,pair,selected!=0)
    tl.store(COUNTS+expert,tl.sum(selected,axis=0))


@tr.jit
def _grouped(X,W,PAIRS,COUNTS,Y,TOKENS:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BK:tl.constexpr):
    tile=tl.program_id(0);expert=tl.program_id(1);chunk=tl.program_id(2)
    count=tl.load(COUNTS+expert)
    if chunk*16<count:
        rows=tile*64+tl.arange(0,64)
        columns=chunk*16+tl.arange(0,16)
        pairs=tl.load(PAIRS+expert*TOKENS+columns,columns<count,0)
        ks=tl.arange(0,BK)
        acc=tl.full((64,16),0,tl.float32)
        for block in range(tr.cdiv(K,BK)):
            ki=block*BK+ks
            a=tl.load(W+expert*N*K+rows[:,None]*K+ki[None,:],(rows[:,None]<N)&(ki[None,:]<K),0)
            b=tl.load(X+pairs[None,:]*K+ki[:,None],(columns[None,:]<count)&(ki[:,None]<K),0)
            acc=tl.dot(a,b,acc)
        tl.store(Y+pairs[None,:]*N+rows[:,None],acc,(rows[:,None]<N)&(columns[None,:]<count))


def linear(x,weight,pairs,counts,tokens,bk=64):
    out=torch.empty((x.shape[0],weight.shape[1]),device=x.device,dtype=x.dtype)
    _grouped[(tr.cdiv(weight.shape[1],64),weight.shape[0],tr.cdiv(tokens,16))](
        x,weight,pairs,counts,out,tokens,weight.shape[1],x.shape[1],bk,
        num_warps=4,matrix_instr_nonkdim=16,kpack=2,enable_fp_fusion=False)
    return out


def experts(module,x,ids,weights):
    tokens,topk=ids.shape
    pairs=torch.empty((module.num_experts,tokens),device=x.device,dtype=torch.int32)
    counts=torch.empty(module.num_experts,device=x.device,dtype=torch.int32)
    _sort[(module.num_experts,)](ids,pairs,counts,tokens,topk,tr.next_power_of_2(tokens*topk),num_warps=4)
    selected=x[:,None,:].expand(tokens,topk,x.shape[-1]).reshape(-1,x.shape[-1]).contiguous()
    gate_up=linear(selected,module.gate_up_proj,pairs,counts,tokens)
    gated=module._apply_gate(gate_up)
    down=linear(gated,module.down_proj,pairs,counts,tokens)
    return (down*weights.reshape(-1,1)).view(tokens,topk,-1).sum(1)

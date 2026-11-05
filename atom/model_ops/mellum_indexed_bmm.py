"""BF16 expert GEMVs with direct indexed weights and MFMA16 accumulation."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _indexed(X,W,IDS,Y,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    tile=tl.program_id(0); pair=tl.program_id(1)
    expert=tl.load(IDS+pair)
    rows=tile*BN+tl.arange(0,BN)
    ks=tl.arange(0,BK)
    cols=tl.arange(0,16)
    acc=tl.full((BN,16),0,tl.float32)
    for offset in range(tr.cdiv(K,BK)):
        ki=offset*BK+ks
        a=tl.load(W+expert*N*K+rows[:,None]*K+ki[None,:],(rows[:,None]<N)&(ki[None,:]<K),0)
        x=tl.load(X+pair*K+ki,ki<K,0)
        b=tl.where(cols[None,:]==0,x[:,None],0).to(tl.bfloat16)
        acc=tl.dot(a,b,acc)
    result=tl.sum(tl.where(cols[None,:]==0,acc,0),axis=1)
    tl.store(Y+pair*N+rows,result,rows<N)


def linear(x,weights,ids,bk=128):
    assert x.is_contiguous() and weights.is_contiguous() and ids.is_contiguous()
    out=torch.empty((x.shape[0],weights.shape[1]),device=x.device,dtype=x.dtype)
    _indexed[(tr.cdiv(weights.shape[1],64),x.shape[0])](x,weights,ids,out,weights.shape[1],x.shape[1],64,bk,
        num_warps=4,matrix_instr_nonkdim=16,kpack=2,enable_fp_fusion=False)
    return out


def experts(module,x,ids,weights,bk=128):
    b,k=ids.shape
    selected=x[:,None,:].expand(b,k,x.shape[-1]).reshape(-1,x.shape[-1]).contiguous()
    gate_up=linear(selected,module.gate_up_proj,ids.flatten(),bk)
    gated=module._apply_gate(gate_up)
    down=linear(gated,module.down_proj,ids.flatten(),bk)
    return (down*weights.reshape(-1,1)).view(b,k,-1).sum(1)

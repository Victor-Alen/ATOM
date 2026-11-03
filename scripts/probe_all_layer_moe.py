#!/usr/bin/env python3
"""Compare CK/HF operators on captured production inputs, all Mellum layers."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open
from aiter import ActivationType, QuantType
from aiter.fused_moe import get_2stage_cfgs, get_padded_M, moe_sorting
from aiter.ops.shuffle import shuffle_weight
from probe_moe_slot_stage2 import remap_ids

MODEL=Path('/data/models/Mellum2.1-12B-A2.5B-Thinking')

def metric(a,b):
    d=(a.float()-b.float()).abs()
    return {'max':float(d.max()),'mean':float(d.mean()),
            'different':int((d!=0).sum()),'total':d.numel()}

def reduce_slots(x):
    acc=torch.zeros((x.shape[0],x.shape[2]),device=x.device,dtype=torch.float32)
    for j in range(x.shape[1]):
        acc.add_(x[:,j].float())
    return acc.to(torch.bfloat16)

def probe(cap,w13_hf,w2_hf,w13,w2):
    x=cap['h2'].cuda()
    ids=cap['sel'].to(device='cuda',dtype=torch.long)
    rw=cap['rw'].to(device='cuda',dtype=torch.float32)
    rw_hf=rw.to(torch.bfloat16)
    m,topk=ids.shape
    inter=w2.shape[-1]
    token_idx=torch.arange(m,device=x.device).unsqueeze(1).expand(-1,topk).reshape(-1)
    eid=ids.reshape(-1)
    perm=torch.argsort(eid)
    inv=torch.argsort(perm)
    offs=torch.histc(eid[perm].int(),bins=64,min=0,max=63).cumsum(0,dtype=torch.int32)
    gate_up=F.grouped_mm(x[token_idx][perm],w13_hf.transpose(-2,-1),offs=offs)
    gate_g,up_g=gate_up.chunk(2,dim=-1)
    hf_gated_g=F.silu(gate_g)*up_g
    hf_down_g=F.grouped_mm(hf_gated_g,w2_hf.transpose(-2,-1),offs=offs)
    hf_term_g=hf_down_g*rw_hf.reshape(-1)[perm,None]
    hf_final=hf_term_g[inv].view(m,topk,-1).sum(1)
    gate=gate_g[inv].view(m,topk,inter)
    up=up_g[inv].view(m,topk,inter)
    hf_gated=hf_gated_g[inv].view(m,topk,inter).contiguous()

    meta=get_2stage_cfgs(get_padded_M(m),2304,896,64,8,
        x.dtype,w13.dtype,w13.dtype,QuantType.No,True,ActivationType.Silu,
        False,0,0,True)
    assert not meta.run_1stage and meta.ksplit==0
    sid,sw,se,nv,_=moe_sorting(ids.int(),rw,64,2304,x.dtype,meta.block_m)
    flat=remap_ids(sid,m,int(nv[0]))
    def stage1(mode):
        os.environ['CK_MELLUM_STAGE1_MODE']=str(mode)
        y=torch.empty((m,topk,inter),dtype=x.dtype,device=x.device)
        meta.stage1(x,w13,w2,sid,se,nv,y,topk,block_m=meta.block_m,
                    a1_scale=None,w1_scale=None,sorted_weights=None)
        return y
    ck_orig=stage1(0)
    ck_align=stage1(1)
    ck_gate=stage1(2)
    ck_up=stage1(3)
    ck_silu=stage1(4)
    os.environ['CK_MELLUM_STAGE1_MODE']='1'
    sid_b,sw_b,se_b,nv_b,_=moe_sorting(ids.int(),rw_hf.float(),64,2304,x.dtype,meta.block_m)
    flat_b=remap_ids(sid_b,m,int(nv_b[0]))
    def stage2(a,weights,round_down,bf16_sort=False):
        os.environ['CK_MELLUM_STAGE2_ROUND_WEIGHT']=str(round_down)
        out=torch.zeros((m*topk,2304),dtype=x.dtype,device=x.device)
        use_flat,use_se,use_nv=(flat_b,se_b,nv_b) if bf16_sort else (flat,se,nv)
        meta.stage2(a.view(m*topk,1,inter),w13,w2,use_flat,use_se,use_nv,out,1,
                    block_m=meta.block_m,w2_scale=None,a2_scale=None,sorted_weights=weights)
        return reduce_slots(out.view(m,topk,2304))
    current=stage2(ck_align,sw,1)
    result={
        'layer':cap['layer_name'],'m':m,'block_m':meta.block_m,
        'route_weight_dtype':str(cap['rw'].dtype),
        'route_weight_bf16_rounding':metric(rw,rw_hf),
        'gemm_gate':metric(ck_gate,gate),'gemm_up':metric(ck_up,up),
        'silu_same_input':metric(ck_silu,F.silu(ck_gate)),
        'multiply_same_input':metric(ck_align,ck_silu*ck_up),
        'original_stage1_vs_hf':metric(ck_orig,hf_gated),
        'aligned_stage1_vs_hf':metric(ck_align,hf_gated),
        'captured_production_vs_recomputed':metric(cap['moe'].cuda(),current),
        'original_moe_vs_hf':metric(stage2(ck_orig,sw,0),hf_final),
        'stage1_aligned_moe_vs_hf':metric(stage2(ck_align,sw,0),hf_final),
        'stage12_aligned_moe_vs_hf':metric(current,hf_final),
        'stage12_and_bf16_route_weight_vs_hf':metric(stage2(ck_align,sw_b,1,True),hf_final),
    }
    if result['aligned_stage1_vs_hf']['max'] >= 0.125:
        delta=(ck_align.float()-hf_gated.float()).abs()
        ix=tuple(int(v) for v in torch.unravel_index(delta.argmax(),delta.shape))
        result['aligned_stage1_max_example']={
            'index':list(ix),'ck_gated':float(ck_align[ix]),'hf_gated':float(hf_gated[ix]),
            'ck_gate':float(ck_gate[ix]),'hf_gate':float(gate[ix]),
            'ck_up':float(ck_up[ix]),'hf_up':float(up[ix]),
            'ck_silu':float(ck_silu[ix]),'hf_silu':float(F.silu(gate)[ix])}
    if cap['layer_name'].find('layers.0.') >= 0 and m > 1:
        from torch.profiler import profile, ProfilerActivity
        from torch.autograd import DeviceType
        result['gemm_profile']={}
        for name,fn in (
            ('hf',lambda:F.grouped_mm(x[token_idx][perm],w13_hf.transpose(-2,-1),offs=offs)),
            ('ck',lambda:stage1(2))):
            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
                fn()
                torch.cuda.synchronize()
            result['gemm_profile'][name]=[e.name for e in prof.events() if e.device_type==DeviceType.CUDA]
        os.environ['CK_MELLUM_STAGE1_MODE']='1'
    mismatch=ck_silu!=F.silu(ck_gate)
    if bool(mismatch.any()):
        result['silu_examples']=[{'input':float(a),'ck':float(b),'torch':float(c)}
            for a,b,c in zip(ck_gate[mismatch][:8],ck_silu[mismatch][:8],F.silu(ck_gate)[mismatch][:8])]
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--captures',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--prefill-tokens',type=int,default=990)
    args=parser.parse_args()
    wm=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    results=[]
    with ExitStack() as stack:
        shards={}
        def weight(name):
            shard=wm[name]
            if shard not in shards:
                shards[shard]=stack.enter_context(safe_open(MODEL/shard,framework='pt',device='cpu'))
            return shards[shard].get_tensor(name)
        for layer in range(28):
            gu=[]; down=[]
            for expert in range(64):
                stem=f'model.layers.{layer}.mlp.experts.{expert}.'
                gu.append(torch.cat((weight(stem+'gate_proj.weight'),weight(stem+'up_proj.weight'))))
                down.append(weight(stem+'down_proj.weight'))
            w13_hf=torch.stack(gu).cuda(); w2_hf=torch.stack(down).cuda()
            w13=shuffle_weight(w13_hf); w2=shuffle_weight(w2_hf)
            for phase in ('prefill','decode'):
                path=Path(args.captures)/f'L{layer:02d}_{phase}.pt'
                cap=torch.load(path,map_location='cpu',weights_only=False)
                assert cap['h2'].shape[0] == (args.prefill_tokens if phase=='prefill' else 1), (path,cap['h2'].shape)
                row=probe(cap,w13_hf,w2_hf,w13,w2)
                row['phase']=phase
                results.append(row)
                Path(args.output).write_text(json.dumps(results,indent=2)+'\n')
                print(json.dumps(row),flush=True)
            del w13_hf,w2_hf,w13,w2,gu,down
    assert len(results)==56

if __name__=='__main__':
    main()

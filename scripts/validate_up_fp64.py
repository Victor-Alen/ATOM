#!/usr/bin/env python3
"""Trace one real CK up accumulator; HF masked-prefix runs are separate probes."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import struct
import statistics
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open
from aiter import ActivationType, QuantType
from aiter.fused_moe import get_2stage_cfgs, get_padded_M, moe_sorting
from aiter.ops.shuffle import shuffle_weight

ROOT=Path('/workspace/logs/step0_continuation/up_fp64_validation')
MODEL=Path('/data/models/Mellum2.1-12B-A2.5B-Thinking')
TARGET=(467,3,132)

def digest(t):
    return hashlib.sha256(t.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--expect-fixed',action='store_true')
    parser.add_argument('--save-baseline',action='store_true')
    args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    cap=torch.load('/workspace/logs/step0_continuation/all_layers_request_stage12_capture/L27_prefill.pt',map_location='cpu',weights_only=False)
    wm=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    with ExitStack() as stack:
        shards={}
        def weight(name):
            shard=wm[name]
            if shard not in shards:
                shards[shard]=stack.enter_context(safe_open(MODEL/shard,framework='pt',device='cpu'))
            return shards[shard].get_tensor(name)
        gu=[]; down=[]
        for expert in range(64):
            stem=f'model.layers.27.mlp.experts.{expert}.'
            gu.append(torch.cat((weight(stem+'gate_proj.weight'),weight(stem+'up_proj.weight'))))
            down.append(weight(stem+'down_proj.weight'))
        w13_hf=torch.stack(gu).cuda(); w2_hf=torch.stack(down).cuda()
    w13=shuffle_weight(w13_hf); w2=shuffle_weight(w2_hf)
    x=cap['h2'].cuda(); ids=cap['sel'].to(device='cuda',dtype=torch.int32)
    rw=cap['rw'].to(device='cuda',dtype=torch.float32)
    m,topk=ids.shape
    assert m==990 and int(ids[TARGET[0],TARGET[1]])==38
    meta=get_2stage_cfgs(get_padded_M(m),2304,896,64,8,x.dtype,w13.dtype,w13.dtype,
                        QuantType.No,True,ActivationType.Silu,False,0,0,True)
    assert meta.block_m==128 and not meta.run_1stage and meta.ksplit==0
    sid,sw,se,nv,_=moe_sorting(ids,rw,64,2304,x.dtype,meta.block_m)
    def stage1(mode):
        os.environ['CK_MELLUM_STAGE1_MODE']=str(mode)
        y=torch.empty((m,topk,896),device=x.device,dtype=x.dtype)
        meta.stage1(x,w13,w2,sid,se,nv,y,topk,block_m=meta.block_m,
                    a1_scale=None,w1_scale=None,sorted_weights=None)
        torch.cuda.synchronize()
        return y
    up=stage1(3); gated=stage1(1); gate_ck=stage1(2)
    token=torch.arange(m,device=x.device).unsqueeze(1).expand(-1,8).reshape(-1)
    eid=ids.long().reshape(-1); perm=torch.argsort(eid); inv=torch.argsort(perm)
    offs=torch.histc(eid[perm].int(),bins=64,min=0,max=63).cumsum(0,dtype=torch.int32)
    hf_gu=F.grouped_mm(x[token][perm],w13_hf.transpose(-2,-1),offs=offs)[inv].view(m,8,1792)
    hf_gate,hf_up=hf_gu.chunk(2,dim=-1)
    hf_gated=F.silu(hf_gate)*hf_up
    def metric(a,b):
        d=(a.float()-b.float()).abs()
        return {'max':float(d.max()),'mean':float(d.mean()),
                'different':int((d!=0).sum()),'total':d.numel()}
    hashes=[]
    for _ in range(40):
        hashes.append(digest(stage1(3)))
    assert len(set(hashes))==1
    for _ in range(10):
        stage1(1)
    timings=[]
    for _ in range(50):
        start=time.perf_counter_ns()
        stage1(1)
        timings.append((time.perf_counter_ns()-start)/1e6)
    report={'case':'L27 prefill M990, token467 slot3 expert38 feature132',
        'up_bf16':float(up[TARGET]),'hf_up_bf16':float(hf_up[TARGET]),
        'gated_bf16':float(gated[TARGET]),'hf_gated_bf16':float(hf_gated[TARGET]),
        'target_up_exact':float(up[TARGET])==float(hf_up[TARGET]),
        'target_gated_exact':float(gated[TARGET])==float(hf_gated[TARGET]),
        'full_up_vs_hf':metric(up,hf_up),'full_gated_vs_hf':metric(gated,hf_gated),
        'repeats':40,'distinct_up_hashes':len(set(hashes)),'up_hash':hashes[0],
        'stage1_median_ms':statistics.median(timings),
        'timing_scope':'fixed sorting, CK stage1 call + Python dispatch and synchronization; not complete MoE or service tok/s',
        'stage1_raw_ms':timings}
    if args.save_baseline:
        torch.save({'up':up.cpu(),'gated':gated.cpu(),'gate':gate_ck.cpu()},ROOT/'baseline.pt')
    elif (ROOT/'baseline.pt').exists():
        base=torch.load(ROOT/'baseline.pt',map_location='cpu',weights_only=False)
        report['gate_vs_baseline']=metric(gate_ck.cpu(),base['gate'])
        report['up_vs_baseline']=metric(up.cpu(),base['up'])
        assert torch.equal(gate_ck.cpu(),base['gate'])
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='stage1_raw_ms'},indent=2),flush=True)
    if args.expect_fixed:
        assert report['target_up_exact'] and report['target_gated_exact'],report

if __name__=='__main__':
    main()

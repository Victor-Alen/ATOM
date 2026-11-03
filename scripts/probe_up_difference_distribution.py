#!/usr/bin/env python3
"""Locate residual BF16 up differences versus HF after CK V3 FP64 chunk merge."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import struct
import statistics
import time
import csv
import math
from collections import Counter, defaultdict

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
    up_cpu=up.cpu(); hf_cpu=hf_up.cpu()
    base=torch.load(ROOT/'baseline.pt',map_location='cpu',weights_only=False)['up']
    mask=up_cpu!=hf_cpu
    old_mask=base!=hf_cpu
    coords=torch.nonzero(mask)
    assert len(coords)==1154,len(coords)
    products=[]
    for start in range(0,len(coords),128):
        ix=coords[start:start+128].to(x.device)
        t,slot,feature=ix.unbind(1)
        expert=ids[t,slot].long()
        dot=(x[t].double()*w13_hf[expert,896+feature].double()).sum(-1)
        products.extend(dot.cpu().tolist())
    def bf16_nearest(v):
        if v==0: return v
        exponent=math.frexp(abs(v))[1]-1
        return math.copysign(math.ldexp(round(math.ldexp(abs(v),7-exponent)),exponent-7),v)
    def bits(v):
        return struct.unpack('!I',struct.pack('!f',v))[0]>>16
    def rank(v):
        b=bits(v)
        return 0x8000-(b&0x7fff) if b&0x8000 else 0x8000+b
    ids_cpu=ids.cpu()
    samples=[]
    for ix,dot in zip(coords.tolist(),products):
        t,slot,feature=ix
        cv,hv,bv=float(up_cpu[t,slot,feature]),float(hf_cpu[t,slot,feature]),float(base[t,slot,feature])
        nearest=bf16_nearest(dot)
        verdict='ck_matches_fp64' if cv==nearest else ('hf_matches_fp64' if hv==nearest else 'neither_matches_fp64')
        samples.append({'token_position':t,'slot':slot,'expert':int(ids_cpu[t,slot]),
            'feature':feature,'ck':cv,'hf':hv,'baseline_ck':bv,'abs_diff':abs(cv-hv),
            'ulp_distance':abs(rank(cv)-rank(hv)),'ck_bits':f'0x{bits(cv):04X}',
            'hf_bits':f'0x{bits(hv):04X}','dot_fp64':dot,'nearest_bf16_fp64':nearest,
            'fp64_verdict':verdict,'origin':'persistent' if bool(old_mask[t,slot,feature]) else 'introduced',
            'ck_error_vs_fp64':abs(cv-dot),'hf_error_vs_fp64':abs(hv-dot)})
    def grouped(key,exposure):
        groups=defaultdict(list)
        for row in samples: groups[key(row)].append(row)
        out=[]
        for value,rows in groups.items():
            denom=exposure(value)
            out.append({'index':value,'count':len(rows),'exposure':denom,
                'rate':len(rows)/denom,'max_abs':max(r['abs_diff'] for r in rows),
                'sum_abs':sum(r['abs_diff'] for r in rows)})
        return sorted(out,key=lambda r:r['count'],reverse=True)
    expert_exposure=Counter(ids_cpu.flatten().tolist())
    summary={'shape':list(up_cpu.shape),'different':len(samples),'total':up_cpu.numel(),
        'different_fraction':len(samples)/up_cpu.numel(),
        'tokens_affected':len({r['token_position'] for r in samples}),
        'features_affected':len({r['feature'] for r in samples}),
        'experts_affected':len({r['expert'] for r in samples}),
        'slots_affected':len({r['slot'] for r in samples}),
        'ulp_histogram':dict(Counter(r['ulp_distance'] for r in samples)),
        'fp64_verdicts':dict(Counter(r['fp64_verdict'] for r in samples)),
        'baseline_different':int(old_mask.sum()),
        'persistent':int((mask&old_mask).sum()),'introduced':int((mask&~old_mask).sum()),
        'resolved':int((~mask&old_mask).sum()),
        'by_token':grouped(lambda r:r['token_position'],lambda _:8*896),
        'by_feature':grouped(lambda r:r['feature'],lambda _:990*8),
        'by_expert':grouped(lambda r:r['expert'],lambda e:expert_exposure[e]*896),
        'by_slot':grouped(lambda r:r['slot'],lambda _:990*896),
        'by_feature_tile128':grouped(lambda r:r['feature']//128,lambda _:990*8*128),
        'by_feature_lane16':grouped(lambda r:r['feature']%16,lambda _:990*8*56),
        'by_token_bucket100':grouped(lambda r:r['token_position']//100,lambda b:min(100,990-b*100)*8*896),
        'largest_differences':sorted(samples,key=lambda r:r['abs_diff'],reverse=True)[:20]}
    dest=Path(args.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(summary,indent=2)+'\n')
    with dest.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(samples[0])); writer.writeheader();writer.writerows(samples)
    torch.save({'up':up_cpu,'hf_up':hf_cpu,'ids':ids_cpu},dest.with_suffix('.pt'))
    print(json.dumps({k:v for k,v in summary.items() if not k.startswith('by_') and k!='largest_differences'},indent=2))

if __name__=='__main__':
    main()

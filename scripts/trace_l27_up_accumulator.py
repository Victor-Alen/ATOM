#!/usr/bin/env python3
"""Trace one real CK up accumulator; HF masked-prefix runs are separate probes."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import struct

import torch
import torch.nn.functional as F
from safetensors import safe_open
from aiter import ActivationType, QuantType
from aiter.fused_moe import get_2stage_cfgs, get_padded_M, moe_sorting
from aiter.ops.shuffle import shuffle_weight

ROOT=Path('/workspace/logs/step0_continuation/l27_accumulator_trace')
MODEL=Path('/data/models/Mellum2.1-12B-A2.5B-Thinking')
TARGET=(467,3,132)

def digest(t):
    return hashlib.sha256(t.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('baseline','trace'))
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
    if args.mode=='baseline':
        up=stage1(3); gated=stage1(1)
        assert float(up[TARGET])==35.5 and float(gated[TARGET])==956.0
        torch.save({'up':up.cpu(),'gated':gated.cpu()},ROOT/'baseline.pt')
        r={'up':float(up[TARGET]),'gated':float(gated[TARGET]),
           'up_hash':digest(up),'gated_hash':digest(gated)}
        (ROOT/'baseline.json').write_text(json.dumps(r,indent=2))
        print(json.dumps(r),flush=True)
        return

    baseline=torch.load(ROOT/'baseline.pt',map_location='cpu',weights_only=False)
    no_trace_up=stage1(3); no_trace_gated=stage1(1)
    assert torch.equal(no_trace_up.cpu(),baseline['up']), 'instrumented binary changed up output with tracing off'
    assert torch.equal(no_trace_gated.cpu(),baseline['gated']), 'instrumented binary changed gated output with tracing off'
    position=torch.nonzero(sid==(TARGET[0] | (TARGET[1]<<24))).flatten()
    assert len(position)==1
    block_m=int(position[0])//meta.block_m
    block_n=TARGET[2]//128
    assert int(se[block_m])==38
    trace=torch.full((4096,),float('nan'),device=x.device,dtype=torch.float32)
    os.environ.update({'CK_MELLUM_TRACE_PTR':str(trace.data_ptr()),
        'CK_MELLUM_TRACE_BLOCK_M':str(block_m),'CK_MELLUM_TRACE_BLOCK_N':str(block_n),
        'CK_MELLUM_TRACE_THREAD':'-1','CK_MELLUM_TRACE_REGISTER':'-1',
        'CK_MELLUM_TRACE_CORRECT':'0'})
    up_map=stage1(3)
    head=trace[:8].cpu().tolist()
    assert torch.equal(up_map.cpu(),baseline['up'])
    tid,reg=int(head[0]),int(head[1])
    os.environ['CK_MELLUM_TRACE_THREAD']=str(tid)
    os.environ['CK_MELLUM_TRACE_REGISTER']=str(reg)
    normal_runs=[]
    for _ in range(3):
        trace.fill_(float('nan'))
        normal=stage1(3)
        assert torch.equal(normal.cpu(),baseline['up']), 'tracing perturbed up output'
        host=trace.cpu()
        count=int(host[4])
        assert count>0 and count==int(host[5])*int(host[7])
        seq=host[16:16+count].tolist()
        assert seq[-1]==float(host[2])
        normal_runs.append({'head':host[:8].tolist(),'values':seq})
    assert normal_runs[0]==normal_runs[1]==normal_runs[2]
    count=len(normal_runs[0]['values']); quantum=2304//count
    assert quantum*count==2304

    os.environ['CK_MELLUM_TRACE_CORRECT']='1'
    trace.fill_(float('nan'))
    fixed_up=stage1(3)
    fixed_trace=trace.cpu().clone()
    fixed_gated=stage1(1)
    changed_up=int((fixed_up.cpu()!=baseline['up']).sum())
    changed_gated=int((fixed_gated.cpu()!=baseline['gated']).sum())
    assert changed_up<=1 and changed_gated<=1, (changed_up,changed_gated)

    # Higher-precision diagnostic prefixes, not HF internal accumulators.
    products=x[TARGET[0]].cpu().double()*w13_hf[38,896+TARGET[2]].cpu().double()
    ref_prefix=products.cumsum(0)
    token=torch.arange(m,device=x.device).unsqueeze(1).expand(-1,8).reshape(-1)
    eid=ids.long().reshape(-1); perm=torch.argsort(eid); inv=torch.argsort(perm)
    selected=x[token][perm]
    offs=torch.histc(eid[perm].int(),bins=64,min=0,max=63).cumsum(0,dtype=torch.int32)
    target_row=int(inv[TARGET[0]*8+TARGET[1]])
    original_row=w13_hf[38,896+TARGET[2]].clone()
    w13_hf[38,896+TARGET[2]].zero_()
    rows=[]
    for i in range(count):
        begin=i*quantum; end=(i+1)*quantum
        w13_hf[38,896+TARGET[2],begin:end]=original_row[begin:end]
        hf=F.grouped_mm(selected,w13_hf.transpose(-2,-1),offs=offs)
        hf_prefix=float(hf[target_row,896+TARGET[2]])
        value=normal_runs[0]['values'][i]
        ck_bf16=float(torch.tensor(value,dtype=torch.float32).to(torch.bfloat16))
        exact=float(ref_prefix[end-1])
        rows.append({'update':i+1,'k_end':end,'ck_acc_fp32':value,
            'dot_prefix_fp64':exact,'ck_error_vs_fp64':value-exact,
            'ck_running_bf16':ck_bf16,'hf_masked_prefix_bf16':hf_prefix,
            'corrected_running_fp32':float(fixed_trace[16+i]),
            'independent_chunk_fp32':float(fixed_trace[512+i])})
    assert rows[-1]['hf_masked_prefix_bf16']==35.25
    report={'target':{'layer':27,'token':467,'slot':3,'expert':38,'feature':132},
        'thread':tid,'register':reg,'block_m':block_m,'block_n':block_n,
        'updates':count,'k_per_update':quantum,'normal_repeat_identical':True,
        'tracing_preserves_entire_up_tensor':True,'trace_off_preserves_entire_gated_tensor':True,
        'raw_ck_up_fp32':normal_runs[0]['head'][2],
        'corrected_up_bf16':float(fixed_up[TARGET]),'corrected_gated_bf16':float(fixed_gated[TARGET]),
        'changed_up_elements':changed_up,'changed_gated_elements':changed_gated,
        'first_ck_fp32_vs_fp64_prefix_difference':next((r['k_end'] for r in rows if r['ck_error_vs_fp64']!=0),None),
        'first_ck_vs_hf_masked_prefix_bf16_difference':next((r['k_end'] for r in rows if r['ck_running_bf16']!=r['hf_masked_prefix_bf16']),None),
        'warning':'HF masked-prefix runs zero the target weight tail with unchanged tensor shapes; they are not snapshots of the original HF kernel internal accumulators.',
        'rows':rows}
    (ROOT/'trace.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2),flush=True)

if __name__=='__main__':
    main()

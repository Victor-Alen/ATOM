#!/usr/bin/env python3
"""Validate functional HF operators against the saved HF step, then isolate CK drift."""
from contextlib import ExitStack
import os
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT=Path(os.environ.get('MELLUM_TRACE_ROOT','/workspace/logs/step0_continuation/first_divergence'))
TARGET=int(os.environ.get('MELLUM_TRACE_STEP','27'))
MODEL=Path('/data/models/Mellum2.1-12B-A2.5B-Thinking')
torch.set_num_threads(8)

def metric(a,b):
    a=a.detach().cpu().reshape(-1).float();b=b.detach().cpu().reshape(-1).float()
    assert a.shape==b.shape,(a.shape,b.shape)
    d=(a-b).abs()
    return {'max':float(d.max()),'mean':float(d.mean()),'different':int((d!=0).sum()),'total':d.numel()}
def info(x):
    x=x.detach().reshape(-1).float().cpu();v,i=x.topk(10)
    return {'argmax':int(x.argmax()),'margin':float(v[0]-v[1]),'top10_ids':i.tolist(),
            'top10_values':v.tolist(),'candidate_1234':float(x[1234]),'candidate_8411':float(x[8411])}
def norm(x,w):
    f=x.float();return w*(f*torch.rsqrt(f.pow(2).mean(-1,keepdim=True)+1e-6)).to(x.dtype)
def experts(h,ids,rw,w13,w2):
    ids=ids.long();token=torch.arange(h.shape[0],device=h.device).unsqueeze(1).expand(-1,8).reshape(-1)
    eid=ids.reshape(-1);p=eid.argsort();inv=p.argsort()
    offs=torch.histc(eid[p].int(),bins=64,min=0,max=63).cumsum(0,dtype=torch.int32)
    gu=F.grouped_mm(h[token][p],w13.transpose(-2,-1),offs=offs)
    gate,up=gu.chunk(2,-1)
    down=F.grouped_mm(F.silu(gate)*up,w2.transpose(-2,-1),offs=offs)
    terms=down*rw.reshape(-1)[p,None]
    return terms[inv].view(h.shape[0],8,-1).sum(1)
def router(h,w):
    gl=F.linear(h,w);rw,ri=gl.float().softmax(-1).topk(8,-1)
    rw=rw/rw.sum(-1,keepdim=True)
    return gl,rw.to(gl.dtype),ri

def main():
    atom=torch.load(ROOT/'atom'/f'step_{TARGET}.pt',map_location='cpu',weights_only=False)['tensors']
    hf=torch.load(ROOT/'hf'/f'step_{TARGET}.pt',map_location='cpu',weights_only=False)['tensors']
    assert int(atom['model.input_ids'].flatten()[0])==int(hf['model.input_ids'].flatten()[0])
    assert int(atom['model.positions'].flatten()[0])==int(hf['model.positions'].flatten()[0])
    wm=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    report={'prompt':'A-0000','step':TARGET,'context_input_token':int(atom['model.input_ids'].reshape(-1)[-1]),'context_position':int(atom['model.positions'].reshape(-1)[-1]),'layers':[]}
    with torch.inference_mode(),ExitStack() as stack:
        shards={}
        def weight(name):
            shard=wm[name]
            if shard not in shards: shards[shard]=stack.enter_context(safe_open(MODEL/shard,framework='pt',device='cpu'))
            return shards[shard].get_tensor(name)
        for layer in range(28):
            row={'layer':layer,'norm_counterfactual':{}}
            for tag in ('input_norm','attn','post_norm','mlp','router'):
                key=f'L{layer}.{tag}.out0';row[tag+'_drift']=metric(atom[key],hf[key])
            row['effective_layer_output_drift']=metric(atom[f'L{layer}.layer.out0']+atom[f'L{layer}.layer.out1'],hf[f'L{layer}.layer.out0'])
            for tag in ('input_norm','post_norm'):
                stem=f'L{layer}.{tag}';w=hf[stem+'.weight'].cuda()
                validation=metric(norm(hf[stem+'.in0'].cuda(),w),hf[stem+'.out0'])
                assert validation['different']==0,(stem,validation)
                raw=atom[stem+'.in0'].cuda();res=atom.get(stem+'.in1')
                eff=raw if res is None else raw+res.cuda()
                f=raw.float() if res is None else raw.float()+res.cuda().float()
                z=f*torch.rsqrt(f.pow(2).mean(-1,keepdim=True)+1e-6)
                actual=atom[stem+'.out0']
                row['norm_counterfactual'][tag]={
                    'hf_on_bf16_sum_vs_atom':metric(norm(eff,w),actual),
                    'fp32_sum_cast_then_weight_vs_atom':metric(z.to(raw.dtype)*w,actual),
                    'fp32_sum_fused_weight_vs_atom':metric((z*w.float()).to(raw.dtype),actual),
                    'weight_matches_hf':torch.equal(atom[stem+'.weight'],hf[stem+'.weight'])}
                if res is not None:row['norm_counterfactual'][tag]['residual_output_vs_bf16_sum']=metric(atom[stem+'.out1'],eff)
            gu=[];down=[]
            for expert in range(64):
                stem=f'model.layers.{layer}.mlp.experts.{expert}.'
                gu.append(torch.cat((weight(stem+'gate_proj.weight'),weight(stem+'up_proj.weight'))))
                down.append(weight(stem+'down_proj.weight'))
            w13=torch.stack(gu).cuda();w2=torch.stack(down).cuda()
            hf_mlp_validation=metric(experts(hf[f'L{layer}.mlp.in0'].cuda(),hf[f'L{layer}.router.out2'].cuda(),hf[f'L{layer}.router.out1'].cuda(),w13,w2),hf[f'L{layer}.mlp.out0'])
            assert hf_mlp_validation['different']==0,(layer,hf_mlp_validation)
            h=atom[f'L{layer}.mlp.in0'].cuda();gw=hf[f'L{layer}.router.weight'].cuda()
            gl,rw,ri=router(h,gw)
            pid=atom[f'L{layer}.routing.ids'].cuda().long();prw=atom[f'L{layer}.routing.weights'].cuda().bfloat16()
            row['routing_same_input']={'logits_diff':metric(gl,atom[f'L{layer}.router.out0']),
                'expert_sets_equal':torch.equal(ri.sort(-1).values,pid.sort(-1).values),
                'hf_ids':ri.cpu().tolist(),'atom_ids':pid.cpu().tolist(),
                'block_m':atom[f'L{layer}.routing.block_m']}
            row['moe_same_input_same_routes_vs_atom']=metric(experts(h,pid,prw,w13,w2),atom[f'L{layer}.mlp.out0'])
            row['moe_same_input_hf_routes_vs_atom']=metric(experts(h,ri,rw,w13,w2),atom[f'L{layer}.mlp.out0'])
            report['layers'].append(row)
            (ROOT/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
            print('LAYER',layer,'validated',flush=True)
            del w13,w2,gu,down
        head=weight('lm_head.weight').cuda()
        assert metric(F.linear(hf['final_norm.out0'].cuda(),head),hf['head.logits'])['different']==0
        ah=atom['head.input'].cuda();on_atom=F.linear(ah,head)
        sum_atom=atom['final_norm.in0'].cuda()+atom['final_norm.in1'].cuda()
        final_norm_hf=norm(sum_atom,hf['final_norm.weight'].cuda())
        report['reference_functional_validated']=True
        report['final_hidden_drift']=metric(ah,hf['final_norm.out0'])
        report['head_same_input_diff']=metric(on_atom,atom['head.logits'])
        report['head_weight_rows_match']=torch.equal(atom['head.weight_rows'],head[atom['head.weight_ids'].cuda()].cpu())
        report['head_vs_sampler']=metric(atom['head.logits'],atom['sampler.logits'])
        report['logits']={'hf':info(hf['head.logits']),'atom':info(atom['head.logits']),
            'hf_head_on_atom_hidden':info(on_atom),
            'hf_final_norm_and_head_on_atom_raw_sum':info(F.linear(final_norm_hf,head))}
        report['final_norm_same_raw_sum_vs_atom']=metric(final_norm_hf,atom['final_norm.out0'])
        (ROOT/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='layers'},indent=2),flush=True)

if __name__=='__main__':main()

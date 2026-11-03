"""First L0 MoE stage difference; validate full-batch HF reconstruction first."""
import os,sys,json,torch
from pathlib import Path
from contextlib import ExitStack
from safetensors import safe_open
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch.nn.functional as F
from aiter import ActivationType,QuantType
from aiter.fused_moe import get_2stage_cfgs,get_padded_M,moe_sorting
from aiter.ops.shuffle import shuffle_weight
root=Path('/workspace/logs/step0_continuation/norm990_active_prod');a=torch.load(root/'atom/step_0.pt',weights_only=False)['tensors'];hf=torch.load('/workspace/logs/step0_continuation/official_generate_prefill_full/hf/step_0.pt',weights_only=False)['tensors'];wp=torch.load(root/'atom/L0_weights.pt',weights_only=False);replay=torch.load(root/'fixed_router_replay.pt',weights_only=False)
w13=wp['w13'].cuda();w2=wp['w2'].cuda();w13.is_shuffled=wp['w13_shuffled'];w2.is_shuffled=wp['w2_shuffled']
h=hf['L0.full.post_norm.out0'].reshape(-1,2304).cuda();ids=hf['L0.full.router.out2'].reshape(-1,8).cuda();rw=hf['L0.full.router.out1'].reshape(-1,8).cuda();m=h.shape[0];model=Path('/data/models/Mellum2.1-12B-A2.5B-Thinking');wm=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
def metric(x,y):
 d=(x.float()-y.to(x.device).reshape_as(x).float()).abs();return {'different':int((d!=0).sum()),'max':float(d.max()),'mean':float(d.mean()),'total':d.numel()}
with ExitStack() as stack:
 shards={}
 def weight(name):
  name='model.layers.0.mlp.experts.'+name;shard=wm[name]
  if shard not in shards:shards[shard]=stack.enter_context(safe_open(model/shard,framework='pt',device='cpu'))
  return shards[shard].get_tensor(name)
 gu=[];down=[]
 for e in range(64):
  gu.append(torch.cat([weight(f'{e}.gate_proj.weight'),weight(f'{e}.up_proj.weight')]))
  down.append(weight(f'{e}.down_proj.weight'))
 w13h=torch.stack(gu).cuda();w2h=torch.stack(down).cuda()

with torch.inference_mode():
 eid=ids.flatten();token=torch.arange(m,device='cuda')[:,None].expand(-1,8).flatten();p=eid.argsort();inv=p.argsort();counts=torch.histc(eid[p].int(),bins=64,min=0,max=63).int();offs=counts.cumsum(0,dtype=torch.int32);inp=h[token][p];wt=w13h.transpose(-1,-2)
 out=F.grouped_mm(inp,wt,offs=offs)
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
  out=F.grouped_mm(inp,wt,offs=offs);torch.cuda.synchronize()
 prof.export_chrome_trace(str(root/'hf_grouped_gateup_profile.json'));events=json.loads((root/'hf_grouped_gateup_profile.json').read_text())['traceEvents'];names=[{'name':e['name'],'launch':e.get('args')} for e in events if e.get('cat')=='kernel']
 iso=torch.load(root/'ck_stage_isolation.pt',weights_only=False);examples=[]
 for t,slot,f in (iso['gate_ck']!=iso['gate_hf']).nonzero()[:8].tolist():
  expert=int(ids[t,slot]);gr=int(inv[t*8+slot]);base=int(offs[expert-1]) if expert else 0;dot=(h[t].double()*w13h[expert,f].double()).sum()
  examples.append({'token':t,'slot':slot,'expert':expert,'feature':f,'expert_m':int(counts[expert]),'row_in_expert':gr-base,'ck':float(iso['gate_ck'][t,slot,f]),'hf':float(iso['gate_hf'][t,slot,f]),'dot_fp64':float(dot),'nearest_bf16_fp64':float(dot.bfloat16())})
 report={'input_shape':list(inp.shape),'weight_shape':list(wt.shape),'kernel_count':len(names),'kernels':names,'first_gate_examples':examples};(root/'hf_grouped_gateup_profile_summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

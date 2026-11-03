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
 eid=ids.flatten();token=torch.arange(m,device='cuda')[:,None].expand(-1,8).flatten();p=eid.argsort();inv=p.argsort();offs=torch.histc(eid[p].int(),bins=64,min=0,max=63).cumsum(0,dtype=torch.int32)
 fused=F.grouped_mm(h[token][p],w13h.transpose(-1,-2),offs=offs);g,u=fused.chunk(2,-1);s=F.silu(g);act=s*u;dn=F.grouped_mm(act,w2h.transpose(-1,-2),offs=offs);weighted=dn*rw.flatten()[p,None];hf_slots=weighted[inv].view(m,8,2304);final=hf_slots.sum(1)
 report={'hf_reconstruction_vs_actual':metric(final,hf['L0.full.mlp.out0']),'weight_w13_matches':torch.equal(shuffle_weight(w13h) if wp['w13_shuffled'] else w13h,w13),'weight_w2_matches':torch.equal(shuffle_weight(w2h) if wp['w2_shuffled'] else w2h,w2)}
 (root/'ck_stage_isolation_profile.json').write_text(json.dumps(report,indent=2)+'\n')
 assert report['hf_reconstruction_vs_actual']['different']==0,report
 assert report['weight_w13_matches'] and report['weight_w2_matches'],report
 gate=g[inv].view(m,8,896);up=u[inv].view(m,8,896);silu=s[inv].view(m,8,896);gated=act[inv].view(m,8,896).contiguous()
 meta=get_2stage_cfgs(get_padded_M(m),2304,896,64,8,h.dtype,w13.dtype,w13.dtype,QuantType.No,True,ActivationType.Silu,False,0,0,True)
 sid,sw,se,nv,_=moe_sorting(ids.int(),rw.float(),64,2304,h.dtype,meta.block_m)
 from atom.model_ops.mellum_ck_korder import ck_hf_korder_plan
 plan=ck_hf_korder_plan(ids.contiguous(),nv)
 def stage1(mode):
  os.environ['CK_MELLUM_STAGE1_MODE']=str(mode);out=torch.empty((m,8,896),device='cuda',dtype=h.dtype);meta.stage1(h,w13,w2,sid,se,plan,out,8,block_m=meta.block_m,a1_scale=None,w1_scale=None,sorted_weights=None);return out
 ckact=stage1(1);ckg=stage1(2);cku=stage1(3);cks=stage1(4);os.environ['CK_MELLUM_STAGE1_MODE']='1'
 report.update({'block_m':meta.block_m,'stage1_vs_actual_replay':metric(ckact,replay['stage1_hf_routes']),'gate':metric(ckg,gate),'up':metric(cku,up),'silu_same_gate':metric(cks,F.silu(ckg)),'multiply_same_gate_up':metric(ckact,cks*cku),'gated':metric(ckact,gated)})
 for name,actual,ref in [('gate',ckg,gate),('up',cku,up),('gated',ckact,gated)]:
  coords=(actual!=ref).nonzero();report['first_'+name]=[{'token':t,'slot':slot,'expert':int(ids[t,slot]),'feature':f,'actual':float(actual[t,slot,f]),'hf':float(ref[t,slot,f])} for t,slot,f in coords[:8].tolist()]
 tokenid=sid&0xffffff;slot=(sid>>24)&0xff;flat=torch.where((tokenid<m)&(slot<8),tokenid*8+slot,m*8)
 def stage2(inp):
  out=torch.zeros((m*8,2304),device='cuda',dtype=h.dtype);meta.stage2(inp.view(m*8,1,896),w13,w2,flat,se,nv,out,1,block_m=meta.block_m,w2_scale=None,a2_scale=None,sorted_weights=sw);return out.view(m,8,2304)
 ck_slots=stage2(gated);acc=torch.zeros((m,2304),device='cuda',dtype=torch.float32)
 for slot in range(8):acc.add_(ck_slots[:,slot].float())
 report['stage2_same_hf_stage1_vs_hf_weighted_slots']=metric(ck_slots,hf_slots);report['final_same_hf_stage1_vs_actual_hf']=metric(acc.bfloat16(),hf['L0.full.mlp.out0'])
 hf_acc=torch.zeros_like(acc)
 for slot in range(8):hf_acc.add_(hf_slots[:,slot].float())
 report['ordered_reduce_hf_slots_vs_hf_sum']=metric(hf_acc.bfloat16(),final)
 torch.save({'ids':ids.cpu(),'gate_hf':gate.cpu(),'up_hf':up.cpu(),'gated_hf':gated.cpu(),'gate_ck':ckg.cpu(),'up_ck':cku.cpu(),'gated_ck':ckact.cpu(),'stage2_same_hf_stage1':ck_slots.cpu(),'weighted_slots_hf':hf_slots.cpu()},root/'ck_stage_isolation_profile.pt')
 (root/'ck_stage_isolation_profile.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

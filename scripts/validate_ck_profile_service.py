from pathlib import Path
import torch,json
r=Path('/workspace/logs/step0_continuation/ck_hf_profile_prod/service_final');a=torch.load(r/'atom/step_0.pt',weights_only=False)['tensors'];hf=torch.load('/workspace/logs/step0_continuation/official_generate_prefill_full/hf/step_0.pt',weights_only=False)['tensors'];iso=torch.load('/workspace/logs/step0_continuation/norm990_active_prod/ck_stage_isolation.pt',weights_only=False);replay=torch.load('/workspace/logs/step0_continuation/norm990_active_prod/ck_stage_isolation_profile.pt',weights_only=False)
def metric(x,y):
 d=(x.reshape(-1).float()-y.reshape(-1).float()).abs();return {'different':int((d!=0).sum()),'total':d.numel(),'max':float(d.max()),'mean':float(d.mean())}
ids=a['L0.full.routing.ids'];hi=iso['ids'];report={};
for name,x,y in [('input_norm',a['L0.full.input_norm.out0'],hf['L0.full.input_norm.out0']),('attn',a['L0.full.post_norm.in0'],hf['L0.full.attn.out0']),('post_norm',a['L0.full.post_norm.out0'],hf['L0.full.post_norm.out0'])]:report[name]=metric(x,y)
report['ck_commit']='5082d415d';report['atom_code_commit']='9082ed20';report['actual_norm_calls']=int(a['norm.hf_prefill.count']);report['gateup_captured_keys']=[k for k in a if 'stage1_' in k]
plan=a['L0.full.moe.stage1_korder_plan'];report['plan_expert49']=int(plan[2+49]);report['plan_expert55']=int(plan[2+55]);report['ids_token0']=ids[0].tolist()
for name in ['gate','up']:
 x=a['L0.full.moe.stage1_'+name];gold=iso[name+'_hf'];canonical=x.gather(1,ids.argsort(-1).unsqueeze(-1).expand_as(x));gold=gold.gather(1,hi.argsort(-1).unsqueeze(-1).expand_as(gold));report[name+'_vs_hf_by_expert']=metric(canonical,gold);report[name+'_token0']=metric(canonical[0],gold[0]);assert report[name+'_token0']['different']==0
 oldnew=replay[name+'_ck'].gather(1,hi.argsort(-1).unsqueeze(-1).expand_as(gold));report[name+'_vs_fixed_replay']=metric(canonical,oldnew)
 dif=(canonical!=gold).nonzero();t,slot,f=dif[0].tolist();report['first_'+name]={'token':t,'expert':int(ids.sort(-1).values[t,slot]),'feature':f,'actual':float(canonical[t,slot,f]),'hf':float(gold[t,slot,f])}
slot49=(ids[0]==49).nonzero().item();slot55=(ids[0]==55).nonzero().item();report['target_up']={'expert':49,'token':0,'feature':783,'actual':float(a['L0.full.moe.stage1_up'][0,slot49,783]),'hf':float(iso['up_hf'][0,(hi[0]==49).nonzero().item(),783])};report['prior_gate']={'actual':float(a['L0.full.moe.stage1_gate'][0,slot55,407]),'hf':float(iso['gate_hf'][0,(hi[0]==55).nonzero().item(),407])};assert report['target_up']['actual']==report['target_up']['hf'];assert report['prior_gate']['actual']==report['prior_gate']['hf'];assert report['gate_vs_fixed_replay']['different']==0 and report['up_vs_fixed_replay']['different']==0
report['moe_final_vs_hf']=metric(a['L0.full.mlp.out0'],hf['L0.full.mlp.out0']);(r/'actual_kernel_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


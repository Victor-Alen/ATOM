import torch,json
from pathlib import Path
from aiter.tuned_gemm import tgemm,get_GEMM_A16W16_config
r=Path('/workspace/logs/step0_continuation/norm990_prod');a=torch.load(r/'atom/step_0.pt',weights_only=False)['tensors'];hf=torch.load('/workspace/logs/step0_continuation/official_generate_prefill_full/hf/step_0.pt',weights_only=False)['tensors']
def metric(x,y):
 x=x.reshape(-1).float();y=y.to(x.device).reshape(-1).float();d=(x-y).abs();return {'different':int((d!=0).sum()),'max':float(d.max()),'mean':float(d.mean()),'total':d.numel()}
report={'input_norm':metric(a['L0.full.input_norm.out0'],hf['L0.full.input_norm.out0']),'attn':metric(a['L0.full.post_norm.in0'],hf['L0.full.attn.out0']),'post_norm':metric(a['L0.full.post_norm.out0'],hf['L0.full.post_norm.out0']),'moe_output':metric(a['L0.full.mlp.out0'],hf['L0.full.mlp.out0'])}
ids=a['L0.full.routing.ids'];rw=a['L0.full.routing.weights'];hi=hf['L0.full.router.out2'].reshape_as(ids);hw=hf['L0.full.router.out1'].reshape_as(rw)
report['routing_ids']=metric(ids,hi);report['routing_weights']=metric(rw,hw);coords=((ids!=hi)|(rw!=hw)).nonzero();report['first_routing_differences']=[{'token':t,'slot':s,'actual_id':int(ids[t,s]),'hf_id':int(hi[t,s]),'actual_weight':float(rw[t,s]),'hf_weight':float(hw[t,s])} for t,s in coords[:10].tolist()]
x=a['L0.full.routing.input'].cuda();w=a['L0.router.weight'].cuda();gl=tgemm.mm(x,w,otype=torch.bfloat16);report['router_gemm_replay_vs_hf']=metric(gl,hf['L0.full.router.out0']);report['router_gemm_replay_last_vs_actual']=metric(gl[-1:],a['L0.router.out0']);report['router_gemm_config']=get_GEMM_A16W16_config(990,64,2304,False,'torch.bfloat16','torch.bfloat16');report['ck_stage1_shape']=list(a['L0.full.moe.stage1'].shape)
(r/'l0_first_moe_boundary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

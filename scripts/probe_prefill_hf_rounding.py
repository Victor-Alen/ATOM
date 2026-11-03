import json,torch,importlib.util
from pathlib import Path
root=Path('/workspace/logs/step0_continuation/qkv_actual_rope_prefill');a=torch.load(root/'atom/step_0.pt',weights_only=False)['tensors']
spec=importlib.util.spec_from_file_location('candidate','/workspace/ATOM-tritonmoe/atom/model_ops/mellum_hf_attention.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
q=a['L0.step0.attn.afterquery'].cuda().reshape(-1,32,128);k=a['L0.step0.attn.afterkey'].cuda().reshape(-1,4,128);v=a['L0.step0.attn.invalue'].cuda().reshape(-1,4,128);n=q.shape[0]
def metric(x,y):
 d=(x.float()-y.float()).abs();return {'max':float(d.max()),'mean':float(d.mean()),'different':int((d!=0).sum()),'total':d.numel()}
with torch.inference_mode():
 out,p,s=m.dense_prefill_hf_rounding(q,k,v,128**-0.5,return_intermediates=True)
 qq=q.transpose(0,1);kk=k.transpose(0,1).repeat_interleave(8,0);vv=v.transpose(0,1).repeat_interleave(8,0)
 score=(qq@kk.transpose(-1,-2))*(128**-0.5);mask=torch.arange(n,device='cuda')[:,None]>=torch.arange(n,device='cuda')[None,:]
 masked=score.masked_fill(~mask,float('-inf'));probs=masked.float().softmax(-1).bfloat16();ref=(probs@vv).transpose(0,1)
 cp=p[:,:,:8,:n].permute(1,2,0,3).reshape(32,n,n);cs=s[:,:,:8,:n].permute(1,2,0,3).reshape(32,n,n)
 coords=(out!=ref).nonzero();result={'actual_rope_class':a['rope.class_file'],'actual_rope_flag':a['rope.hf_inv_freq_flag'],'baseline_vs_hf':metric(a['L0.step0.o_proj.in0'].cuda().reshape_as(ref),ref),'new_vs_hf':metric(out,ref),'scores_vs_hf':metric(cs[:,mask],score[:,mask]),'probabilities_vs_hf':metric(cp,probs),'first_output_differences':coords[:10].tolist()}
 (root/'prefill_candidate.json').write_text(json.dumps(result,indent=2)+'\n');torch.save({'scores':cs.cpu(),'ref_scores':score.cpu(),'p':cp.cpu(),'ref_p':probs.cpu(),'out':out.cpu(),'ref':ref.cpu()},root/'prefill_candidate.pt');print(json.dumps(result,indent=2))

import json,torch,importlib.util
from pathlib import Path
from aiter.ops.triton.gluon.pa_decode_gluon import pa_decode_gluon
s=importlib.util.spec_from_file_location('hfround','/workspace/ATOM-tritonmoe/atom/model_ops/mellum_hf_attention.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
root=Path('/workspace/logs/step0_continuation/kv_isolation');a=torch.load(root/'fixed_attention.pt',weights_only=False)
q=a['q'].cuda().reshape(32,1,128).transpose(0,1).contiguous();k=a['k_raw'].cuda();v=a['v_raw'].cuda();n=a['k'].shape[0];bt=torch.arange(k.shape[0],device='cuda',dtype=torch.int32)[None,:];lens=torch.tensor([n],device='cuda',dtype=torch.int32)
def metric(x,y):
 d=(x.float()-y.to(x.device).reshape(x.shape).float()).abs();return {'max':float(d.max()),'mean':float(d.mean()),'different':int((d!=0).sum()),'total':d.numel()}
with torch.inference_mode():
 out,p,logits=m.paged_decode_hf_rounding(q,k,v,bt,lens,128**-0.5,return_intermediates=True)
 kk=a['k'].cuda().transpose(0,1).repeat_interleave(8,0);vv=a['v'].cuda().transpose(0,1).repeat_interleave(8,0)
 hq=q[0,:,None,:];scores=(hq@kk.transpose(-1,-2))*(128**-0.5);probs=scores.float().softmax(-1).bfloat16();ref=(probs@vv).reshape(1,32,128)
 ps=p[0,:,:8,:n].reshape(32,1,n);ls=logits[0,:,:8,:n].reshape(32,1,n)
 baseline=torch.empty_like(q)
 def base():
  pa_decode_gluon(baseline,q,k,v,lens,bt,128**-0.5,1,1,128,torch.bfloat16,sliding_window=1024,ps=True)
 base()
 report={'baseline_vs_captured':metric(baseline,a['actual']),'reference_recreated':metric(ref,a['ref']),'baseline_vs_hf':metric(baseline,ref),'new_vs_hf':metric(out,ref),'scores_vs_hf':metric(ls,scores),'probabilities_vs_hf':metric(ps,probs)}
 for name,fun in [('baseline',base),('new',lambda:m.paged_decode_hf_rounding(q,k,v,bt,lens,128**-0.5))]:
  for _ in range(5):fun()
  beg=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);beg.record()
  for _ in range(100):fun()
  end.record();end.synchronize();report[name+'_ms']=beg.elapsed_time(end)/100
 torch.save({'out':out.cpu(),'p':ps.cpu(),'scores':ls.cpu(),'ref_scores':scores.cpu(),'ref_p':probs.cpu()},root/'candidate_attention.pt')
 (root/'candidate_attention.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

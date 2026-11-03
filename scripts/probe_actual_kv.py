import torch,json,os
from pathlib import Path
root=Path(os.environ.get('MELLUM_TRACE_ROOT','/workspace/logs/step0_continuation/kv_isolation'));target=int(os.environ.get('MELLUM_TRACE_STEP','3'))
a=torch.load(root/'atom'/f'step_{target}.pt',map_location='cpu',weights_only=False)['tensors']
def metric(x,y):
 d=(x.float()-y.to(x.device).float()).abs();return {'max':float(d.max()),'mean':float(d.mean()),'different':int((d!=0).sum()),'total':d.numel()}
k=a['kv.k_raw'];v=a['kv.v_raw'];n=int(a['kv.context_lens'][0]);print('CACHE',k.shape,v.shape,a['kv.format'],a['kv.use_flash_layout'],n,flush=True)
nb,nh,ds,bs,x=k.shape
kd=k.permute(0,3,1,2,4).reshape(-1,nh,ds*x)[:n]
assert v.ndim==4 and v.shape[-1]==bs
vd=v.permute(0,3,1,2).reshape(-1,nh,ds*x)[:n]
expectedk=torch.cat([a[f'L0.step{s}.attn.afterkey'].reshape(-1,4,128) for s in range(target+1)])
expectedv=torch.cat([a[f'L0.step{s}.qkv_proj.out0'][...,4608:].reshape(-1,4,128) for s in range(target+1)])
r={'n':n,'k_shape':list(k.shape),'v_shape':list(v.shape),'format':a['kv.format'],'k_cache_vs_rotated_history':metric(kd,expectedk),'v_cache_vs_projected_history':metric(vd,expectedv)}
q=a['kv.q'].reshape(1,32,128).transpose(0,1).unsqueeze(0).cuda();kk=kd.transpose(0,1).unsqueeze(0).repeat_interleave(8,dim=1).cuda();vv=vd.transpose(0,1).unsqueeze(0).repeat_interleave(8,dim=1).cuda()
f32qk=q.float()@kk.float().transpose(-1,-2)
bf16qk=q@kk.transpose(-1,-2)
refp=(bf16qk*(128**-0.5)).softmax(-1,dtype=torch.float32).bfloat16();ref=(refp@vv).transpose(1,2).reshape(1,4096)
actual=a[f'L0.step{target}.o_proj.in0'].cuda()
variants={}
for name,logits in [('hf_qk_scale',bf16qk*(128**-0.5)),('f32qk_scale',f32qk*(128**-0.5)),('qk_bf16_scale_f32',bf16qk.float()*(128**-0.5))]:
 p=logits.float().softmax(-1)
 for pvname,pv in [('normalized_bf16',(p.bfloat16()@vv)),('normalized_f32',(p@vv.float()).bfloat16())]:
  out=pv.transpose(1,2).reshape(1,4096)
  variants[name+'/'+pvname]={'vs_actual':metric(out,actual),'vs_hf':metric(out,ref)}
r['variants']=variants;r['hf_vs_actual']=metric(ref,actual)
(root/'cache_and_rounding.json').write_text(json.dumps(r,indent=2)+'\n');torch.save({'q':q.cpu(),'k':kd,'v':vd,'k_raw':k,'v_raw':v,'ref':ref.cpu(),'actual':actual.cpu()},root/'fixed_attention.pt');print(json.dumps(r,indent=2))

import json,torch,sys
from pathlib import Path
from aiter.ops.triton.fused_kv_cache import fused_qk_rope_reshape_and_cache
from hf_reference_generate import load_mellum_classes,MODEL_PATH
Config,_=load_mellum_classes();hf=sys.modules['transformers.models.mellum.modeling_mellum'];cfg=Config.from_pretrained(MODEL_PATH)
root=Path('/workspace/logs/step0_continuation/qkv_split_prefill_rerun');a=torch.load(root/'atom/step_0.pt',weights_only=False)['tensors']
q=a['L0.step0.attn.inquery'].cuda().reshape(-1,32,128);k=a['L0.step0.attn.inkey'].cuda().reshape(-1,4,128);v=a['L0.step0.attn.invalue'].cuda().reshape(-1,4,128);pos=a['L0.step0.attn.inpositions'].cuda();n=q.shape[0]
def metric(x,y):
 d=(x.float()-y.to(x.device).reshape(x.shape).float()).abs();return {'max':float(d.max()),'mean':float(d.mean()),'different':int((d!=0).sum()),'total':d.numel()}
rope=hf.MellumRotaryEmbedding(cfg,device='cuda');cos,sin=rope(q,pos[None],cfg.layer_types[0]);hq,hk=hf.apply_rotary_pos_emb(q.transpose(0,1)[None],k.transpose(0,1)[None],cos,sin);hq=hq[0].transpose(0,1);hk=hk[0].transpose(0,1)
r={}
for device in ['cuda','cpu']:
 inv=1/(500000.0**(torch.arange(0,128,2,dtype=torch.float32,device=device)/128));freq=pos.float()[:,None]*inv.cuda()[None,:];cc=freq.cos().bfloat16().reshape(n,1,1,64);ss=freq.sin().bfloat16().reshape(n,1,1,64)
 kc=torch.empty(((n+15)//16,4,16,16,8),device='cuda',dtype=torch.bfloat16);vc=torch.empty(((n+15)//16,4,128,16),device='cuda',dtype=torch.bfloat16)
 out=fused_qk_rope_reshape_and_cache(q,k,v,kc,vc,torch.arange(n,device='cuda',dtype=torch.int64),pos,cc,ss,None,None,True,False,False,output_zeros=False)
 qo,ko=out[:2]
 r[device]={'q_vs_captured':metric(qo,a['L0.step0.attn.afterquery']),'k_vs_captured':metric(ko,a['L0.step0.attn.afterkey']),'q_vs_hf':metric(qo,hq),'k_vs_hf':metric(ko,hk),'cos_vs_hf':metric(cc.reshape(n,64),cos[0,:,:64]),'sin_vs_hf':metric(ss.reshape(n,64),sin[0,:,:64])}
(root/'rope_inv_freq.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))

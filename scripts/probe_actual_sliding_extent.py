import json
from pathlib import Path
from types import SimpleNamespace
import torch
from atom.model_ops.mellum_bf16_attention import attention
from atom.model_ops.mellum_hf_rope import rotate
import sys
sys.path.insert(0,'/workspace/ATOM-general-norm/scripts')
from hf_reference_generate import MODEL_PATH,load_mellum_classes
Config,_=load_mellum_classes();cfg=Config.from_pretrained(MODEL_PATH)
root=Path('/workspace/logs/step0_causal_boundary/current_A0002_step3')
a=torch.load(root/'atom/step_3.pt',map_location='cpu',weights_only=True)['tensors']
h=torch.load(root/'hf/step_3.pt',map_location='cpu',weights_only=True)['tensors']
position=a['model.positions'].cuda().reshape(-1)
check_q,_=rotate(cfg,0,position,a['L0.step3.q_norm.out0'].cuda(),a['L0.step3.k_norm.out0'].cuda())
assert torch.equal(check_q.reshape_as(a['kv.q']),a['kv.q'].cuda())
q,_=rotate(cfg,0,position-1,a['L0.step2.q_norm.out0'].cuda(),a['L0.step2.k_norm.out0'].cuda())
q=q.reshape(1,32,128)
kc=a['kv.k_raw'].cuda();vc=a['kv.v_raw'].cuda()
context=int(a['kv.context_lens'].item())-1
metadata=SimpleNamespace(context_lens=torch.tensor([context],device='cuda'),cu_seqlens_q=torch.tensor([0,1],device='cuda'),block_tables=torch.arange(kc.shape[0],device='cuda')[None])
actual=attention(q,torch.empty(1,4,128,device='cuda',dtype=q.dtype),torch.empty(1,4,128,device='cuda',dtype=q.dtype),kc,vc,metadata,128**-0.5,1024,a['kv.format']).cpu()
expected=h['L0.step2.o_proj.in0'].reshape_as(actual)
old=a['L0.step2.o_proj.in0'].reshape_as(actual)
report={'context':context,'cache_format':a['kv.format'],'old_different':int((old!=expected).sum()),'new_different':int((actual!=expected).sum()),'max_new_error':float((actual.float()-expected.float()).abs().max())}
(root/'sliding_extent_probe.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
assert report['new_different']==0

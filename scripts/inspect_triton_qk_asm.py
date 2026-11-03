import torch,importlib.util
from pathlib import Path
r=Path('/workspace/logs/step0_continuation/qkv_actual_rope_prefill');a=torch.load(r/'atom/step_0.pt',weights_only=False)['tensors'];q=a['L0.step0.attn.afterquery'].cuda().reshape(-1,32,128);k=a['L0.step0.attn.afterkey'].cuda().reshape(-1,4,128);n=q.shape[0];lens=torch.arange(1,n+1,device='cuda',dtype=torch.int32);s=torch.empty((n,4,16,1024),device='cuda',dtype=torch.bfloat16)
spec=importlib.util.spec_from_file_location('candidate','/workspace/ATOM-tritonmoe/atom/model_ops/mellum_hf_attention.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
c=m._qk.warmup(q,k,lens,lens,s,*q.stride()[:2],*k.stride(),0,0,0,16,8,8,4,1024,128**-0.5,DENSE=True,grid=(n,4,8),num_warps=4,enable_fp_fusion=False)
asm=c.asm['amdgcn'];(r/'triton_qk.s').write_text(asm);print('\n'.join(x for x in asm.splitlines() if 'mfma' in x)[:2500])

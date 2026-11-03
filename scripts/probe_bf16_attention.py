import json
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from atom.model_ops.mellum_bf16_attention import attention
sys.path.insert(0,'/workspace/ATOM-general-norm/scripts')
from hf_reference_generate import load_mellum_classes
load_mellum_classes()
hf=sys.modules['transformers.models.mellum.modeling_mellum']
torch.manual_seed(1729)
reports=[]
for n in [17,65,990,1574]:
 q=torch.randn(n,32,128,device='cuda',dtype=torch.bfloat16)
 k=torch.randn(n,4,128,device='cuda',dtype=torch.bfloat16)
 v=torch.randn_like(k)
 nb=(n+15)//16
 kp=torch.zeros(nb*16,4,128,device='cuda',dtype=k.dtype);vp=torch.zeros_like(kp)
 kp[:n]=k;vp[:n]=v
 logical_k=kp.view(nb,16,4,16,8).permute(0,2,3,1,4).contiguous()
 logical_v=vp.view(nb,16,4,128).permute(0,2,3,1).contiguous()
 permutation=torch.randperm(nb,device='cuda')
 kc=torch.empty_like(logical_k);vc=torch.empty_like(logical_v)
 kc[permutation]=logical_k;vc[permutation]=logical_v
 for count in [n,1]:
  for window in [-1,1024]:
   qq=q[-count:]
   qi=torch.arange(n-count,n,device='cuda')[:,None];ki=torch.arange(n,device='cuda')[None,:]
   allowed=ki<=qi
   if window>0:allowed=allowed & (ki>qi-window)
   mask=torch.zeros(count,n,device='cuda',dtype=q.dtype).masked_fill(~allowed,torch.finfo(q.dtype).min)[None,None]
   reference,_=hf.eager_attention_forward(SimpleNamespace(num_key_value_groups=8,training=False),qq.transpose(0,1)[None],k.transpose(0,1)[None],v.transpose(0,1)[None],mask,128**-0.5)
   metadata=SimpleNamespace(context_lens=torch.tensor([n],device='cuda'),cu_seqlens_q=torch.tensor([0,count],device='cuda'),block_tables=permutation[None])
   shuffled=vp.view(nb,2,8,4,128).permute(0,3,1,4,2).contiguous().view(nb,4,128,16)
   vcs=torch.empty_like(shuffled);vcs[permutation]=shuffled
   for cache_format,cache in [('NHD',vc),('SHUFFLE',vcs)]:
    actual=attention(qq,k[-count:],v[-count:],kc,cache,metadata,128**-0.5,window,cache_format)
    delta=(actual.float()-reference.reshape_as(actual).float()).abs()
    row={'n':n,'query_count':count,'window':window,'cache_format':cache_format,'different':int((delta!=0).sum()),'max':float(delta.max())}
    reports.append(row);print(json.dumps(row),flush=True)
Path('/workspace/logs/step0_causal_boundary/bf16_attention_probe.json').write_text(json.dumps(reports,indent=2))
assert not any(r['different'] for r in reports)
print('BF16_ATTENTION_EXACT_ALL_CASES',flush=True)

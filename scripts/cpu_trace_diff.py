import argparse
import json
from pathlib import Path
import torch
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--step',type=int,required=True);args=ap.parse_args();root=Path(args.root)
a=torch.load(root/'atom'/f'step_{args.step}.pt',map_location='cpu',weights_only=True)['tensors']
h=torch.load(root/'hf'/f'step_{args.step}.pt',map_location='cpu',weights_only=True)['tensors']
def metric(x,y):
 x=x.reshape(-1).float();y=y.reshape(-1).float();assert x.shape==y.shape
 d=(x-y).abs();return {'different':int((d!=0).sum()),'total':d.numel(),'max':float(d.max()),'mean':float(d.mean())}
report={'input':metric(a['model.input_ids'],h['model.input_ids']),'positions':metric(a['model.positions'],h['model.positions']),'layers':[]}
for i in range(28):
 row={'layer':i}
 for tag in ['input_norm','attn','post_norm','mlp','router']:
  row[tag]=metric(a[f'L{i}.{tag}.out0'],h[f'L{i}.{tag}.out0'])
 report['layers'].append(row);print(json.dumps(row),flush=True)
report['final_hidden']=metric(a['head.input'],h['final_norm.out0']);report['logits']=metric(a['head.logits'],h['head.logits'])
report['hf_token']=int(h['head.logits'].reshape(-1).argmax());report['atom_token']=int(a['head.logits'].reshape(-1).argmax())
(root/'cpu_drift.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='layers'}),flush=True)

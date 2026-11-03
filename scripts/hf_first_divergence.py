#!/usr/bin/env python3
"""Diagnostic manual cached loop; not the Step0 generate protocol.
Use hf_generate_step_trace.py for validated Step0 traces."""
import importlib.util
import os
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from hf_reference_generate import MODEL_PATH,load_mellum_classes,load_checkpoint_experts

ROOT=Path(os.environ.get('MELLUM_TRACE_ROOT','/workspace/logs/step0_continuation/first_divergence'))
TARGET=int(os.environ.get('MELLUM_TRACE_STEP','27'))
torch.set_num_threads(8)
spec=importlib.util.spec_from_file_location('standalone_step_trace','/workspace/ATOM-tritonmoe/atom/models/step_tensor_trace.py')
trace=importlib.util.module_from_spec(spec);spec.loader.exec_module(trace)

def metric(a,b):
    a=a.detach().cpu().reshape(-1).float();b=b.detach().cpu().reshape(-1).float()
    assert a.shape==b.shape,(a.shape,b.shape)
    d=(a-b).abs()
    return {'max':float(d.max()),'mean':float(d.mean()),'different':int((d!=0).sum()),'total':d.numel()}
def logits_info(x):
    x=x.detach().reshape(-1).float().cpu();v,i=x.topk(10)
    return {'argmax':int(x.argmax()),'margin':float(v[0]-v[1]),
        'top10_ids':i.tolist(),'top10_values':v.tolist(),
        'candidate_1234':float(x[1234]),'candidate_8411':float(x[8411])}

def main():
    rec=next(json.loads(s) for s in Path('/workspace/correctness/hf_scenario_a_20_eager.jsonl').read_text().splitlines() if json.loads(s)['id']=='A-0000')
    Config,LM=load_mellum_classes();cfg=Config.from_pretrained(MODEL_PATH)
    model=LM.from_pretrained(MODEL_PATH,config=cfg,dtype=torch.bfloat16,attn_implementation='eager')
    load_checkpoint_experts(model,MODEL_PATH);model=model.cuda().eval()
    print('HF_READY',flush=True)
    recorder=trace.TensorRecorder(ROOT/'hf',TARGET);recorder.attach(model.model)
    prompt=torch.tensor([rec['prompt_token_ids']],device='cuda',dtype=torch.long)
    predictions=[];cache=None;out=None
    with torch.inference_mode():
        for step in range(TARGET+1):
            recorder.begin(step)
            ids=prompt if step==0 else torch.tensor([[rec['token_ids'][step-1]]],device='cuda',dtype=torch.long)
            recorder.add('model.input_ids',ids)
            if step>0: recorder.add('model.positions',torch.tensor([prompt.shape[1]+step-1],device='cuda'))
            mask=torch.ones((1,prompt.shape[1]+step),device='cuda',dtype=torch.long)
            out=model(input_ids=ids,attention_mask=mask,past_key_values=cache,use_cache=True,logits_to_keep=1)
            cache=out.past_key_values
            pred=int(out.logits[0,-1].argmax()); predictions.append(pred)
            print(f'HF_STEP {step} argmax={pred} saved={rec["token_ids"][step]}',flush=True)
        recorder.add('head.logits',out.logits)
        recorder.save();recorder.active=False
        assert predictions==rec['token_ids'][:TARGET+1], 'HF teacher-forced protocol does not match frozen reference'
        (ROOT/'hf_capture.json').write_text(json.dumps({
            'prompt':'A-0000','step':TARGET,'hf_prefix_reproduced':True,
            'hf_predictions':predictions},indent=2)+'\n')

if __name__=='__main__':main()

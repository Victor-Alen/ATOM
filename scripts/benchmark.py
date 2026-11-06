"""Run the fixed public A/B/C workloads; reductions are explicitly screening."""
import argparse,datetime,hashlib,json,os,shlex,subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default=os.getenv('SERVER_URL','http://127.0.0.1:8000/v1/completions'))
    p.add_argument('--model',default=os.getenv('MODEL_NAME'))
    p.add_argument('--tokenizer-path',default=os.getenv('TOKENIZER_PATH','/data/models/Mellum2.1-12B-A2.5B-Thinking'))
    p.add_argument('--client',default=os.getenv('EVALSCOPE_BIN','/workspace/bench-venv/bin/evalscope'))
    p.add_argument('--scenarios',default='ABC');p.add_argument('--repeat',type=int,default=3)
    p.add_argument('--mtp',choices=('on','off'),default=None)
    p.add_argument('--screen',action='store_true',help='A64/B8/C12 only; never a full evaluation')
    p.add_argument('--a-source',choices=('all','filtered'),default='all')
    p.add_argument('--out',default=None);p.add_argument('--seed',type=int,default=20261105)
    args=p.parse_args()
    if args.repeat<1 or any(x not in 'ABC' for x in args.scenarios):p.error('Invalid repeat/scenarios')
    if args.model is None:
        import urllib.request
        with urllib.request.urlopen(args.url.rsplit('/v1/',1)[0]+'/v1/models') as response:args.model=json.load(response)['data'][0]['id']
    root=Path(args.out or ('results/'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    root.mkdir(parents=True,exist_ok=False)
    source=Path('/data/datasets/scenario_a_prompts.jsonl')
    records=[json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    prompts=[r['prompt'] for r in records]
    if args.a_source=='filtered':
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(args.tokenizer_path)
        prompts=[s for s in prompts if 512<=len(tok.encode(s))<=4096]
    derived=root/'scenario_a_prompt_bodies.jsonl'
    derived.write_text('\n'.join(json.dumps({'prompt':s},ensure_ascii=False) for s in prompts)+'\n')
    manifest=dict(scope='SCREENING' if args.screen else 'FULL_PUBLIC',num_gpus=1,counts=dict(A=64,B=8,C=12) if args.screen else dict(A=512,B=50,C=200),
        a_source=args.a_source,a_original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),a_rows=len(prompts),
        adaptation='EvalScope 1.14.1 requires a registered dataset name; line_by_line forwards unchanged prompt fields. all preserves all 512 source rows; filtered is a separate supplementary workload.',
        seed=args.seed,mtp=None if args.mtp is None else args.mtp=='on')
    (root/'protocol.json').write_text(json.dumps(manifest,indent=2))
    for repeat in range(args.repeat):
        for scene in args.scenarios:
            out=root/f'r{repeat}'/scene;out.mkdir(parents=True)
            cmd=[args.client,'perf','--url',args.url,'--api','openai','--model',args.model,'--tokenizer-path',args.tokenizer_path,
                 '--number',str(manifest['counts'][scene]),'--max-tokens','256','--min-tokens','256','--stream',
                 '--extra-args',json.dumps({'ignore_eos':True,'temperature':0.0}),'--name',f'Scenario_{scene}_N1_r{repeat}',
                 '--outputs-dir',str(out),'--seed',str(args.seed)]
            if scene=='A':cmd+=['--parallel','32','--dataset','line_by_line','--dataset-path',str(derived),'--min-prompt-length','512','--max-prompt-length','4096']
            elif scene=='B':cmd+=['--parallel','1','--dataset','random','--min-prompt-length','1024','--max-prompt-length','1024']
            else:cmd+=['--open-loop','--rate','0.30','--dataset','line_by_line','--dataset-path','/data/datasets/scenario_c_prompts.jsonl','--total-timeout','180']
            (out/'command.json').write_text(json.dumps(cmd,indent=2));(out/'command.sh').write_text(shlex.join(cmd)+'\n')
            print('RUN',repeat,scene,manifest['scope'],flush=True)
            with (out/'client.log').open('w') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            (out/'exit_code').write_text(str(result.returncode))
            if result.returncode:raise RuntimeError(f'Workload failed; inspect {out}/client.log')
    print('DONE',root,flush=True)

if __name__=='__main__':main()

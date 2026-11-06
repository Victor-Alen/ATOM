"""Replay pinned public HF references through the ATOM HTTP endpoint only."""
import argparse,asyncio,datetime,gzip,hashlib,json,os,re,subprocess,sys,tempfile,time
from pathlib import Path
import aiohttp

ROOT=Path(__file__).resolve().parents[1]
REF=ROOT/'validation/references'

def read(name):
    path=REF/name
    if path.suffix=='.gz':
        with gzip.open(path,'rt') as f:text=f.read()
    else:text=path.read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()] if '.jsonl' in name else json.loads(text)

def compare(got,expected):
    return dict(matched=sum(a==b for a,b in zip(got,expected)),total=len(expected),returned=len(got),exact=got==expected,complete=len(got)==len(expected))

def sanitize(text):
    if '</think>' in text:text=text.rsplit('</think>',1)[1]
    elif '<think>' in text:return ''
    blocks=re.findall(r'```(?:python|py)?[ \t]*\n(.*?)```',text,re.S|re.I)
    return (blocks[0] if blocks else text).strip()+'\n'

def check_program(task,code):
    with tempfile.TemporaryDirectory(prefix='csig_he_') as folder:
        p=Path(folder)/'input.json';p.write_text(json.dumps(dict(task=task,code=code)))
        env=os.environ.copy();env.update(HIP_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        result=subprocess.run([sys.executable,str(ROOT/'validation/he_worker.py'),str(p)],capture_output=True,text=True,timeout=30,env=env)
        if result.returncode:raise RuntimeError(result.stderr[-1000:])
        return json.loads(result.stdout)

def verify_references():
    manifest=json.loads((REF/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        assert hashlib.sha256((REF/name).read_bytes()).hexdigest()==digest,name
    # Negative controls make length errors and a one-token mismatch observable.
    assert compare([1,2],[1,2])['exact']
    assert not compare([1],[1,2])['complete']
    assert compare([1,3],[1,2])['matched']==1
    assert not compare([1,2,3],[1,2])['complete']
    return manifest

async def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8000/v1/completions')
    parser.add_argument('--model',default=None)
    parser.add_argument('--mtp',choices=('on','off'),required=False,help='Runtime annotation; caller must match the server configuration')
    parser.add_argument('--tasks',default='/data/datasets/HumanEval.jsonl')
    parser.add_argument('--concurrency',type=int,default=8)
    parser.add_argument('--screen',action='store_true',help='C4 + strict16k only, not full acceptance')
    parser.add_argument('--verify-files-only',action='store_true')
    parser.add_argument('--out')
    args=parser.parse_args();manifest=verify_references()
    if args.verify_files_only:print('Pinned references and comparator negative controls verified. No GPU validation performed.');return
    if args.model is None:
        import urllib.request
        with urllib.request.urlopen(args.url.rsplit('/v1/',1)[0]+'/v1/models') as response:args.model=json.load(response)['data'][0]['id']
    out=Path(args.out or ROOT/'results'/('public-validation-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    out.mkdir(parents=True,exist_ok=False)
    (out/'protocol.json').write_text(json.dumps(dict(arguments=vars(args),reference_manifest=manifest,scope='SCREENING' if args.screen else 'FULL_PUBLIC_CORRECTNESS'),indent=2))
    semaphore=asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=7200),connector=aiohttp.TCPConnector(force_close=True)) as session:
        async def call(prompt,n,ignore=True):
            async with semaphore:
                start=time.monotonic()
                async with session.post(args.url,json=dict(model=args.model,prompt=prompt,max_tokens=n,temperature=0.,ignore_eos=ignore,stream=False)) as response:
                    response.raise_for_status();data=await response.json()
                assert isinstance(data.get('token_ids'),list),'ATOM token_ids extension required'
                return data,time.monotonic()-start
        strict=read('strict_16k_hf.json');body,seconds=await call(strict['prompt'],64)
        strict_result=dict(compare(body['token_ids'],strict['token_ids']),input_tokens=body['usage']['prompt_tokens'],seconds=seconds,response=body)
        (out/'strict_16k.json').write_text(json.dumps(strict_result,indent=2))
        refs=read('hf_scenario_c_200_eager.jsonl.gz')
        if args.screen:refs=[refs[i] for i in (0,49,100,199)]
        async def one(row):
            body,seconds=await call(row['prompt'],64)
            result=dict(id=row['id'],**compare(body['token_ids'],row['token_ids'][:64]),seconds=seconds,response=body)
            with (out/'C.responses.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
            return result
        rows=await asyncio.gather(*(one(row) for row in refs))
        summary=dict(requests=len(rows),matched=sum(x['matched'] for x in rows),total=len(rows)*64,exact_requests=sum(x['exact'] for x in rows),complete=all(x['complete'] for x in rows))
        summary['agreement']=summary['matched']/summary['total'];(out/'C.summary.json').write_text(json.dumps(summary,indent=2))
        passed=summary['complete'] and summary['agreement']>=.9 and strict_result['exact'] and strict_result['input_tokens']==16384
        he_summary=None
        if not args.screen:
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(args.model)
            taskpath=Path(args.tasks)
            assert hashlib.sha256(taskpath.read_bytes()).hexdigest()==manifest['humaneval_dataset_sha256'],'HumanEval dataset differs from pinned protocol'
            tasks=[json.loads(x) for x in taskpath.read_text().splitlines() if x.strip()]
            inputs={x['task_id']:x for x in read('he_inputs.jsonl')};hf={x['task_id']:x for x in read('he_hf.jsonl')}
            async def he(task):
                inp=inputs[task['task_id']];body,seconds=await call(inp['prompt'],2048,False)
                assert body['usage']['prompt_tokens']==len(inp['prompt_token_ids'])
                ids=body['token_ids'];text=tokenizer.decode(ids,skip_special_tokens=True);code=sanitize(text)
                evaluation=await asyncio.to_thread(check_program,task,code)
                row=dict(task_id=task['task_id'],token_ids=ids,text=text,code=code,passed=evaluation['passed'],result=evaluation['result'],exact_HF=ids==hf[task['task_id']]['token_ids'],seconds=seconds)
                with (out/'he_atom.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
                print('HumanEval',task['task_id'],evaluation['passed'],flush=True)
                return row
            generated=await asyncio.gather(*(he(task) for task in tasks))
            count=len(generated);hits=sum(x['passed'] for x in generated);reference=sum(x['passed'] for x in hf.values())
            he_summary=dict(count=count,passed=hits,pass_at_1=hits/count,HF_passed=reference,gap_pp=abs(hits-reference)/count*100,exact_HF_sequences=sum(x['exact_HF'] for x in generated))
            passed=passed and count==164 and he_summary['gap_pp']<=2
            (out/'he_summary.json').write_text(json.dumps(he_summary,indent=2))
        acceptance=dict(passed=passed,scope='SCREENING' if args.screen else 'FULL_PUBLIC_CORRECTNESS',C=summary,strict16k_exact=strict_result['exact'],HumanEval=he_summary,MTP=None if args.mtp is None else args.mtp=='on',num_gpus=1)
        (out/'acceptance.json').write_text(json.dumps(acceptance,indent=2));print(json.dumps(acceptance,indent=2))
        if not passed:raise RuntimeError('Correctness gate failed; inspect saved responses')

if __name__=='__main__':asyncio.run(main())

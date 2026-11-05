"""Compare three fresh HF eager 16K references, sequentially and concurrently."""
import argparse, asyncio, hashlib, json, time
from pathlib import Path
import aiohttp

async def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--url',default='http://127.0.0.1:8000/v1/completions')
    parser.add_argument('--model',default='/data/models/Mellum2.1-12B-A2.5B-Thinking')
    args=parser.parse_args();ref=Path(args.references);out=Path(args.out)
    manifest=json.loads((ref/'manifest.json').read_text())
    assert hashlib.sha256((ref/'references.jsonl').read_bytes()).hexdigest()==manifest['reference_sha256']
    assert all(row['exact_historical'] for row in manifest['anchors'])
    rows=[json.loads(line) for line in (ref/'references.jsonl').read_text().splitlines()]
    assert len(rows)==3 and len({row['id'] for row in rows})==3
    assert all(row['prompt_tokens']==16384 and len(row['token_ids'])==64 for row in rows)
    out.mkdir(parents=True,exist_ok=False)
    (out/'protocol.json').write_text(json.dumps(dict(arguments=vars(args),reference_manifest=manifest,scope='SUPPLEMENTAL_CORRECTNESS; 3 unique prompts, 6 requests, not performance'),indent=2))
    results=[]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as session:
        async def call(row,phase):
            start=time.monotonic()
            async with session.post(args.url,json=dict(model=args.model,prompt=row['prompt'],max_tokens=64,temperature=0.,ignore_eos=True,stream=False)) as response:
                response.raise_for_status();body=await response.json()
            got=body['token_ids'];expected=row['token_ids']
            result=dict(id=row['id'],phase=phase,input_tokens=body['usage']['prompt_tokens'],returned=len(got),matched=sum(a==b for a,b in zip(got,expected)),exact=got==expected,seconds=time.monotonic()-start,response=body)
            with (out/'responses.jsonl').open('a') as file:file.write(json.dumps(result)+'\n')
            return result
        for row in rows:results.append(await call(row,'sequential'))
        results.extend(await asyncio.gather(*(call(row,'concurrency3') for row in rows)))
    passed=all(row['exact'] and row['input_tokens']==16384 and row['returned']==64 for row in results)
    summary=dict(passed=passed,unique_prompts=3,requests=6,total_tokens=384,matched_tokens=sum(row['matched'] for row in results),all_exact=all(row['exact'] for row in results),phases=['sequential','concurrency3'],reference_sha256=manifest['reference_sha256'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
    assert passed,'Additional long-context comparison failed'

if __name__=='__main__':asyncio.run(main())

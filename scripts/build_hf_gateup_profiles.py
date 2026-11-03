from pathlib import Path
import json,re,subprocess,hashlib,torch
r=Path('/workspace/logs/step0_continuation/ck_hf_profile_prod');p=r/'heuristic32768.tsv'
with p.open('w') as out:subprocess.run(['/tmp/query_hf_gateup_profiles','32768'],stdout=out,check=True)
entries={}
for line in p.read_text().splitlines():
 m,name,ws=line.split('\t');entries[int(m)]=name
iso=torch.load('/workspace/logs/step0_continuation/norm990_active_prod/ck_stage_isolation.pt',weights_only=False);active=iso['ids'].flatten().bincount(minlength=64);prof=json.loads(Path('/workspace/logs/step0_continuation/norm990_active_prod/hf_grouped_gateup_profile_summary.json').read_text())['kernels'];diff=[]
for e,k in zip(active.nonzero().flatten().tolist(),prof):
 m=int(active[e]);
 if entries[m]!=k['name']:diff.append({'e':e,'m':m,'actual':k['name'],'heuristic':entries[m]})
assert not diff,diff
flags=[0]+[int('_MI32x32x1_' in entries[m]) for m in range(1,32769)];ranges=[]
for m in range(1,32769):
 if ranges and entries[m]==ranges[-1]['kernel']:ranges[-1]['end']=m
 else:ranges.append({'start':m,'end':m,'kernel':entries[m]})
lib=Path('/opt/rocm/lib/libhipblaslt.so').resolve();data={'format':1,'n':1792,'k':2304,'dtype':'bf16','arch':'gfx942','max_tokens':32768,'workspace_bytes':33554432,'library':str(lib),'library_sha256':hashlib.sha256(lib.read_bytes()).hexdigest(),'validated_captured_experts':51,'mi32_flags':flags,'ranges':ranges}
dst=Path('/workspace/ATOM-tritonmoe/atom/model_ops/mellum_hf_gateup_profiles.json');dst.write_text(json.dumps(data,separators=(',',':'))+'\n');print('VERIFIED51','RANGES',len(ranges),'TABLE',dst,'LIB',lib,flush=True)

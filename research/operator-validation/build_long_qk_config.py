"""Convert exhaustive frozen-library shape heuristics to a guarded QK profile.

Only run after collecting query_hf_qk_profiles output on the allocated idle GPU.
No public prompt list is used. Correctness/shape-class tests are separate gates.
"""
from pathlib import Path
import json,re,hashlib
import torch

r=Path('/workspace/logs/step0_operator_parallel_20261104')
table=r/'startup_delivery/hf_qk_all_lengths.tsv'
old=json.loads((r/'native_sparse_qk_config.json').read_text())
profiles=dict(old['profiles']);classes={};unsupported=[];parsed=[]
for line in table.read_text().splitlines():
    n,name,workspace=line.split('\t');n=int(n);parsed.append(n)
    if n<=4096:continue
    match=re.search(r'_MT(\d+)x(\d+)x(\d+)_MI(16|32)x',name)
    if not match or 'GSUAMBSK' not in name:
        unsupported.append(n);continue
    tile,depth,mi=int(match[2]),int(match[3]),int(match[4])
    if tile%32 or depth not in (32,64,128):
        unsupported.append(n);continue
    # MI16/depth32 retains the original 4-wide BF16 packing. The longer
    # MI32/depth32 and MI16/depth64 kernels require 8-wide packing.
    profile=dict(row_tile=tile,depth=depth,mi=mi,kw=4 if mi==16 and depth==32 else 8)
    profiles[str(n)]=profile
    key=json.dumps(profile,sort_keys=True)
    classes.setdefault(key,[]).append(n)
assert parsed==list(range(1,16385)),(min(parsed),max(parsed),len(parsed))
out=dict(torch_version=torch.__version__,arch='gfx942',profiles=profiles,
    query_source='scripts/query_hf_qk_profiles.cpp',query_table_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),
    shapes='All sequence lengths 1..16384; Q=(N,32,128), K=(N,4,128), Q stride=(4096,128,1)',
    unsupported_lengths=unsupported,validation_status='pending GPU/operator/model validation')
(r/'startup_delivery/native_sparse_qk_long_config.json').write_text(json.dumps(out,indent=2))
brief={k:dict(count=len(v),first=v[:3],last=v[-3:]) for k,v in classes.items()}
(r/'startup_delivery/native_sparse_qk_long_classes.json').write_text(json.dumps(dict(classes=classes,summary=brief),indent=2))
print(json.dumps(dict(classes=brief,unsupported=len(unsupported),profiles=len(profiles)),indent=2))

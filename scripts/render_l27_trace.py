#!/usr/bin/env python3
"""Export the measured accumulator trace; never interprets HF prefixes as snapshots."""
import csv
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('/workspace/logs/step0_continuation/l27_accumulator_trace')
d=json.loads((root/'trace.json').read_text())
def bf16_round(x):
    if x==0:
        return x
    e=math.frexp(abs(x))[1]-1
    return math.copysign(math.ldexp(round(math.ldexp(abs(x),7-e)),e-7),x)
previous=0.0
for r in d['rows']:
    r['fp64_prefix_bf16']=bf16_round(r['dot_prefix_fp64'])
    r['corrected_error_vs_fp64']=r['corrected_running_fp32']-r['dot_prefix_fp64']
    r['chunk_dot_fp64']=r['dot_prefix_fp64']-previous
    previous=r['dot_prefix_fp64']
    r['independent_chunk_error_vs_fp64']=r['independent_chunk_fp32']-r['chunk_dot_fp64']
d['first_ck_bf16_vs_fp64_prefix_difference']=next((r['k_end'] for r in d['rows'] if r['ck_running_bf16']!=r['fp64_prefix_bf16']),None)
d['maximum_independent_chunk_error_vs_fp64']=max(abs(r['independent_chunk_error_vs_fp64']) for r in d['rows'])
d['final']=d['rows'][-1]
(root/'trace_enriched.json').write_text(json.dumps(d,indent=2)+'\n')
with (root/'trace.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(d['rows'][0]))
    writer.writeheader(); writer.writerows(d['rows'])
fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
xs=[r['k_end'] for r in d['rows']]
ax.plot(xs,[r['ck_error_vs_fp64']*1e6 for r in d['rows']],label='Original CK running FP32',lw=1.8)
ax.plot(xs,[r['corrected_error_vs_fp64']*1e6 for r in d['rows']],label='Independent FP32 chunks + FP64 merge',lw=1.8)
ax.axhline(0,color='black',lw=.7)
ax.set(xlabel='K elements included',ylabel='Accumulator error vs FP64 prefix (1e-6)',
       title='L27 up projection: token 467, expert 38, feature 132')
ax.grid(alpha=.2); ax.legend()
fig.savefig(root/'accumulator_error.png',dpi=170)
fig.savefig(root/'accumulator_error.svg')
print(json.dumps({k:v for k,v in d.items() if k!='rows'},indent=2))

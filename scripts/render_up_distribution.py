#!/usr/bin/env python3
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('/workspace/logs/step0_continuation')
base=root/'up_difference_distribution'
d=json.loads(base.with_suffix('.json').read_text())
rows=list(csv.DictReader(base.with_suffix('.csv').open()))
with (root/'up_difference_not_matching_fp64.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(r for r in rows if r['fp64_verdict']!='ck_matches_fp64')
payload=torch.load(base.with_suffix('.pt'),map_location='cpu',weights_only=False)
assert bool(torch.isfinite(payload['up']).all()) and bool(torch.isfinite(payload['hf_up']).all())
bitmask=payload['up'].contiguous().view(torch.int16)!=payload['hf_up'].contiguous().view(torch.int16)
d['bit_different']=int(bitmask.sum())
d['signed_zero_bit_differences']=int(((payload['up']==0)&(payload['hf_up']==0)&bitmask).sum())
d['active_experts']=len(set(payload['ids'].flatten().tolist()))
d['fixed_target_absent']=not any(int(r['token_position'])==467 and int(r['slot'])==3 and int(r['feature'])==132 for r in rows)
d['fp64_verdict_by_origin']={origin:dict(Counter(r['fp64_verdict'] for r in rows if r['origin']==origin)) for origin in ('persistent','introduced')}
d['maximum_ulp_samples']=sorted(rows,key=lambda r:int(r['ulp_distance']),reverse=True)[:10]
experts=d['by_expert'][:7]
d['top7_expert_count_share']=sum(r['count'] for r in experts)/d['different']
d['top7_expert_exposure_share']=sum(r['exposure'] for r in experts)/d['total']
one=[r for r in rows if int(r['ulp_distance'])==1]
near=np.array([abs(float(r['dot_fp64'])-(float(r['ck'])+float(r['hf']))/2)/float(r['abs_diff']) for r in one])
d['one_ulp_distance_to_boundary_in_bf16_spacing']={'median':float(np.median(near)),'max':float(near.max())}
(root/'up_difference_distribution_enriched.json').write_text(json.dumps(d,indent=2)+'\n')

fig,axs=plt.subplots(2,2,figsize=(13,8.5),layout='constrained')
ax=axs[0,0]
for cond,label,color in ((lambda r:int(r['ulp_distance'])==1,'1 BF16 ULP','#377eb8'),(lambda r:int(r['ulp_distance'])>1,'>1 BF16 ULP','#e41a1c')):
    subset=[r for r in rows if cond(r)]
    ax.scatter([int(r['token_position']) for r in subset],[int(r['feature']) for r in subset],s=10,alpha=.6,label=label,c=color)
ax.set(xlabel='Input token position',ylabel='Up feature index',title='Residual locations (slots combined)',xlim=(-5,995),ylim=(-5,900))
ax.legend(loc='upper right',fontsize=8)
ax=axs[0,1]
values=sorted(d['by_expert'],key=lambda r:r['index'])
colors=['#ff7f00' if r['exposure']//896<30 else '#377eb8' for r in values]
ax.bar([str(r['index']) for r in values],[r['rate']*100 for r in values],color=colors)
for i,r in enumerate(values):
    if r['exposure']//896<30:
        ax.text(i,r['rate']*100+.002,f"n={r['exposure']//896}",ha='center',fontsize=8)
ax.axhline(d['different_fraction']*100,c='black',ls='--',lw=1,label='Whole tensor rate')
ax.set(xlabel='Expert ID',ylabel='Different scalar values (%)',title='Expert rates; orange: fewer than 30 routed rows')
ax.legend(fontsize=8)
ax=axs[1,0]
values=sorted(d['by_feature_tile128'],key=lambda r:r['index'])
ax.bar([f"{r['index']*128}-{r['index']*128+127}" for r in values],[r['count'] for r in values],color='#4daf4a')
ax.set(xlabel='Up feature tile',ylabel='Different scalar count',title='All seven 128-feature tiles are affected')
ax.tick_params(axis='x',rotation=25)
ax=axs[1,1]
labels=['CK matches FP64','HF matches FP64','Neither matches']
values=[d['fp64_verdicts'][key] for key in ('ck_matches_fp64','hf_matches_fp64','neither_matches_fp64')]
bars=ax.bar(labels,values,color=['#4daf4a','#ff7f00','#999999'])
ax.bar_label(bars)
ax.set(ylabel='Residual scalar count',title='FP64 dot-product + direct BF16 rounding')
fig.suptitle('L27 BF16 up residuals after FP64 chunk merge: 1154 / 7,096,320',fontsize=14)
fig.savefig(root/'up_difference_distribution.png',dpi=160)
fig.savefig(root/'up_difference_distribution.svg')
print(json.dumps({key:d[key] for key in ('active_experts','fixed_target_absent','fp64_verdict_by_origin','top7_expert_count_share','top7_expert_exposure_share','one_ulp_distance_to_boundary_in_bf16_spacing')},indent=2))

from pathlib import Path
import json
r=Path('/workspace/logs/step0_continuation/norm990_active_prod');a=json.loads((r/'all_experts_korder_profiles.json').read_text());b=json.loads((r/'local_split_korder_profiles.json').read_text());tested=[x for x in a+b if 'gate_different' in x];assert len(tested)==51 and len({x['expert'] for x in tested})==51
report={'active_experts':51,'gate_values':sum(x['m']*896 for x in tested),'up_values':sum(x['m']*896 for x in tested),'gate_different':sum(x['gate_different'] for x in tested),'up_different':sum(x['up_different'] for x in tested),'direct_F_linear_vs_grouped_different':sum(x['direct_hf_different'] for x in tested),'local_split_profiles':sum('gate_different' in x for x in b),'source':'actual captured HF Tensile profiles, frozen L0 inputs and checkpoint weights','scope':'standalone Triton replay; not production CK adoption'}
(r/'all51_korder_summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

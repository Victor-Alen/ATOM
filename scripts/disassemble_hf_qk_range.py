from pathlib import Path
import json,subprocess
r=Path('/workspace/logs/step0_continuation/qkv_actual_rope_prefill');info=json.loads((r/'hf_qk_code_objects.json').read_text());p=info['objects'][0];name=info['kernel'];tool='/opt/rocm/lib/llvm/bin/llvm-objdump'
rows=subprocess.check_output([tool,'-t',p],text=True).splitlines();func=[];start=None
for row in rows:
 parts=row.split()
 if len(parts)>4 and parts[2]=='F':
  addr=int(parts[0],16);func.append(addr)
  if parts[-1]==name:start=addr
assert start is not None
end=min(x for x in func if x>start)
with (r/'hf_qk_native.s').open('w') as f:subprocess.run([tool,'-d','--start-address='+hex(start),'--stop-address='+hex(end),p],stdout=f,check=True)
lines=(r/'hf_qk_native.s').read_text().splitlines();print('RANGE',hex(start),hex(end),'LINES',len(lines));print('\n'.join(x for x in lines if 'mfma' in x)[:3200])

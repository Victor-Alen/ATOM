from pathlib import Path
import json,subprocess
r=Path('/workspace/logs/step0_continuation/norm990_active_prod'); q=Path('/workspace/logs/step0_continuation/qkv_actual_rope_prefill/hf_codeobjects')
p=json.loads((r/'hf_grouped_gateup_profile_summary.json').read_text());name=next(x['name'] for x in p['kernels'] if 'MT192x128x64_MI32' in x['name']);tool='/opt/rocm/lib/llvm/bin/llvm-objdump'
found=None
for f in q.glob('*.hsaco'):
 if name.encode() in f.read_bytes():found=f;break
if found is None:
 for f in Path('/opt/rocm/lib/hipblaslt/library').glob('*BB*Alik_Bljk*gfx942.co'):
  out=q/(f.stem+'.hsaco')
  if not out.exists():subprocess.run(['/opt/rocm/lib/llvm/bin/clang-offload-bundler','--unbundle','--type=o','--targets=hipv4-amdgcn-amd-amdhsa--gfx942','--input='+str(f),'--output='+str(out)],check=True,capture_output=True)
  if name.encode() in out.read_bytes():found=out;break
assert found
func=[];start=None
for row in subprocess.check_output([tool,'-t',str(found)],text=True).splitlines():
 parts=row.split()
 if len(parts)>4 and parts[2]=='F':
  addr=int(parts[0],16);func.append(addr)
  if parts[-1]==name:start=addr
assert start is not None
end=min(x for x in func if x>start)
with (r/'hf_expert55_native.s').open('w') as f:subprocess.run([tool,'-d','--start-address='+hex(start),'--stop-address='+hex(end),str(found)],stdout=f,check=True)
print(name,found,hex(start),hex(end));lines=(r/'hf_expert55_native.s').read_text().splitlines()
for i,l in enumerate(lines):
 if 'Stagger' in l or 'GSU' in l:print(i,l)
print('\n'.join(lines[:450]))

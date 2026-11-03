import json,subprocess
from pathlib import Path
root=Path('/workspace/logs/step0_continuation/qkv_actual_rope_prefill');name=json.loads((root/'first_qk_reduction.json').read_text())['torch_qk_kernel_names'][0]
found=[]
for folder in ['/opt/rocm/lib/hipblaslt/library','/opt/rocm/lib/rocblas/library']:
 for p in Path(folder).glob('*gfx942*.co'):
  if name.encode() in p.read_bytes():found.append(str(p))
print('FOUND',found,flush=True)
for i,p in enumerate(found):
 out=root/f'hf_qk_{i}.s'
 with out.open('w') as f:subprocess.run(['/opt/rocm/llvm/bin/llvm-objdump','-d','--disassemble-symbols='+name,p],stdout=f,check=True)
 lines=[s for s in out.read_text().splitlines() if 'mfma' in s];print('\n'.join(lines[:12]),flush=True)
(root/'hf_qk_code_objects.json').write_text(json.dumps({'kernel':name,'objects':found},indent=2)+'\n')



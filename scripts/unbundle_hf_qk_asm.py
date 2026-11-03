from pathlib import Path
import json,subprocess
r=Path('/workspace/logs/step0_continuation/qkv_actual_rope_prefill');d=r/'hf_codeobjects';d.mkdir(exist_ok=True);name=json.loads((r/'first_qk_reduction.json').read_text())['torch_qk_kernel_names'][0]
files=list(Path('/opt/rocm/lib/hipblaslt/library').glob('*BB*Alik_Bljk*gfx942.co'));files.sort(key=lambda p:('_CU' in p.name,p.name));found=[]
for i,p in enumerate(files):
 out=d/(p.stem+'.hsaco')
 subprocess.run(['/opt/rocm/lib/llvm/bin/clang-offload-bundler','--unbundle','--type=o','--targets=hipv4-amdgcn-amd-amdhsa--gfx942','--input='+str(p),'--output='+str(out)],check=True,capture_output=True)
 if name.encode() not in out.read_bytes():continue
 print('FOUND',p,out,flush=True);found.append(str(out));asm=r/'hf_qk_native.s'
 with asm.open('w') as f:subprocess.run(['/opt/rocm/lib/llvm/bin/llvm-objdump','-d','--disassemble-symbols='+name,str(out)],stdout=f,check=True)
 print('\n'.join(x for x in asm.read_text().splitlines() if 'mfma' in x)[:3000],flush=True);break
(r/'hf_qk_code_objects.json').write_text(json.dumps({'kernel':name,'objects':found},indent=2)+'\n');print('CHECKED',len(files),'FOUND_COUNT',len(found),flush=True)

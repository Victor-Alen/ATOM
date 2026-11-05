"""Package modified native dependencies and their complete build inputs."""
from pathlib import Path
import json,os,re,subprocess,hashlib,tarfile,shutil
r=Path('/workspace/logs/step0_operator_parallel_20261104');work=r/'startup_delivery';repo=Path('/workspace/ATOM-csig-final')
dest=repo/'deployment';dest.mkdir(exist_ok=True)
audit=json.loads((work/'dependency_audit.json').read_text())
mapping={};origins={};builds=[]
def add(src,rel):
    src=Path(src)
    if src.is_file():mapping[str(rel)]=src
def tree(src,rel,tracked=False):
    src=Path(src)
    names=subprocess.check_output(['git','ls-files'],cwd=src,text=True).splitlines() if tracked else [str(p.relative_to(src)) for p in src.rglob('*') if p.is_file()]
    for name in names:
        if any(x in Path(name).parts for x in ('.git','__pycache__')):continue
        if Path(name).suffix in ('.o','.pyc','.lock'):continue
        add(src/name,Path(rel)/name)
for root,name in [('/workspace/aiter-mellum-head-parallel','aiter'),('/workspace/ck-mellum-hf-profile','ck')]:
    tree(root,'sources/'+name,True)
    origins[name]=dict(path=root,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),status=subprocess.check_output(['git','status','--short'],cwd=root,text=True))
    patch=subprocess.check_output(['git','diff','--binary'],cwd=root)
    (dest/(name+'.working-tree.patch')).write_bytes(patch)
tree('/app/aiter-test/csrc','sources/aiter-base/csrc')
tree('/app/aiter-test/3rdparty/ck_helper','sources/aiter-base/3rdparty/ck_helper')
for name in ('LICENSE','LICENSE.txt'):
    add(Path('/app/aiter-test')/name,'sources/aiter-base/'+name)
folders=set()
for row in audit['loaded_native_dependencies']:
    p=Path(row['path'])
    if p.suffix=='.so' and p.parent.parent==r:
        add(p,'runtime/'+str(p.relative_to(r)));folders.add(p.parent)
folders.add(r/'native_moe_prefill_jit')
extra_roots=set()
for label in ('long16','long32','long32k32','lsu2','lsu4'):
    tree(r/f'native_prefill_{label}_abi_codegen','runtime/'+f'native_prefill_{label}_abi_codegen')
for script in r.glob('build_*.py'):add(script,'build-history/'+script.name)
for manifest_path in r.glob('*build*.json'):add(manifest_path,'build-history/'+manifest_path.name)
for folder in sorted(folders):
    for p in folder.rglob('*'):
        if p.is_file() and p.suffix in ('.cu','.cpp','.h','.hpp','.cuh','.hip','.py','.json','.csv','.ninja'):
            add(p,'runtime/'+str(p.relative_to(r)))
    for ninja in folder.rglob('build.ninja'):
        builds.append(str(ninja.relative_to(r)))
        text=ninja.read_text()
        for prefix in re.findall(r'/workspace/(aiter-[^/\s]+|ck-[^/\s]+)',text):
            extra_roots.add(Path('/workspace')/prefix)
        for raw in re.findall(r'-I(/workspace/[^\s]+)',text):
            path=Path(raw)
            if path.is_dir() and path.is_relative_to(r) and not path.is_relative_to(folder):
                tree(path,'runtime/'+str(path.relative_to(r)))
for path in sorted(extra_roots):
    if str(path) in ('/workspace/aiter-mellum-head-parallel','/workspace/ck-mellum-hf-profile'):continue
    if path.name.startswith('ck-'):
        tree(path,'sources/'+path.name,True)
    else:
        tree(path/'csrc','sources/'+path.name+'/csrc')
        tree(path/'aiter','sources/'+path.name+'/aiter')
        add(path/'LICENSE','sources/'+path.name+'/LICENSE')
    origins[path.name]=dict(path=str(path),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip(),status=subprocess.check_output(['git','status','--short'],cwd=path,text=True))
# AITER's active JIT entry points must preserve the symlink routing, without
# referring to mutable experiment directories. Base-image modules stay external.
for p in (r/'native_router_bf16_jit').glob('*.so'):
    resolved=p.resolve()
    if resolved.is_relative_to(r):add(resolved,'runtime/native_router_bf16_jit/'+p.name)
profile=json.loads((repo/'scripts/server_profiles/fast.json').read_text())
configs=set()
for value in profile['env'].values():
    p=Path(value)
    if p.is_file() and p.suffix=='.json' and p.is_relative_to(r):configs.add(p)
configs.update(r/name for name in ('native_prefill_stage1_config16k.json','native_prefill_down_dispatch16k.json'))
for p in configs:add(p,'runtime/'+str(p.relative_to(r)))
manifest=[]
for name,p in sorted(mapping.items()):
    with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    manifest.append(dict(path=name,source=str(p),sha256=sha,size=p.stat().st_size))
metadata=dict(origins=origins,builds=builds,files=manifest,external_frozen_dependencies=['/opt/venv','/opt/rocm-7.2.4','/app/aiter-test/aiter/jit'],scope='Complete modified AITER and CK sources plus loaded private native binaries, generated source and build.ninja. No weights, prompts, output-token oracle, or profiler cache.')
(dest/'runtime-manifest.json').write_text(json.dumps(metadata,indent=2))
archive=dest/'runtime-gfx942.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=6,dereference=True) as tf:
    for name,p in sorted(mapping.items()):tf.add(p,arcname=name,recursive=False)
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
(dest/'runtime-gfx942.sha256').write_text(digest+'  '+archive.name+'\n')
print(json.dumps(dict(files=len(mapping),uncompressed_MB=sum(x['size'] for x in manifest)/1e6,archive_MB=archive.stat().st_size/1e6,sha256=digest,builds=len(builds))),flush=True)
parts=[]
with archive.open('rb') as stream:
    i=0
    while True:
        block=stream.read(70*1024*1024)
        if not block:break
        part=dest/(archive.name+f'.part{i:02d}');part.write_bytes(block)
        parts.append(dict(name=part.name,sha256=hashlib.sha256(block).hexdigest(),size=len(block)));i+=1
(dest/'runtime-parts.json').write_text(json.dumps(dict(archive_sha256=digest,parts=parts),indent=2))
archive.unlink()
print('PARTS',json.dumps(parts),flush=True)

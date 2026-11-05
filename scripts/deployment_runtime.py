"""Extract and verify frozen native dependencies without changing packages."""
from pathlib import Path
import fcntl,hashlib,json,os,tarfile,tempfile

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def prepare(repo):
    source=repo/'deployment'
    parts=json.loads((source/'runtime-parts.json').read_text())
    manifest=json.loads((source/'runtime-manifest.json').read_text())
    root=repo/'.runtime'/parts['archive_sha256'][:16]
    root.mkdir(parents=True,exist_ok=True)
    with (root/'extract.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not (root/'EXTRACTED').exists():
            print('Preparing frozen native runtime...',flush=True)
            with tempfile.TemporaryFile(dir=root) as combined:
                total=hashlib.sha256()
                for part in parts['parts']:
                    p=source/part['name']
                    if sha(p)!=part['sha256']:raise RuntimeError(f'Archive hash mismatch: {p}')
                    with p.open('rb') as f:
                        while block:=f.read(8*1024*1024):combined.write(block);total.update(block)
                if total.hexdigest()!=parts['archive_sha256']:raise RuntimeError('Combined archive mismatch')
                combined.seek(0)
                with tarfile.open(fileobj=combined,mode='r:gz') as archive:
                    for member in archive:
                        target=(root/member.name).resolve()
                        if not target.is_relative_to(root.resolve()) or not member.isfile():
                            raise RuntimeError(f'Unexpected archive member: {member.name}')
                        archive.extract(member,root,filter='data')
            (root/'EXTRACTED').write_text(parts['archive_sha256']+'\n')
        for item in manifest['files']:
            p=root/item['path']
            if not p.is_file() or sha(p)!=item['sha256']:
                raise RuntimeError(f'Frozen dependency changed: {p}')
        jit=root/'runtime/native_router_bf16_jit'
        for p in Path('/app/aiter-test/aiter/jit').glob('*.so'):
            target=jit/p.name
            if not target.exists():target.symlink_to(p)
    cache=repo/'.runtime/cache';cache.mkdir(exist_ok=True)
    return root,cache

def resolve(profile,repo):
    runtime,cache=prepare(repo)
    replacements={'${REPO}':str(repo),'${RUNTIME}':str(runtime),'${CACHE}':str(cache)}
    def visit(value):
        if isinstance(value,str):
            for old,new in replacements.items():value=value.replace(old,new)
            return value
        if isinstance(value,list):return [visit(v) for v in value]
        if isinstance(value,dict):return {visit(k):visit(v) for k,v in value.items()}
        return value
    return visit(profile)

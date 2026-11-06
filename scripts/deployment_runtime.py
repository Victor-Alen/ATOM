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
    replacements={'${REPO}':str(repo),'${RUNTIME}':str(runtime),'${CACHE}':str(cache),'${MODEL}':str(prepare_model(repo))}
    def visit(value):
        if isinstance(value,str):
            for old,new in replacements.items():value=value.replace(old,new)
            return value
        if isinstance(value,list):return [visit(v) for v in value]
        if isinstance(value,dict):return {visit(k):visit(v) for k,v in value.items()}
        return value
    return visit(profile)


def prepare_model(repo):
    import shutil
    metadata=repo/'deployment/mtp-model'
    manifest=json.loads((metadata/'manifest.json').read_text())
    base=Path(os.getenv('MODEL_BASE_PATH','/data/models/Mellum2.1-12B-A2.5B-Thinking')).resolve()
    from mtp_asset import resolve_head
    head=resolve_head(repo,manifest)
    model=repo/'.runtime/model';model.mkdir(parents=True,exist_ok=True)
    for name,digest in manifest['metadata'].items():
        if sha(metadata/name)!=digest:raise RuntimeError('MTP metadata changed: '+name)
        target=model/name
        if target.exists() and sha(target)!=digest:raise RuntimeError('Existing model metadata differs: '+str(target))
        if not target.exists():shutil.copy2(metadata/name,target)
    index=json.loads((model/'model.safetensors.index.json').read_text())
    names=set(index['weight_map'].values())|{'tokenizer.json','tokenizer_config.json','special_tokens_map.json','chat_template.jinja'}
    for name in names:
        source=head if name=='model-mtp.safetensors' else base/name
        if not source.is_file():raise RuntimeError('Missing supplied model file: '+str(source))
        target=model/name
        if target.is_symlink() or target.exists():
            if target.resolve()!=source.resolve():
                # Permit relocation only of a verified identical MTP head.
                if name=='model-mtp.safetensors' and target.is_symlink() and sha(target)==manifest['head_sha256']:
                    target.unlink();target.symlink_to(source)
                else:raise RuntimeError('Existing model link points elsewhere: '+str(target))
        else:target.symlink_to(source)
    return model

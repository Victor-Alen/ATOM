"""Resolve or fetch the pinned official MTP head without a developer workspace."""
from pathlib import Path
import fcntl,hashlib,json,os,tempfile,urllib.request

def _sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def _validate(path,head):
    if not path.is_file():
        raise RuntimeError(f'MTP head missing: {path}')
    if path.stat().st_size!=head['bytes'] or _sha(path)!=head['sha256']:
        raise RuntimeError(f'MTP head size/SHA256 differs from the measured official checkpoint: {path}')
    return path.resolve()

def resolve_head(repo,model_manifest):
    repo=Path(repo)
    official=json.loads((repo/'deployment/mtp-model/official-checkpoint-manifest.json').read_text())
    head=official['head']
    if head['sha256']!=model_manifest['head_sha256']:
        raise RuntimeError('Official MTP receipt and model manifest disagree')
    explicit=os.environ.get('MTP_HEAD_PATH')
    if explicit:
        return _validate(Path(explicit).expanduser(),head)
    bundled=repo/'deployment/model-assets'/head['file']
    if bundled.exists():
        return _validate(bundled,head)
    cache=repo/'.runtime/model-assets';cache.mkdir(parents=True,exist_ok=True)
    target=cache/head['file']
    with (cache/'download.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if target.exists():
            return _validate(target,head)
        if os.environ.get('MTP_DOWNLOAD','1')=='0':
            raise RuntimeError('MTP head absent and MTP_DOWNLOAD=0. Place the supplied file in deployment/model-assets/ or set MTP_HEAD_PATH.')
        override=os.environ.get('MTP_DOWNLOAD_BASE_URL')
        bases=[override] if override else ['https://huggingface.co','https://hf-mirror.com']
        errors=[]
        for base in bases:
            if not base.startswith('https://'):
                raise RuntimeError('MTP download base must use HTTPS')
            url=base.rstrip('/')+'/'+official['repo']+'/resolve/'+official['revision']+'/'+head['file']
            temporary=None
            try:
                print(f'Fetching pinned official MTP head from {base} ({head["bytes"]} bytes)...',flush=True)
                request=urllib.request.Request(url,headers={'User-Agent':'ATOM-CSIG-pinned-asset/1'})
                total=0;digest=hashlib.sha256();next_report=128*1024**2
                with urllib.request.urlopen(request,timeout=30) as response, tempfile.NamedTemporaryFile(dir=cache,prefix='.mtp-download-',delete=False) as f:
                    temporary=Path(f.name)
                    while block:=response.read(8*1024**2):
                        total+=len(block)
                        if total>head['bytes']:
                            raise RuntimeError('Download larger than pinned checkpoint')
                        f.write(block);digest.update(block)
                        if total>=next_report:
                            print(f'MTP download: {total}/{head["bytes"]} bytes',flush=True);next_report+=128*1024**2
                    f.flush();os.fsync(f.fileno())
                if total!=head['bytes'] or digest.hexdigest()!=head['sha256']:
                    raise RuntimeError('Downloaded MTP head failed pinned size/SHA256 verification')
                temporary.replace(target);temporary=None
                (cache/'download-receipt.json').write_text(json.dumps(dict(repo=official['repo'],revision=official['revision'],transport=base,bytes=total,sha256=digest.hexdigest()),indent=2))
                return target.resolve()
            except (OSError,RuntimeError) as exc:
                errors.append(type(exc).__name__+': '+str(exc))
                print(f'MTP asset transfer failed: {errors[-1]}',flush=True)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        raise RuntimeError('Pinned MTP download failed. Supply deployment/model-assets/model-mtp.safetensors or MTP_HEAD_PATH. '+ '; '.join(errors))

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'deployment/mtp-model/manifest.json').read_text())
    print(resolve_head(root,manifest))

"""Stop only a server created by this checkout and matching its saved command."""
from pathlib import Path
import argparse,json,os,signal

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8000);args=p.parse_args()
repo=Path(__file__).resolve().parents[1]
root=Path(os.getenv('LOG_ROOT','/workspace/logs/mellum-start-server'))
latest=json.loads((root/f'latest-{args.port}.json').read_text())
manifest=json.loads((Path(latest['run_dir'])/'manifest.json').read_text())
pid=latest['pid'];proc=Path('/proc')/str(pid)
if not proc.exists():print('Already stopped',pid);raise SystemExit(0)
if proc.joinpath('cwd').resolve()!=repo or manifest['cwd']!=str(repo):raise RuntimeError('Server belongs to another checkout')
cmd=[s.decode() for s in proc.joinpath('cmdline').read_bytes().split(b'\0') if s]
if cmd!=manifest['command'] or os.getpgid(pid)!=pid:raise RuntimeError('Process identity changed')
os.killpg(pid,signal.SIGTERM);print('SIGTERM sent to verified server process group',pid)

"""Launch a frozen Mellum profile, then verify a real generation request."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request


def read_json(path):
    return json.loads(Path(path).read_text())


def api(url, payload=None, timeout=5):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("fast",), default=os.getenv("PROFILE", "fast"))
    parser.add_argument("--dry-run", action="store_true", help="Validate and print configuration without loading GPU")
    parser.add_argument("--port", type=int, default=int(os.getenv("SERVER_PORT", "8000")))
    parser.add_argument("--engine-port", type=int, default=int(os.getenv("ENGINE_PORT", "18000")))
    parser.add_argument("--gpu", default=os.getenv("MELLUM_GPU", "1"))
    parser.add_argument("--host", default=os.getenv("SERVER_HOST", "127.0.0.1"))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--ablation",choices=("step0","point1","point12","point123"),default="point123")
    parser.add_argument("--moe-opt",choices=("0","1"),default=os.getenv("OPT_MOE"))
    parser.add_argument("--attention-opt",choices=("0","1"),default=os.getenv("OPT_ATTENTION"))
    parser.add_argument("--compile-graph-opt",choices=("0","1"),default=os.getenv("OPT_COMPILE_GRAPH"))
    args = parser.parse_args()
    if args.port == args.engine_port or not all(1024 <= p <= 65535 for p in (args.port, args.engine_port)):
        parser.error("Use different HTTP/engine ports in 1024..65535")
    if args.gpu not in ("0", "1"):
        parser.error("This validated profile uses one GPU; --gpu must be 0 or 1")
    profile_path = Path(__file__).parent / "server_profiles" / ("mtp.json" if args.profile.startswith("mtp") else "fast.json")
    from deployment_runtime import resolve
    profile = resolve(read_json(profile_path), Path(__file__).resolve().parents[1])
    env = os.environ.copy()
    # Prevent inherited diagnostic/reference flags from changing the measured path.
    for key in list(env):
        if key.startswith(("ATOM_", "AITER_", "CK_MELLUM_")):
            del env[key]
    env.update(profile["env"])
    # Explicit overrides are logged as an unmeasured ablation, never silently called the winner.
    overrides = json.loads(os.getenv("OPT_OVERRIDES_JSON", "{}"))
    if not isinstance(overrides, dict) or any(k not in profile["env"] or not isinstance(v, str) for k, v in overrides.items()):
        parser.error("OPT_OVERRIDES_JSON must map existing profile environment keys to strings")
    selections={"step0":(0,0,0),"point1":(1,0,0),"point12":(1,1,0),"point123":(1,1,1)}
    points=list(selections[args.ablation])
    for i,selected in enumerate((args.moe_opt,args.attention_opt,args.compile_graph_opt)):
        if selected is not None:points[i]=int(selected)
    groups=(
        ("ATOM_MELLUM_NATIVE_DECODE_MOE","ATOM_MELLUM_NATIVE_PREFILL_STAGE1","ATOM_MELLUM_NATIVE_PREFILL_STAGE2","ATOM_MELLUM_NATIVE_HF_ROUTER","ATOM_MELLUM_ROUTER_FUSED_BF16","ATOM_MELLUM_BATCH_DECODE_MOE","ATOM_MELLUM_PREFILL_MOE_STREAMS"),
        ("ATOM_MELLUM_NATIVE_HF_ATTENTION","ATOM_MELLUM_FUSED_PREFILL_SOFTMAX","ATOM_MELLUM_PREFILL_SPARSE_QK","ATOM_MELLUM_PREFILL_LONG_QK","ATOM_MELLUM_PREFILL_SPARSE_PV","ATOM_MELLUM_PREFILL_SPARSE_WINDOW"),
        ("ATOM_MELLUM_COMPILE_PIECES","ATOM_MELLUM_GRAPH_PIECES"))
    for enabled,keys in zip(points,groups):
        if not enabled:
            for key in keys:env[key]="0"
    env.update(overrides)
    env.update(ROCR_VISIBLE_DEVICES="0,1", HIP_VISIBLE_DEVICES=args.gpu)
    command = list(profile["cmd"])
    command[0] = sys.executable
    if not points[2] and "--enforce-eager" not in command:command.append("--enforce-eager")
    for flag, value in (("--server-port", args.port), ("--port", args.engine_port), ("--host", args.host)):
        command[command.index(flag) + 1] = str(value)
    if args.profile == "mtp-off":
        env.update(ATOM_MELLUM_MTP_NATIVE_ATTN="0", ATOM_MELLUM_MTP_GRAPH_METADATA="0")
        for flag in ("--method", "--num-speculative-tokens"):
            i = command.index(flag)
            del command[i:i + 2]
    cwd = Path(profile["cwd"])
    for path in profile["required_paths"]:
        if not Path(path).exists():
            raise RuntimeError(f"Missing deployment dependency: {path}")
    for name, expected in profile.get("source_hashes", {}).items():
        actual = hashlib.sha256((cwd / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Source differs from measured snapshot: {cwd / name}")
    for name, expected in profile.get("dependency_hashes", {}).items():
        with Path(name).open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected:
            raise RuntimeError(f"Dependency differs from accepted snapshot: {name}")
    model = Path(command[command.index("--model") + 1])
    index = read_json(model / "model.safetensors.index.json")
    for shard in set(index["weight_map"].values()):
        if not (model / shard).is_file():
            raise RuntimeError(f"Missing model shard: {shard}")
    if args.profile.startswith("mtp"):
        with (model / "model-mtp.safetensors").open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != profile["mtp_head_sha256"]:
            raise RuntimeError("MTP head SHA256 mismatch")
    record = {"profile": args.profile, "cwd": str(cwd), "command": command,
              "profile_path": str(profile_path), "evidence": profile["evidence"],
              "env": {k: env[k] for k in profile["env"]}, "overrides": overrides, "optimization_points":points, "ablation":args.ablation}
    record["env"].update(ROCR_VISIBLE_DEVICES=env["ROCR_VISIBLE_DEVICES"], HIP_VISIBLE_DEVICES=env["HIP_VISIBLE_DEVICES"])
    if args.dry_run:
        print(json.dumps(record, indent=2))
        return
    log_root = Path(os.getenv("LOG_ROOT", "/workspace/logs/mellum-start-server"))
    log_root.mkdir(parents=True, exist_ok=True)
    lock = (log_root / f"port-{args.port}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for port in (args.port, args.engine_port):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", port))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = log_root / f"{stamp}-{args.profile}-{args.port}-{os.getpid()}"
    run.mkdir()
    logfile = run / "server.log"
    print(f"PROFILE={args.profile} MTP={args.profile == 'mtp'} GPU={args.gpu}", flush=True)
    print(f"REPO={cwd}\nLOG={logfile}\nCOMMAND={shlex.join(command)}", flush=True)
    print("Scope: " + profile["evidence"]["scope"], flush=True)
    with logfile.open("wb") as output:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    record["pid"] = process.pid
    (run / "pid").write_text(str(process.pid) + "\n")
    (run / "manifest.json").write_text(json.dumps(record, indent=2))
    address = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    url = f"http://{address}:{args.port}"
    try:
        deadline = time.monotonic() + args.timeout
        next_notice = time.monotonic() + 30
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"Server exited with {process.returncode}; inspect {logfile}")
            try:
                models = api(url + "/v1/models")
                model_id = models["data"][0]["id"]
                break
            except (OSError, ValueError, KeyError, IndexError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(f"Startup timed out; inspect {logfile}")
            if time.monotonic() > next_notice:
                print(f"Loading/compiling; PID={process.pid}, log={logfile}", flush=True)
                next_notice = time.monotonic() + 30
            time.sleep(2)
        payload = {"model": model_id, "prompt": "def add(a, b):\n    return", "max_tokens": 8,
                   "temperature": 0.0, "ignore_eos": True, "stream": False}
        result = api(url + "/v1/completions", payload, timeout=args.timeout)
        if result.get("usage", {}).get("completion_tokens") != 8 or not result.get("choices"):
            raise RuntimeError(f"Generation smoke did not return 8 tokens: {result}")
        (run / "smoke.json").write_text(json.dumps({"request": payload, "response": result}, indent=2))
        (run / "READY").write_text(f"{url}\nGeneration smoke: 8 tokens\n")
        (log_root / f"latest-{args.port}.json").write_text(json.dumps({"run_dir": str(run), "pid": process.pid, "url": url}))
        print(f"READY {url}/v1 PID={process.pid}; actual generation=8 tokens; record={run}", flush=True)
    except BaseException:
        # Only the process group created by this invocation; never stop another service.
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=20)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise


if __name__ == "__main__":
    main()

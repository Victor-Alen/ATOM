import json
import os
from pathlib import Path

os.environ["ATOM_MELLUM_COMPILE_PIECES"] = "1"
os.environ["ATOM_MELLUM_GRAPH_PIECES"] = "1"
import argparse

import torch

from atom.model_ops import mellum_piecewise as p

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = args.output
root.mkdir(parents=True, exist_ok=True)
torch.manual_seed(271)
rows = []
with torch.inference_mode():
    w = torch.randn(4096, 2304, device="cuda", dtype=torch.bfloat16)
    nw = torch.randn(2304, device="cuda", dtype=torch.bfloat16)
    for n in [1, 8, 16, 32, 64, 32, 8, 1]:
        x = torch.randn(n, 2304, device="cuda", dtype=torch.bfloat16)
        parts = tuple((i, i + 1) for i in range(n))
        for kind in ["linear", "norm"]:
            weight = w if kind == "linear" else nw
            eps = None if kind == "linear" else 1e-6
            ref = torch.cat(
                [
                    (
                        torch.nn.functional.linear(x[a:b], weight)
                        if eps is None
                        else p.norm_tensor(x[a:b], weight, eps)
                    )
                    for a, b in parts
                ]
            )
            got = p.request_region(x, weight, None, parts, True, norm_eps=eps)
            snapshot = got.clone()
            changed = p.request_region(x + 1, weight, None, parts, True, norm_eps=eps)
            rows.append(
                {
                    "op": kind,
                    "n": n,
                    "equal": torch.equal(ref, got),
                    "storage_safe": torch.equal(snapshot, got),
                    "fresh_input": not torch.equal(got, changed),
                }
            )
        for heads in [4, 32]:
            x = torch.randn(n, heads, 128, device="cuda", dtype=torch.bfloat16)
            weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
            ref = torch.cat(
                [p.norm_tensor(x[i : i + 1], weight, 1e-6) for i in range(n)]
            )
            got = p.norm_tensor(x, weight, 1e-6)
            rows.append(
                {
                    "op": "batch_qknorm",
                    "n": n,
                    "heads": heads,
                    "equal": torch.equal(ref, got),
                }
            )
        q = torch.randn(n, 32, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(n, 4, 128, device="cuda", dtype=torch.bfloat16)
        cos = torch.randn(n, 1, 128, device="cuda", dtype=torch.bfloat16)
        sin = torch.randn_like(cos)
        ref = p.rope_tensor(q, k, cos, sin)
        got = p.rope(q, k, cos, sin)
        rows.append(
            {
                "op": "rope",
                "n": n,
                "equal": all(torch.equal(a, b) for a, b in zip(ref, got)),
            }
        )
        values = torch.randn(n * 8, 1792, device="cuda", dtype=torch.bfloat16)
        rows.append(
            {
                "op": "gate",
                "n": n,
                "equal": torch.equal(p.gate_tensor(values), p.gate(values)),
            }
        )
    x = torch.randn(8, 2304, device="cuda", dtype=torch.bfloat16)
    parts = tuple((i, i + 1) for i in range(8))
    for weight in [w, w[:2048], w]:
        ref = torch.cat([torch.nn.functional.linear(x[a:b], weight) for a, b in parts])
        got = p.request_region(x, weight, None, parts, True)
        rows.append(
            {
                "op": "shared_storage_weight_view",
                "outputs": weight.shape[0],
                "equal": torch.equal(ref, got),
            }
        )
    report = {
        "cases": len(rows),
        "all_equal": all(
            r["equal"] and r.get("storage_safe", True) and r.get("fresh_input", True)
            for r in rows
        ),
        "rows": rows,
    }
    (root / "final_fixed_input_probe.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    assert report["all_equal"], report

    # Isolated operator traces: count GPU kernel events and host launch calls.
    n = 32
    x = torch.randn(n, 2304, device="cuda", dtype=torch.bfloat16)
    parts = tuple((i, i + 1) for i in range(n))

    def old():
        return torch.cat([p.norm_tensor(x[a:b], nw, 1e-6) for a, b in parts])

    def new():
        return p.request_region(x, nw, None, parts, True, norm_eps=1e-6)

    stats = {}
    for name, fn in [("norm_eager", old), ("norm_graph", new)]:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
        trace = root / (name + ".trace.json")
        prof.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text())["traceEvents"]
        stats[name] = {
            "kernel_events": sum(e.get("cat") == "kernel" for e in events),
            "launch_events": sum(
                "launch" in e.get("name", "").lower()
                and e.get("cat") in ["cuda_runtime", "hip_runtime"]
                for e in events
            ),
            "graph_launch_events": sum(
                "graphlaunch" in e.get("name", "").lower() for e in events
            ),
            "repetitions": 3,
        }
    (root / "piece_profile_summary.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats), flush=True)

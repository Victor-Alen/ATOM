"""Env-gated per-layer tracing for the Mellum forward.

Enabled only when ATOM_LAYER_TRACE_DIR is set. Each model forward appends one
JSON line per hooked submodule with a digest and cheap numeric fingerprints, so
two identical requests can be compared to find the first divergent operator.
"""
import hashlib
import json
import os

import torch

STATE = {"step": 0, "path": None, "installed": False}


def _reinterpret_bytes(x):
    if x.dtype in (torch.bfloat16, torch.float16):
        return x.contiguous().view(torch.uint8)
    return x.contiguous().view(torch.uint8)


def _rec(name, t):
    if not torch.is_tensor(t):
        return None
    if t.dtype not in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
        return None
    x = t.detach().to("cpu")
    raw = _reinterpret_bytes(x)
    digest = hashlib.sha1(raw.numpy().tobytes()).hexdigest()[:12]
    f = x.float()
    last = f.reshape(-1, f.shape[-1])[-1] if f.numel() else f.reshape(1)
    return {
        "name": name,
        "shape": list(t.shape),
        "digest": digest,
        "absmax": round(float(f.abs().max()), 6) if f.numel() else 0.0,
        "mean": round(float(f.mean()), 8) if f.numel() else 0.0,
        "last_absmax": round(float(last.abs().max()), 6) if last.numel() else 0.0,
        "last_sum": round(float(last.sum()), 6) if last.numel() else 0.0,
    }


def _first_tensor(obj):
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (list, tuple)):
        for item in obj:
            got = _first_tensor(item)
            if got is not None:
                return got
    if isinstance(obj, dict):
        for item in obj.values():
            got = _first_tensor(item)
            if got is not None:
                return got
    return None


class _Step:
    def __init__(self):
        self.records = []

    def add(self, name, obj):
        for i, t in enumerate(obj if isinstance(obj, (list, tuple)) else [obj]):
            rec = _rec(name if i == 0 else f"{name}#{i}", t)
            if rec is not None:
                self.records.append(rec)


def _hook(name, current):
    def fn(module, args, output):
        tensors = []
        for a in args:
            t = _first_tensor(a)
            if t is not None:
                tensors.append(t)
        if tensors:
            current.add(f"{name}.in", tensors[0])
        out = _first_tensor(output)
        if out is not None:
            current.add(f"{name}.out", out)

    return fn


def _flush(step_idx, records):
    path = STATE["path"]
    if not path or not records:
        return
    rec = {"step": step_idx, "rows": records[0].get("shape", [0])[0], "items": records}
    with open(path, "a") as fh:
        fh.write(json.dumps(rec) + chr(10))


def install_layer_trace(model):
    if STATE["installed"]:
        return
    outdir = os.environ.get("ATOM_LAYER_TRACE_DIR")
    if not outdir:
        return
    os.makedirs(outdir, exist_ok=True)
    STATE["path"] = os.path.join(outdir, "layers.jsonl")
    current = _Step()
    handles = []

    layers = getattr(model, "layers", None)
    if layers is None:
        return

    for idx, layer in enumerate(layers):
        for sub_name in ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp"):
            sub = getattr(layer, sub_name, None)
            if sub is not None:
                handles.append(sub.register_forward_hook(_hook(f"L{idx}.{sub_name}", current)))
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            for name, child in mlp.named_children():
                if "gate" in name.lower() or "router" in name.lower():
                    handles.append(
                        child.register_forward_hook(_hook(f"L{idx}.mlp.{name}", current))
                    )
        handles.append(layer.register_forward_hook(_hook(f"L{idx}.layer", current)))

    orig_model_forward = model.forward

    def wrapped(*args, **kwargs):
        current.records = []
        try:
            return orig_model_forward(*args, **kwargs)
        finally:
            step_idx = STATE["step"]
            STATE["step"] += 1
            _flush(step_idx, current.records)
            current.records = []

    model.forward = wrapped
    STATE["installed"] = True
    STATE["handles"] = handles
    print(f"[layer-trace] installed {len(handles)} hooks -> {STATE['path']}", flush=True)

#!/usr/bin/env python3
"""Per-layer drift profile: HF vs ATOM on the same prompt, last-token fingerprints.

Runs one HF eager forward over a scenario-C prompt, hooks every decoder layer's
attention output and MoE output, and writes the same JSONL schema that
atom/models/layer_trace.py produces, so the two traces can be compared directly.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

MODEL_PATH = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
SHARED = Path("/workspace/ATOM")
REFERENCE = SHARED / "scripts" / "hf_reference_generate.py"


def load_reference_module():
    spec = importlib.util.spec_from_file_location("hf_reference_generate", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fp(name, t):
    if not torch.is_tensor(t) or t.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return None
    x = t.detach().to("cpu")
    last = x if x.dim() == 2 else x.reshape(-1, x.shape[-1])
    last = last[-1:]
    digest = hashlib.sha1(last.float().numpy().tobytes()).hexdigest()[:12]
    return {
        "name": name,
        "shape": list(x.shape),
        "digest_last": digest,
        "absmax_last": round(float(last.float().abs().max()), 6),
        "mean_last": round(float(last.float().mean()), 8),
        "absmax_all": round(float(x.float().abs().max()), 6),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-file", default="/workspace/correctness/hf_scenario_c_200_eager.jsonl")
    ap.add_argument("--prompt-id", default="C-0000")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    text = None
    with open(args.prompt_file, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["id"] == args.prompt_id:
                text = rec["prompt"]
                break
    assert text is not None, "prompt not found"

    ref = load_reference_module()
    MellumConfig, MellumForCausalLM = ref.load_mellum_classes()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    config = MellumConfig.from_pretrained(MODEL_PATH)
    config._attn_implementation = "eager"
    print("loading HF ...", flush=True)
    model = MellumForCausalLM.from_pretrained(
        MODEL_PATH, config=config, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    ref.load_checkpoint_experts(model, MODEL_PATH)
    model = model.to(args.device).eval()
    print("HF ready", flush=True)

    items = []
    handles = []
    layers = model.model.layers

    def hook(tag):
        def fn(module, args_in, output):
            tensors = [t for t in (output if isinstance(output, (tuple, list)) else [output]) if torch.is_tensor(t)]
            if not tensors:
                return
            rec = fp(tag, tensors[0])
            if rec:
                items.append(rec)
        return fn

    for idx, layer in enumerate(layers):
        handles.append(layer.self_attn.register_forward_hook(hook(f"L{idx}.self_attn.out")))
        handles.append(layer.mlp.register_forward_hook(hook(f"L{idx}.mlp.out")))
        handles.append(layer.register_forward_hook(hook(f"L{idx}.layer.out")))

    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(args.device)
    print("prompt tokens:", ids.shape[1], flush=True)
    with torch.no_grad():
        model(input_ids=ids, logits_to_keep=1)
    for h in handles:
        h.remove()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = {"step": 0, "rows": int(ids.shape[1]), "items": items}
    out.write_text(json.dumps(rec) + chr(10))
    print("wrote", out, "items", len(items), flush=True)
    print("### DONE", flush=True)


if __name__ == "__main__":
    main()

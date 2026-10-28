#!/usr/bin/env python3
"""Generate greedy Mellum reference tokens with native HF model code.

The downloaded Mellum modules are loaded without modifying the frozen
Transformers installation. Results are appended after every prompt for resume.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

import torch
from transformers import AutoTokenizer

MODEL_PATH = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
REPO_ROOT = Path(__file__).resolve().parents[1]


def load_mellum_classes():
    source_dir = REPO_ROOT / "reference" / "hf_mellum"
    package_name = "transformers.models.mellum"
    package = types.ModuleType(package_name)
    package.__path__ = [str(source_dir)]
    package.__package__ = package_name
    sys.modules[package_name] = package
    for name in ("configuration_mellum", "modeling_mellum"):
        fullname = f"{package_name}.{name}"
        spec = importlib.util.spec_from_file_location(fullname, source_dir / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = module
        spec.loader.exec_module(module)
    return (
        sys.modules[f"{package_name}.configuration_mellum"].MellumConfig,
        sys.modules[f"{package_name}.modeling_mellum"].MellumForCausalLM,
    )


def load_checkpoint_experts(model, model_path):
    """Pack per-expert checkpoint tensors into native Mellum's 3D parameters."""
    from contextlib import ExitStack

    from safetensors import safe_open

    model_path = Path(model_path)
    with (model_path / "model.safetensors.index.json").open(encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]

    with ExitStack() as stack:
        shards = {}

        def get_tensor(name):
            shard_name = weight_map[name]
            if shard_name not in shards:
                shards[shard_name] = stack.enter_context(
                    safe_open(model_path / shard_name, framework="pt", device="cpu")
                )
            return shards[shard_name].get_tensor(name)

        for layer_idx, layer in enumerate(model.model.layers):
            experts = layer.mlp.experts
            gate_up = torch.empty_like(experts.gate_up_proj, device="cpu")
            down = torch.empty_like(experts.down_proj, device="cpu")
            intermediate_size = experts.intermediate_dim
            for expert_idx in range(experts.num_experts):
                prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
                gate_up[expert_idx, :intermediate_size].copy_(
                    get_tensor(f"{prefix}.gate_proj.weight")
                )
                gate_up[expert_idx, intermediate_size:].copy_(
                    get_tensor(f"{prefix}.up_proj.weight")
                )
                down[expert_idx].copy_(get_tensor(f"{prefix}.down_proj.weight"))
            experts.gate_up_proj.data.copy_(gate_up)
            experts.down_proj.data.copy_(down)
            if layer_idx % 4 == 3 or layer_idx + 1 == len(model.model.layers):
                print(f"Packed checkpoint experts through layer {layer_idx + 1}.", flush=True)


def read_items(args):
    if args.prompt is not None:
        return [{"id": "smoke", "prompt": args.prompt}]
    items = []
    with open(args.dataset, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < args.start:
                continue
            item = json.loads(line)
            items.append(item)
            if args.limit is not None and len(items) >= args.limit:
                break
    return items


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if output_path.exists():
        with output_path.open(encoding="utf-8") as handle:
            for line in handle:
                completed.add(json.loads(line)["id"])

    config_cls, model_cls = load_mellum_classes()
    config = config_cls.from_pretrained(MODEL_PATH)
    print(f"Loading HF reference with {args.attention} attention...", flush=True)
    model = model_cls.from_pretrained(
        MODEL_PATH,
        config=config,
        dtype=torch.bfloat16,
        attn_implementation=args.attention,
    )
    load_checkpoint_experts(model, MODEL_PATH)
    model = model.to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print(f"Model ready on {model.device}; resuming with {len(completed)} saved items.", flush=True)

    for item in read_items(args):
        item_id = item["id"]
        if item_id in completed:
            print(f"SKIP {item_id}", flush=True)
            continue
        prompt = item["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
        token_ids = generated[0, inputs.input_ids.shape[1] :].tolist()
        record = {
            "id": item_id,
            "prompt": prompt,
            "prompt_token_ids": inputs.input_ids[0].tolist(),
            "token_ids": token_ids,
            "text": tokenizer.decode(token_ids, skip_special_tokens=False),
            "prompt_tokens": inputs.input_ids.shape[1],
            "elapsed_s": time.perf_counter() - started,
            "attention": args.attention,
            "dtype": str(model.dtype),
        }
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
        print(
            f"DONE {item_id} prompt={record['prompt_tokens']} generated={len(token_ids)} "
            f"elapsed={record['elapsed_s']:.3f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare ATOM's dumped parameters with the checkpoint tensors, bitwise."""
import json
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open

MODEL = Path("/data/models/Mellum2.1-12B-A2.5B-Thinking")
DUMP = Path("/workspace/logs/atom_params.pt")

blob = torch.load(DUMP, map_location="cpu")
print("dumped keys:", len(blob))

index = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
stack = ExitStack()
shards = {}


def get(name):
    shard = index[name]
    if shard not in shards:
        shards[shard] = stack.enter_context(safe_open(MODEL / shard, framework="pt", device="cpu"))
    return shards[shard].get_tensor(name)


def cmp(tag, a, b):
    same = a.shape == b.shape and torch.equal(a, b)
    if not same and a.shape == b.shape:
        d = (a.float() - b.float()).abs().max().item()
    else:
        d = float("nan")
    print(f"{tag:32s} same={same} shape={tuple(a.shape)} vs {tuple(b.shape)} max|d|={d}")
    return same


with stack:
    for li in (0, 2):
        p = f"model.layers.{li}"
        cmp(f"L{li}.q", blob[f"L{li}.q"], get(f"{p}.self_attn.q_proj.weight"))
        cmp(f"L{li}.k", blob[f"L{li}.k"], get(f"{p}.self_attn.k_proj.weight"))
        cmp(f"L{li}.v", blob[f"L{li}.v"], get(f"{p}.self_attn.v_proj.weight"))
        cmp(f"L{li}.o", blob[f"L{li}.o"], get(f"{p}.self_attn.o_proj.weight"))
        cmp(f"L{li}.q_norm", blob[f"L{li}.q_norm"], get(f"{p}.self_attn.q_norm.weight"))
        cmp(f"L{li}.k_norm", blob[f"L{li}.k_norm"], get(f"{p}.self_attn.k_norm.weight"))
        cmp(f"L{li}.in_ln", blob[f"L{li}.in_ln"], get(f"{p}.input_layernorm.weight"))
        cmp(f"L{li}.post_ln", blob[f"L{li}.post_ln"], get(f"{p}.post_attention_layernorm.weight"))
        cmp(f"L{li}.gate", blob[f"L{li}.gate"], get(f"{p}.mlp.gate.weight"))

        w13 = blob[f"L{li}.w13"]
        w2 = blob[f"L{li}.w2"]
        print(f"L{li}.w13 shape {tuple(w13.shape)} w2 {tuple(w2.shape)}")
        for e in (0, 1):
            gate = get(f"{p}.mlp.experts.{e}.gate_proj.weight")
            up = get(f"{p}.mlp.experts.{e}.up_proj.weight")
            down = get(f"{p}.mlp.experts.{e}.down_proj.weight")
            print(f"  expert {e}: gate {tuple(gate.shape)} up {tuple(up.shape)} down {tuple(down.shape)}")
            cmp(f"    e{e} w13 == cat[gate,up]", w13[e], torch.cat([gate, up], 0))
            cmp(f"    e{e} w13 == cat[up,gate]", w13[e], torch.cat([up, gate], 0))
            cmp(f"    e{e} w2  == down", w2[e], down)
print("### DONE", flush=True)

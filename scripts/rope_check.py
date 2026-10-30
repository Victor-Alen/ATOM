#!/usr/bin/env python3
"""Compare the in-engine rope tables with HF's MellumRotaryEmbedding for both layer types."""
import importlib.util
import sys
import types
from pathlib import Path

import torch

MODEL_PATH = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
REF = Path("/workspace/ATOM/reference/hf_mellum")


def load_hf_modeling():
    pkg = "_hfpkg"
    if pkg not in sys.modules:
        m = types.ModuleType(pkg)
        m.__path__ = [str(REF)]
        sys.modules[pkg] = m
    spec = importlib.util.spec_from_file_location(pkg + ".modeling_mellum", REF / "modeling_mellum.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg + ".modeling_mellum"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    mod = load_hf_modeling()
    cfg = mod.MellumConfig.from_pretrained(MODEL_PATH)
    rot = mod.MellumRotaryEmbedding(config=cfg).to("cpu")
    pos = torch.arange(0, 3000)
    for lt in ("sliding_attention", "full_attention"):
        cos_hf, sin_hf = rot(torch.zeros(1, pos.numel(), cfg.hidden_size), pos[None, :], lt)
        # in-engine reimplementation
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        params = cfg.rope_parameters[lt]
        head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
        if params["rope_type"] == "default":
            inv = 1.0 / (params["rope_theta"] ** (torch.arange(0, head_dim, 2).float() / head_dim))
            sc = 1.0
        else:
            inv, sc = ROPE_INIT_FUNCTIONS[params["rope_type"]](cfg, torch.device("cpu"), layer_type=lt)
        freqs = pos.float()[:, None] * inv[None, :].float()
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_me = (emb.cos() * sc).to(torch.bfloat16)
        sin_me = (emb.sin() * sc).to(torch.bfloat16)
        dh = (cos_hf[0].float() - cos_me.float()).abs().max().item()
        ds = (sin_hf[0].float() - sin_me.float()).abs().max().item()
        print(f"{lt}: cos max|d|={dh:.8f} sin max|d|={ds:.8f} | equal bits: {torch.equal(cos_hf[0], cos_me)} {torch.equal(sin_hf[0], sin_me)}")
    print("### DONE", flush=True)


if __name__ == "__main__":
    main()

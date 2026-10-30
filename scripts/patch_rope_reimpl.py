#!/usr/bin/env python3
"""Replace the HF-module rope import with an exact reimplementation that uses
transformers' own rope parameter initialisers."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
start = s.index("def _rotary_module(cfg, dev):")
end = s.index("def reference_forward(model, input_ids, positions):")
new = '''def _rotary_module(cfg, dev):
    """Exact reimplementation of HF's MellumRotaryEmbedding, per layer type."""
    if "fn" in _ROT:
        return _ROT["fn"], _ROT["types"], _ROT["W"]
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    types = list(cfg.layer_types)
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
    tables = {}
    for lt in sorted(set(types)):
        params = cfg.rope_parameters[lt]
        rtype = params["rope_type"]
        if rtype == "default":
            inv = 1.0 / (
                params["rope_theta"]
                ** (torch.arange(0, head_dim, 2, device=dev).float() / head_dim)
            )
            scaling = 1.0
        else:
            inv, scaling = ROPE_INIT_FUNCTIONS[rtype](cfg, dev, layer_type=lt)
        tables[lt] = (inv.to(dev).float(), float(scaling))

    def fn(li, pos, dtype):
        lt = types[li]
        inv, scaling = tables[lt]
        freqs = pos.float()[:, None] * inv[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * scaling).to(dtype), (emb.sin() * scaling).to(dtype)

    _ROT["fn"] = fn
    _ROT["types"] = types
    _ROT["W"] = getattr(cfg, "sliding_window", None)
    return fn, types, _ROT["W"]


'''
s = s[:start] + new + s[end:]
p.write_text(s)
ast.parse(s)
print("rotary reimplemented")

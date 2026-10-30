#!/usr/bin/env python3
"""Use HF's own MellumRotaryEmbedding (via the same package shim the offline probes
use) so the rope tables are identical by construction instead of reimplemented."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
start = s.index("def _rotary(cfg, dev):")
end = s.index("def reference_forward(")
new = '''def _rotary(cfg, dev):
    """HF's own per-layer-type rotary embedding, loaded from the reference tree."""
    if "fn" in _ROT:
        return _ROT["fn"], _ROT["types"], _ROT["W"]
    import importlib.util
    import sys
    import types
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "reference" / "hf_mellum"
    pkg_name = "transformers.models.mellum"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(src)]
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg
        for name in ("configuration_mellum", "modeling_mellum"):
            full = f"{pkg_name}.{name}"
            spec = importlib.util.spec_from_file_location(full, src / f"{name}.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[full] = mod
            spec.loader.exec_module(mod)
    MellumConfig = sys.modules[pkg_name + ".configuration_mellum"].MellumConfig
    modeling = sys.modules[pkg_name + ".modeling_mellum"]
    hf_cfg = MellumConfig.from_pretrained(os.environ["ATOM_REFERENCE_MODEL_DIR"])
    rot = modeling.MellumRotaryEmbedding(config=hf_cfg).to(dev)
    layer_types = list(hf_cfg.layer_types)
    hidden = hf_cfg.hidden_size

    def fn(li, pos, dtype):
        lt = layer_types[li]
        cos, sin = rot(
            torch.zeros(1, pos.numel(), hidden, dtype=dtype, device=dev),
            pos.view(1, -1),
            lt,
        )
        return cos[0], sin[0]

    _ROT.update(fn=fn, types=layer_types, W=getattr(hf_cfg, "sliding_window", None))
    return fn, layer_types, _ROT["W"]


'''
s = s[:start] + new + s[end:]
p.write_text(s)
ast.parse(s)
print("rope now uses HF's own module")

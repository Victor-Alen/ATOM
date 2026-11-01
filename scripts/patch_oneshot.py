#!/usr/bin/env python3
"""One-shot L0 dump: latch on the first prefill so every dumped tensor provably comes
from the same forward call, and include emb / h_ln / q / k / attn / blk."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()

# remove the scattered, racy dumps
for frag in [
    """    _ldir0 = os.environ.get("ATOM_REFERENCE_DUMPL")
    if _ldir0 and int(x.shape[0]) > 1:
        torch.save(x[-1].detach().float().cpu(), f"{_ldir0}/emb.pt")
        torch.save(
            torch.tensor(input_ids[-8:].tolist(), dtype=torch.long), f"{_ldir0}/last8.pt"
        )
""",
    """            _ld3 = os.environ.get("ATOM_REFERENCE_DUMPL")
            if _ld3 and li == 0 and q.shape[0] > 1:
                torch.save(q[-1].detach().float().cpu(), f"{_ld3}/q_L0.pt")
                torch.save(k[-1].detach().float().cpu(), f"{_ld3}/k_L0.pt")
                if hi - lo > 1:
                    torch.save(h[-1].detach().float().cpu(), f"{_ld3}/h_ln_L0.pt")
""",
]:
    s = s.replace(frag, "", 1)

# one latch
s = s.replace("_HIST: list = []", "_HIST: list = []\n_DUMPED: set = set()", 1)

# single dump inside the layer-0 prefill
old = """            h = _hf_norm(x[lo:hi], layer.input_layernorm.weight, eps)"""
new = """            h = _hf_norm(x[lo:hi], layer.input_layernorm.weight, eps)
            _d1 = os.environ.get("ATOM_REFERENCE_DUMPL")
            _do_dump = bool(_d1) and li == 0 and (hi - lo) > 1 and "l0" not in _DUMPED
            if _do_dump:
                _DUMPED.add("l0")
                torch.save(
                    {
                        "emb_last": x[hi - 1].detach().float().cpu(),
                        "h_ln_last": h[-1].detach().float().cpu(),
                        "ln_w": layer.input_layernorm.weight.detach().float().cpu(),
                        "eps": torch.tensor([float(eps)]),
                        "ids_last8": torch.tensor(input_ids[-8:].tolist(), dtype=torch.long),
                        "n_new": torch.tensor([hi - lo]),
                    },
                    f"{_d1}/l0_in.pt",
                )"""
assert old in s
s = s.replace(old, new, 1)

old2 = """            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin"""
new2 = """            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
            if _do_dump:
                torch.save(
                    {
                        "q_last": q[-1].detach().float().cpu(),
                        "k_last": k[-1].detach().float().cpu(),
                        "cos_last": cos[-1].detach().float().cpu(),
                        "sin_last": sin[-1].detach().float().cpu(),
                    },
                    f"{_d1}/l0_qk.pt",
                )"""
assert old2 in s
s = s.replace(old2, new2, 1)
p.write_text(s)
ast.parse(s)
print("one-shot L0 dump installed")

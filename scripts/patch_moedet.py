#!/usr/bin/env python3
"""Add a deterministic MoE path (ATOM_REFERENCE_MOE_DET=1).

F.grouped_mm accumulates expert outputs with atomics, so two identical calls can
return different bits - that is exactly the run-to-run nondeterminism observed at
layer 0. The deterministic variant runs one dense GEMM per expert over its contiguous
token slice and scatters the results back, which removes the atomics entirely.
"""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """    gate_up = F.grouped_mm(sel_g, w13.transpose(-2, -1), offs=offsets)
    gate, up = gate_up.chunk(2, dim=-1)
    gated = F.silu(gate) * up
    out_g = F.grouped_mm(gated, w2.transpose(-2, -1), offs=offsets)
    out_g = out_g * w_g.unsqueeze(-1)"""
new = """    if os.environ.get("ATOM_REFERENCE_MOE_DET") == "1":
        bounds = [0] + [int(v) for v in offsets.tolist()]
        out_g = torch.empty(
            sel_g.shape[0], w2.shape[-2], dtype=sel_g.dtype, device=sel_g.device
        )
        for e in range(num_experts):
            a, b = bounds[e], bounds[e + 1]
            if b <= a:
                continue
            seg = sel_g[a:b]
            gate_up = F.linear(seg, w13[e])
            gate, up = gate_up.chunk(2, dim=-1)
            gated = F.silu(gate) * up
            out_g[a:b] = F.linear(gated, w2[e])
        out_g = out_g * w_g.unsqueeze(-1)
    else:
        gate_up = F.grouped_mm(sel_g, w13.transpose(-2, -1), offs=offsets)
        gate, up = gate_up.chunk(2, dim=-1)
        gated = F.silu(gate) * up
        out_g = F.grouped_mm(gated, w2.transpose(-2, -1), offs=offsets)
        out_g = out_g * w_g.unsqueeze(-1)"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("deterministic MoE path added")

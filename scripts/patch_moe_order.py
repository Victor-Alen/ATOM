#!/usr/bin/env python3
"""Match HF's expert-loop row order: torch.where(expert_mask[e]) yields (slot, token),
slot-major, whereas torch.where(topi == e) yields (token, slot), token-major. The
difference changes the bf16 index_add_ summation order."""
from pathlib import Path

for name in ("ref_arith_probe3.py", "moe_micro.py"):
    p = Path("/workspace/ATOM-tritonmoe/scripts") / name
    s = p.read_text()
    if "expert_mask" in s:
        print(name, "already patched")
        continue
    old = """            for e in range(cfg.num_experts):
                m = topi == e
                if not bool(m.any()):
                    continue
                ti, si = torch.where(m)"""
    new = """            expert_mask = F.one_hot(topi, num_classes=cfg.num_experts + 1).permute(2, 1, 0)
            for e in range(cfg.num_experts):
                if int(expert_mask[e].sum()) == 0:
                    continue
                si, ti = torch.where(expert_mask[e])"""
    if old in s:
        s = s.replace(old, new, 1)
        p.write_text(s)
        import ast
        ast.parse(s)
        print(name, "patched to HF slot-major order")
    else:
        old2 = """        for e in range(cfg.num_experts):
            m = topi == e
            if not bool(m.any()):
                continue
            ti, si = torch.where(m)"""
        new2 = """        em = F.one_hot(topi, num_classes=cfg.num_experts + 1).permute(2, 1, 0)
        for e in range(cfg.num_experts):
            if int(em[e].sum()) == 0:
                continue
            si, ti = torch.where(em[e])"""
        if old2 in s:
            s = s.replace(old2, new2, 1)
            p.write_text(s)
            import ast
            ast.parse(s)
            print(name, "patched (micro variant)")
        else:
            print(name, "PATTERN NOT FOUND")

from pathlib import Path
import ast

p = Path("/workspace/ATOM-tritonmoe/atom/models/mellum.py")
s = p.read_text()
anchor = '''        _prod_moe = __import__("os").environ.get("ATOM_PROD_MOE")'''
assert anchor in s
insert = '''        _wd = __import__("os").environ.get("ATOM_PROD_WDUMP")
        if _wd and "prodmoe-w" not in globals():
            try:
                globals()["prodmoe-w"] = True
                import torch as _pt

                _exp0 = self.experts
                _pt.save(
                    {
                        "w13_e0": _exp0.w13_weight[0].detach().float().cpu(),
                        "w2_e0": _exp0.w2_weight[0].detach().float().cpu(),
                        "w13_shape": _pt.tensor(_exp0.w13_weight.shape),
                        "w2_shape": _pt.tensor(_exp0.w2_weight.shape),
                    },
                    _wd,
                )
                print("[prod-wdump] saved", _wd, flush=True)
            except Exception as _e:  # noqa: BLE001
                print("[prod-wdump] failed:", repr(_e), flush=True)

'''
s = s.replace(anchor, insert + anchor, 1)
p.write_text(s)
ast.parse(s)
print("prod weight dump added")

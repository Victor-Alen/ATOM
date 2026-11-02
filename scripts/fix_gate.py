from pathlib import Path
import ast

p = Path("/workspace/ATOM-tritonmoe/atom/models/mellum.py")
s = p.read_text()
old = '''                final_hidden_states = _moe_grouped(
                    _w13, _w2, self.gate.weight, hidden_states, _ne, _tk, _norm
                )'''
new = '''                _gw = self.gate.weight
                if _gw.shape[0] != _ne:
                    _gw = _gw.t()
                final_hidden_states = _moe_grouped(
                    _w13, _w2, _gw, hidden_states, _ne, _tk, _norm
                )'''
assert old in s, "anchor missing"
p.write_text(s.replace(old, new, 1))
ast.parse(p.read_text())
print("gate weight orientation handled")

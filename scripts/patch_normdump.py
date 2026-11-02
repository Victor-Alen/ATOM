from pathlib import Path
import ast

p = Path("/workspace/ATOM-tritonmoe/atom/models/mellum.py")
s = p.read_text()
old = '''        else:
            # Add qk-norm (per-head)
            q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
                -1, self.num_heads * self.head_dim
            )'''
new = '''        else:
            _nq = __import__("os").environ.get("ATOM_NORM_DUMP")
            if _nq and getattr(self, "layer_num", 0) == 0 and not globals().get("_norm_dumped"):
                globals()["_norm_dumped"] = True
                __import__("torch").save(
                    {
                        "q_raw": q.detach().float().cpu(),
                        "k_raw": k.detach().float().cpu(),
                        "q_normed": self.q_norm(
                            q.view(-1, self.num_heads, self.head_dim)
                        ).detach().float().cpu(),
                        "k_normed": self.k_norm(
                            k.view(-1, self.num_kv_heads, self.head_dim)
                        ).detach().float().cpu(),
                        "eps": __import__("torch").tensor(float(self.q_norm.eps)),
                    },
                    _nq,
                )
                print("[norm-dump] saved", _nq, flush=True)

            # Add qk-norm (per-head)
            q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
                -1, self.num_heads * self.head_dim
            )'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("norm dump added")

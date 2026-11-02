from pathlib import Path
import ast

p = Path("/workspace/ATOM-tritonmoe/atom/model_ops/attention_mha.py")
s = p.read_text()
old = '''        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)'''
new = '''        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        _dp0 = __import__("os").environ.get("ATOM_ROPE_DUMP")
        if _dp0 and self.layer_num == 0 and globals().get("_rope_dumped") and not globals().get("_rope_pre_dumped"):
            globals()["_rope_pre_dumped"] = True
            __import__("torch").save(
                {"q_pre": q.detach().float().cpu(), "k_pre": k.detach().float().cpu()},
                _dp0 + ".pre",
            )
            print("[rope-dump] pre saved", flush=True)'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("pre-norm dump added")

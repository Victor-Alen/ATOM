from pathlib import Path
import ast

p = Path("/workspace/ATOM-tritonmoe/atom/model_ops/attention_mha.py")
s = p.read_text()

old = '''            # v_cache is [nb, nkv, hd, bs] in the shuffle layout
            v_layout_tail = v_cache.dim() == 4 and v_cache.shape[-1] == bs'''
new = '''            # v_cache may be [nb, nkv, hd, bs] (shuffled) or [nb, nkv, bs, hd] (plain)
            v_layout_tail = v_cache.dim() == 4 and v_cache.shape[-1] == bs
            v_layout_plain = v_cache.dim() == 4 and v_cache.shape[-2] == bs'''
assert old in s
s = s.replace(old, new, 1)

old2 = '''                if v_layout_tail:
                    vv = vsel.permute(0, 3, 1, 2).reshape(-1, nKV, D)[:n_ctx]
                else:
                    vv = vsel.permute(0, 2, 4, 1, 3).reshape(-1, nKV, D)[:n_ctx]'''
new2 = '''                if v_layout_tail:
                    vv = vsel.permute(0, 3, 1, 2).reshape(-1, nKV, D)[:n_ctx]
                elif v_layout_plain:
                    vv = vsel.permute(0, 2, 1, 3).reshape(-1, nKV, D)[:n_ctx]
                else:
                    vv = vsel.permute(0, 2, 4, 1, 3).reshape(-1, nKV, D)[:n_ctx]'''
assert old2 in s
s = s.replace(old2, new2, 1)
p.write_text(s)
ast.parse(s)
print("v layout variants handled")

from pathlib import Path
import ast

p = Path("/workspace/ATOM-tritonmoe/atom/model_ops/attention_mha.py")
s = p.read_text()

old = '''        if k5:
            _nb, _nkv, _bsh, _hd, _x = k_cache.shape
            bs = _bsh * _x
            # v_cache may be [nb, nkv, hd, bs] (shuffled) or [nb, nkv, bs, hd] (plain)
            v_layout_tail = v_cache.dim() == 4 and v_cache.shape[-1] == bs
            v_layout_plain = v_cache.dim() == 4 and v_cache.shape[-2] == bs'''
new = '''        if k5:
            # measured layout: [nb, nkv, hd//x, bs, x]   (x = 16 // element_size)
            _nb, _nkv, _hdx, _bs, _x = k_cache.shape
            bs = _bs
            # v_cache is [nb, nkv, hd, bs]
            v_layout_tail = v_cache.dim() == 4 and v_cache.shape[-1] == bs
            v_layout_plain = v_cache.dim() == 4 and v_cache.shape[-2] == bs'''
assert old in s
s = s.replace(old, new, 1)

old2 = '''                ksel = k_cache.index_select(0, blocks)
                kk = ksel.permute(0, 2, 4, 1, 3).reshape(-1, nKV, D)[:n_ctx]'''
new2 = '''                ksel = k_cache.index_select(0, blocks)
                kk = ksel.permute(0, 3, 1, 2, 4).reshape(-1, nKV, D)[:n_ctx]'''
assert old2 in s
s = s.replace(old2, new2, 1)

s = s.replace('''                else:
                    vv = vsel.permute(0, 2, 4, 1, 3).reshape(-1, nKV, D)[:n_ctx]''',
'''                else:
                    vv = vsel.permute(0, 3, 1, 2).reshape(-1, nKV, D)[:n_ctx]''')
p.write_text(s)
ast.parse(s)
print("k layout corrected to [nb,nkv,hd//x,bs,x]")

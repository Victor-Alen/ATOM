from pathlib import Path
import ast

p = Path("/workspace/ATOM-tritonmoe/atom/model_ops/attention_mha.py")
s = p.read_text()

old = '''        nH = self.num_heads
        nKV = self.num_kv_heads
        D = self.head_dim
        rep = nH // nKV
        bs = k_cache.shape[-2] if k_cache.dim() == 4 else k_cache.shape[3]

        # k_cache: [nb, nkv, bs, D]  (or 5-D shuffled; fall back to the kernel then)
        if k_cache.dim() != 4:
            raise RuntimeError("eager_attention_hf expects 4-D k_cache")
'''
new = '''        nH = self.num_heads
        nKV = self.num_kv_heads
        D = self.head_dim
        rep = nH // nKV

        k5 = k_cache.dim() == 5
        if k5:
            _nb, _nkv, _bsh, _hd, _x = k_cache.shape
            bs = _bsh * _x
            # v_cache is [nb, nkv, hd, bs] in the shuffle layout
            v_layout_tail = v_cache.dim() == 4 and v_cache.shape[-1] == bs
        else:
            bs = k_cache.shape[-2]
            v_layout_tail = False
'''
assert old in s
s = s.replace(old, new, 1)

old2 = '''            blocks = bt[b, :n_blk].long()
            kk = k_cache[:, :, :, :].index_select(0, blocks)
            vv = v_cache[:, :, :, :].index_select(0, blocks)
            kk = kk.permute(0, 2, 1, 3).reshape(-1, nKV, D)[:n_ctx]
            vv = vv.permute(0, 2, 1, 3).reshape(-1, nKV, D)[:n_ctx]
'''
new2 = '''            blocks = bt[b, :n_blk].long()
            if k5:
                ksel = k_cache.index_select(0, blocks)
                kk = ksel.permute(0, 2, 4, 1, 3).reshape(-1, nKV, D)[:n_ctx]
                vsel = v_cache.index_select(0, blocks)
                if v_layout_tail:
                    vv = vsel.permute(0, 3, 1, 2).reshape(-1, nKV, D)[:n_ctx]
                else:
                    vv = vsel.permute(0, 2, 4, 1, 3).reshape(-1, nKV, D)[:n_ctx]
            else:
                ksel = k_cache.index_select(0, blocks)
                vsel = v_cache.index_select(0, blocks)
                kk = ksel.permute(0, 2, 1, 3).reshape(-1, nKV, D)[:n_ctx]
                vv = vsel.permute(0, 2, 1, 3).reshape(-1, nKV, D)[:n_ctx]
'''
assert old2 in s
s = s.replace(old2, new2, 1)
p.write_text(s)
ast.parse(s)
print("5-D shuffle layout handled")

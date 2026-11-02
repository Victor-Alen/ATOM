from pathlib import Path
import ast

p = Path("/workspace/ATOM-tritonmoe/atom/model_ops/attention_mha.py")
s = p.read_text()

anchor = """    def dispatch_backend(self, fwd_ctx: ForwardContext):

        ctx = fwd_ctx.context

        if ctx.is_prefill:"""
assert anchor in s

new_fn = '''    def eager_attention_hf(self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx):
        """HF-ordered attention: one full-row fp32 softmax per query row.

        Gated by ATOM_EAGER_ATTN=1; used to reproduce HF's numbers exactly instead
        of the online-softmax tiling used by the fused kernels.
        """
        import torch as _t

        am = fwd_ctx.attn_metadata
        ctx_lens = am.context_lens
        bt = am.block_tables
        cu = am.cu_seqlens_q if getattr(am, "cu_seqlens_q", None) is not None else None
        if cu is None:
            cu = _t.tensor([0, q.shape[0]], dtype=_t.int32, device=q.device)

        nH = self.num_heads
        nKV = self.num_kv_heads
        D = self.head_dim
        rep = nH // nKV
        bs = k_cache.shape[-2] if k_cache.dim() == 4 else k_cache.shape[3]

        # k_cache: [nb, nkv, bs, D]  (or 5-D shuffled; fall back to the kernel then)
        if k_cache.dim() != 4:
            raise RuntimeError("eager_attention_hf expects 4-D k_cache")

        o = _t.empty_like(q)
        W = self.sliding_window if self.sliding_window and self.sliding_window > 0 else None
        q3 = q.view(-1, nH, D)

        for b in range(ctx_lens.shape[0]):
            q0 = int(cu[b])
            q1 = int(cu[b + 1])
            if q1 <= q0:
                continue
            n_ctx = int(ctx_lens[b])
            n_q = q1 - q0
            n_blk = (n_ctx + bs - 1) // bs
            blocks = bt[b, :n_blk].long()
            kk = k_cache[:, :, :, :].index_select(0, blocks)
            vv = v_cache[:, :, :, :].index_select(0, blocks)
            kk = kk.permute(0, 2, 1, 3).reshape(-1, nKV, D)[:n_ctx]
            vv = vv.permute(0, 2, 1, 3).reshape(-1, nKV, D)[:n_ctx]

            qb = q3[q0:q1].float()                              # [n_q, nH, D]
            kb = kk.float().repeat_interleave(rep, dim=1)       # [n_ctx, nH, D]
            vb = vv.float().repeat_interleave(rep, dim=1)

            qi = _t.arange(n_ctx - n_q, n_ctx, device=q.device).unsqueeze(1)   # [n_q,1]
            ki = _t.arange(n_ctx, device=q.device).unsqueeze(0)                # [1,n_ctx]
            mask = ki <= qi
            if W is not None:
                mask = mask & ((qi - ki) < W)

            sc = _t.einsum("qhd,khd->hqk", qb * self.scale, kb)   # [nH, n_q, n_ctx]
            sc = sc.masked_fill(~mask.unsqueeze(0), float("-inf"))
            p = _t.softmax(sc, dim=-1)
            out = _t.einsum("hqk,khd->qhd", p, vb)               # [n_q, nH, D]
            o[q0:q1] = out.to(o.dtype)

        return o

'''

s = s.replace(anchor, new_fn + anchor, 1)

# route in dispatch_backend when the flag is set
old_disp = """        ctx = fwd_ctx.context

        if ctx.is_prefill:"""
new_disp = """        ctx = fwd_ctx.context

        if __import__("os").environ.get("ATOM_EAGER_ATTN") == "1":
            return self.eager_attention_hf

        if ctx.is_prefill:"""
assert old_disp in s
s = s.replace(old_disp, new_disp, 1)

p.write_text(s)
ast.parse(s)
print("eager_attention_hf added and wired behind ATOM_EAGER_ATTN=1")

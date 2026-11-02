from pathlib import Path
import ast

p = Path("/workspace/ATOM-tritonmoe/atom/model_ops/attention_mha.py")
s = p.read_text()

# remove the old pre-dump block
old_pre = '''        _dp0 = __import__("os").environ.get("ATOM_ROPE_DUMP")
        if _dp0 and self.layer_num == 0 and globals().get("_rope_dumped") and not globals().get("_rope_pre_dumped"):
            globals()["_rope_pre_dumped"] = True
            __import__("torch").save(
                {"q_pre": q.detach().float().cpu(), "k_pre": k.detach().float().cpu()},
                _dp0 + ".pre",
            )
            print("[rope-dump] pre saved", flush=True)'''
assert old_pre in s
s = s.replace(old_pre, "", 1)

# stash pre tensors on the instance so the post block can save both together
old_call = '''        q, k, v, k_cache, v_cache, k_scale, v_scale = self.rope_cache(
            q, k, v, qkv, position, fwd_ctx
        )'''
new_call = '''        _dp0 = __import__("os").environ.get("ATOM_ROPE_DUMP")
        _save_here = bool(_dp0) and self.layer_num == 0 and not globals().get("_rope_dumped")
        if _save_here:
            self._pre_q = q.detach().float().cpu()
            self._pre_k = k.detach().float().cpu()
            self._pre_pos = position.detach().cpu() if position is not None else None

        q, k, v, k_cache, v_cache, k_scale, v_scale = self.rope_cache(
            q, k, v, qkv, position, fwd_ctx
        )'''
assert old_call in s
s = s.replace(old_call, new_call, 1)

old_post = '''        _dp = __import__("os").environ.get("ATOM_ROPE_DUMP")
        if _dp and self.layer_num == 0 and not globals().get("_rope_dumped"):'''
new_post = '''        _dp = __import__("os").environ.get("ATOM_ROPE_DUMP")
        if _save_here:'''
assert old_post in s
s = s.replace(old_post, new_post, 1)

old_save = '''                {
                    "pos": position.detach().cpu(),'''
new_save = '''                {
                    "q_pre": getattr(self, "_pre_q", None),
                    "k_pre": getattr(self, "_pre_k", None),
                    "pos": position.detach().cpu(),'''
assert old_save in s
s = s.replace(old_save, new_save, 1)
p.write_text(s)
ast.parse(s)
print("same-call pre+post dump")

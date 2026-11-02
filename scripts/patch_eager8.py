from pathlib import Path
import ast

p = Path("/workspace/ATOM-tritonmoe/atom/model_ops/attention_mha.py")
s = p.read_text()
old = '''            if not globals().get("_eager_diff_once"):'''
new = '''            _tag = "decode" if not fwd_ctx.context.is_prefill else "prefill"
            if not globals().get("_eager_diff_" + _tag):'''
assert old in s
s = s.replace(old, new, 1)
old2 = '''                globals()["_eager_diff_once"] = True'''
new2 = '''                globals()["_eager_diff_" + _tag] = True'''
assert old2 in s
s = s.replace(old2, new2, 1)
old3 = '''                    "[eager] max|eager-fused| = %.6g  mean = %.6g  shapes %s / %s"
                    % (float(d.max()), float(d.mean()), tuple(o.shape), tuple(ref.shape)),'''
new3 = '''                    "[eager] %s max|eager-fused| = %.6g  mean = %.6g"
                    % (_tag, float(d.max()), float(d.mean())),'''
assert old3 in s
s = s.replace(old3, new3, 1)
p.write_text(s)
ast.parse(s)
print("per-phase diff diagnostic")

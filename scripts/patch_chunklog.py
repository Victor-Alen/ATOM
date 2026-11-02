#!/usr/bin/env python3
"""Instrument the reference path for chunked-prefill diagnosis:
- log each forward call (chunk) with its absolute position range and ctx
- maintain a real per-layer KV 'written' bitmap from actual writes
- report, per chunk, how many *visible* keys are unwritten
- store per-chunk, per-layer last-row outputs so nothing is overwritten"""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()

# 1) globals
anchor = "_HIST: list = []"
glob = '''_HIST: list = []
_CHUNK: dict = {}
_WRITTEN: dict = {}
_LCHUNK: dict = {}

'''
if "_WRITTEN: dict = {}" not in s:
    s = s.replace(anchor, glob.strip("\n") + "\n", 1)

# 2) chunk logging + written bitmap at the KV write site
old2 = """            k_flat[write_slots] = k
            v_flat[write_slots] = v"""
new2 = """            k_flat[write_slots] = k
            v_flat[write_slots] = v
            _wr = _WRITTEN.get(li)
            if _wr is None or _wr.numel() < k_flat.shape[0]:
                _wr = torch.zeros(int(k_flat.shape[0]), dtype=torch.bool, device=dev)
                _WRITTEN[li] = _wr
            _wr[write_slots] = True
            if li == 0 and s == 0:
                _n = int(_LCHUNK.get("n", 0)) + 1
                _LCHUNK["n"] = _n
                _CHUNK["n"] = _n
                _CHUNK["pos"] = (int(new_pos.min().item()), int(new_pos.max().item()))
                _CHUNK["ctx"] = ctx
                _CHUNK["bs"] = bs
                if os.environ.get("ATOM_REFERENCE_CHUNKLOG") == "1":
                    print(
                        f"[chunk] n={_n} bs={bs} pos={_CHUNK['pos']} ctx={ctx} "
                        f"slots={int(len(ctx_slots[s]))}",
                        flush=True,
                    )"""
assert old2 in s
s = s.replace(old2, new2, 1)

# 3) visible-but-unwritten check, right after the mask is built
old3 = """                mask = ki <= qi
                if lt == "sliding_attention" and W is not None:
                    mask = mask & ((qi - ki) < W)"""
new3 = """                mask = ki <= qi
                if lt == "sliding_attention" and W is not None:
                    mask = mask & ((qi - ki) < W)
                if os.environ.get("ATOM_REFERENCE_KVCHK") == "1":
                    _wr2 = _WRITTEN.get(li)
                    if _wr2 is not None:
                        _vis = mask[-1]
                        _w = _wr2[: _vis.numel()]
                        _unw = int((_vis & ~_w).sum().item())
                        print(
                            f"[kvchk] L{li} chunk={_CHUNK.get('n')} q={int(qi[-1].item())} "
                            f"visible={int(_vis.sum().item())} unwritten_visible={_unw}",
                            flush=True,
                        )"""
assert old3 in s
s = s.replace(old3, new3, 1)

# 4) per-chunk per-layer last-row dumps
old4 = """        if _dump_call:
            _LDUMP[f"blk_{li}"] = x[-1].detach().float().cpu()"""
new4 = """        if _dump_call:
            _LDUMP[f"blk_{li}"] = x[-1].detach().float().cpu()
            _LDUMP[f"c{_CHUNK.get('n')}_blk_{li}"] = x[-1].detach().float().cpu()"""
assert old4 in s
s = s.replace(old4, new4, 1)
p.write_text(s)
ast.parse(s)
print("chunk/KV instrumentation added")

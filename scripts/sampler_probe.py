#!/usr/bin/env python3
"""Probe 4: is the sampler a true argmax at temperature 0, or a random race?"""
import torch
from atom.model_ops.sampler import Sampler, get_per_token_exponential

DEV = "cuda"
SAMPLER_EPS = 1e-10
s = Sampler().to(DEV).eval()

def run(logits, T, tag, n=20, top_ks=None, top_ps=None, all_greedy=True):
    temps = torch.clamp(T, min=SAMPLER_EPS).to(DEV)
    outs = []
    with torch.no_grad():
        for _ in range(n):
            o = s(logits, temps, top_ks, top_ps, all_greedy)
            outs.append(tuple(o.tolist()))
    uniq = {}
    for o in outs:
        uniq[o] = uniq.get(o, 0) + 1
    am = tuple(logits.argmax(dim=-1).tolist())
    print(f"{tag}: argmax={am} distinct={len(uniq)} counts={sorted(uniq.values(), reverse=True)[:5]} "
          f"modal_is_argmax={max(uniq, key=uniq.get) == am}", flush=True)
    return outs

torch.manual_seed(0)
V = 4096
# 1) random bf16 logits, no ties expected
lg = torch.randn(1, V, dtype=torch.bfloat16, device=DEV) * 5.0
run(lg, torch.tensor([0.0]), "random logits, T=0")

top = float(lg.max()) + 50.0
# 2) exact tie FOR THE MAXIMUM (what bf16 quantization creates all the time)
lg2 = lg.clone()
lg2[0, 7] = top
lg2[0, 1234] = top
run(lg2, torch.tensor([0.0]), "exact top-2 tie (both = max), T=0")

# 3) near tie: 1 bf16 step apart at that magnitude
lg3 = lg2.clone()
lg3[0, 1234] = torch.nextafter(lg2[0, 7], torch.tensor(0.0, dtype=torch.bfloat16, device=DEV))
run(lg3, torch.tensor([0.0]), "1-ulp gap below max, T=0")

# 4) clear gap at the top
lg4 = lg.clone()
lg4[0, 7] = top
lg4[0, 1234] = top - 2.0
run(lg4, torch.tensor([0.0]), "gap 2.0 at top, T=0")

# 5) batch of 4 like a real decode step, all greedy, exact ties
lg5 = lg2.repeat(4, 1)
run(lg5, torch.zeros(4), "batch 4, exact tie, T=0")

# 6) same but through the top-k filtering branch (the branch that honours all_greedy)
run(lg2, torch.tensor([0.0]), "exact tie, T=0, top_k filtering path",
    top_ks=torch.tensor([40], dtype=torch.int32, device=DEV), top_ps=torch.tensor([1.0], device=DEV))

# 8) three-way tie
lg8 = lg2.clone()
lg8[0, 4000] = top
run(lg8, torch.tensor([0.0]), "3-way exact tie at max, T=0")

# 7) how often does a fresh exponential row change?
e1 = get_per_token_exponential(64, DEV)
e2 = get_per_token_exponential(64, DEV)
print("exponential rows identical across calls:", bool(torch.equal(e1, e2)), flush=True)

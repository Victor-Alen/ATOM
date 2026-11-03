#!/usr/bin/env python3
"""Repeat the production helper under explicitly selected CK rounding modes."""
import json
import os
import torch
from aiter import ActivationType
from atom.model_ops.mellum_ordered_moe import mellum_ordered_moe
from probe_moe_fixed_router import CAPTURE, load_weights
from verify_mellum_ordered_moe import digest, EXPECTED

cap = torch.load(CAPTURE, map_location="cpu", weights_only=False)
w13, w2 = load_weights()
for name, sl in (("decode_1", slice(-1, None)), ("prefill_866", slice(None))):
    x = cap["h2"][sl].cuda()
    ids = cap["sel"][sl].to(device="cuda", dtype=torch.int32)
    weights = cap["rw"][sl].to(device="cuda", dtype=torch.float32)
    hashes=[]
    for _ in range(40):
        y=mellum_ordered_moe(x,w13,w2,weights,ids,ActivationType.Silu)
        hashes.append(digest(y.cpu()))
    assert len(set(hashes)) == 1, hashes
    if os.environ.get("CK_MELLUM_STAGE1_MODE", "0") == "0" and os.environ.get("CK_MELLUM_STAGE2_ROUND_WEIGHT", "0") == "0":
        assert hashes[0] == EXPECTED[name], (name, hashes[0])
    print(json.dumps({"case":name,"repeats":40,"distinct":len(set(hashes)),"hash":hashes[0]}),flush=True)

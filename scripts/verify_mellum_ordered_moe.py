#!/usr/bin/env python3
"""Check production ordered MoE helper on captured layer-0 tensors."""
import hashlib
import json

import torch
from aiter import ActivationType

from atom.model_ops.mellum_ordered_moe import mellum_ordered_moe
from probe_moe_fixed_router import CAPTURE, load_weights

EXPECTED = {"decode_1": "583fec11fe6764c9", "prefill_866": "e0699e24c2f5dcb2"}


def digest(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def main():
    cap = torch.load(CAPTURE, map_location="cpu", weights_only=False)
    w13, w2 = load_weights()
    results = []
    for name, sl in (("decode_1", slice(-1, None)), ("prefill_866", slice(None))):
        x = cap["h2"][sl].cuda()
        ids = cap["sel"][sl].to(device="cuda", dtype=torch.int32)
        weights = cap["rw"][sl].to(device="cuda", dtype=torch.float32)
        hashes = []
        for _ in range(40):
            result = mellum_ordered_moe(x, w13, w2, weights, ids, activation=ActivationType.Silu)
            torch.cuda.synchronize()
            hashes.append(digest(result.cpu()))
        assert len(set(hashes)) == 1, hashes
        assert hashes[0] == EXPECTED[name], (name, hashes[0], EXPECTED[name])
        row = {"case": name, "distinct": len(set(hashes)), "hash": hashes[0]}
        print(json.dumps(row), flush=True)
        results.append(row)
    return results


if __name__ == "__main__":
    main()

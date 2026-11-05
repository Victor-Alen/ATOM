"""Native CK prefill MoE with independently gated stage1 and stage2.

Stage2 defaults to the accepted HF grouped projection; the opt-in native path
uses GPU profile plans and CK kernels validated separately before model gates.
"""
import os
import torch
from aiter.ops.shuffle import shuffle_weight
from transformers.integrations.moe import _grouped_linear
_ROUTE_VARIANT = os.environ.get("ATOM_MELLUM_PREFILL_ROUTE_VARIANT", "original")
if _ROUTE_VARIANT == "hybrid":
    _GROUP0_TILE = os.environ.get("ATOM_MELLUM_PREFILL_GROUP0_TILE", "")
    if _GROUP0_TILE == "m32n64k64v1":
        from atom.model_ops.mellum_native_prefill_rotv1m32 import Stage1
    elif not _GROUP0_TILE:
        from atom.model_ops.mellum_native_prefill_stage1_hybrid import Stage1
    else:
        raise ValueError(f"Unvalidated native prefill group0 tile: {_GROUP0_TILE}")
elif _ROUTE_VARIANT == "fused":
    from atom.model_ops.mellum_native_prefill_stage1_fused import Stage1
elif _ROUTE_VARIANT == "original":
    from atom.model_ops.mellum_native_prefill_stage1 import Stage1
else:
    raise ValueError(f"Unknown native prefill route variant: {_ROUTE_VARIANT}")


def experts(layer, reference, x, ids, weights):
    if os.getenv("CK_MELLUM_STAGE1_MODE") != "1" or os.getenv("CK_MELLUM_STAGE1_KORDER_PLAN", "0") != "0":
        raise RuntimeError("Native prefill stage1 requires mode=1 and KORDER_PLAN=0; metadata is carried on GPU")
    assert x.dtype == weights.dtype == torch.bfloat16
    assert ids.shape == (x.shape[0], 8) and x.shape[1] == 2304
    candidate = getattr(layer, "_mellum_native_prefill_stage1", None)
    if candidate is None:
        candidate = Stage1(layer.w13_weight, shuffle_weight(reference.gate_up_proj, layout=(32, 8)), layer.w2_weight)
        layer._mellum_native_prefill_stage1 = candidate
    m = x.shape[0]
    if m == 0:
        return torch.empty_like(x)
    assert m <= 16384, "Native prefill profile table currently covers 1..16384 rows per request"
    gated = torch.empty((m, 8, 896), device=x.device, dtype=x.dtype)
    shared = None
    ids_int = ids.int()
    if os.getenv("ATOM_MELLUM_SHARE_PREFILL_ROUTES") == "1":
        if (os.getenv("ATOM_MELLUM_NATIVE_PREFILL_STAGE2") != "1"
                or _ROUTE_VARIANT != "hybrid"
                or _GROUP0_TILE != "m32n64k64v1"):
            raise RuntimeError("Shared prefill routes require validated native stage1 and stage2")
        from aiter.fused_moe import moe_sorting
        from atom.model_ops.mellum_native_prefill_hybrid_core import expert_counts
        si, sw, _, nv, _ = moe_sorting(ids_int, weights.float(), 64, 2304, x.dtype, 64)
        shared = (si, sw, nv, expert_counts(ids_int))
    plan = (candidate.prepare(ids_int, weights, shared) if shared is not None
            else candidate.prepare(ids_int, weights))
    candidate.run(x, plan, gated)
    if os.getenv("ATOM_MELLUM_NATIVE_PREFILL_STAGE2") == "1":
        from atom.model_ops.mellum_native_prefill_down import Stage2
        stage2 = getattr(layer, "_mellum_native_prefill_stage2", None)
        if stage2 is None:
            stage2 = Stage2(layer.w13_weight, layer.w2_weight,
                            os.environ["ATOM_MELLUM_PREFILL_NATIVE_ROOT"],
                            raw_w2=reference.down_proj)
            layer._mellum_native_prefill_stage2 = stage2
        weighted_slots = torch.empty((m * 8, 2304), device=x.device, dtype=x.dtype)
        stage2.run(gated, stage2.prepare(ids_int, weights, shared), weighted_slots)
        return weighted_slots.view(m, 8, 2304).sum(dim=1)
    expert_ids = ids.reshape(-1)
    perm = torch.argsort(expert_ids)
    inv_perm = torch.argsort(perm)
    counts = torch.histc(expert_ids[perm].int(), bins=64, min=0, max=63)
    offsets = torch.cumsum(counts, 0, dtype=torch.int32)
    down = _grouped_linear(gated.reshape(-1, 896)[perm], reference.down_proj, offs=offsets)
    weighted = down * weights.reshape(-1)[perm, None]
    return weighted[inv_perm].view(m, 8, 2304).sum(dim=1)

"""GPU-only CK K-order metadata for Mellum BF16 gate/up GEMMs.

The table contains hipBLASLt dispatch choices, not weights or reference values.
Both Gate and Up use the same expert-row-count-dependent MFMA grouping.
"""
import functools
import hashlib
import json
import os
from pathlib import Path

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _count_expert_rows(IDS, COUNTS, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ids = tl.load(IDS + offsets, offsets < N, other=64)
    counts = tl.histogram(ids.to(tl.int32), 64, mask=(offsets < N) & (ids >= 0) & (ids < 64))
    tl.atomic_add(COUNTS + tl.arange(0, 64), counts, sem="relaxed")


@tr.jit
def _build_plan(COUNTS, TABLE, NUM_VALID, PLAN):
    expert = tl.arange(0, 64)
    rows = tl.load(COUNTS + expert)
    flags = tl.load(TABLE + rows)
    tl.store(PLAN + 2 + expert, flags)
    prefix = tl.arange(0, 2)
    tl.store(PLAN + prefix, tl.load(NUM_VALID + prefix))


@functools.lru_cache(maxsize=8)
def _profile_table(device_index):
    path = Path(__file__).with_name("mellum_hf_gateup_profiles.json")
    data = json.loads(path.read_text())
    if torch.cuda.get_device_properties(device_index).gcnArchName.split(":")[0] != data["arch"]:
        raise RuntimeError("CK HF K-order profile table does not match GPU architecture")
    if data.get("ck_stage1_metadata_abi") != 1:
        raise RuntimeError("Rebuild CK to seal its K-order metadata ABI and binary fingerprint")
    binary = Path(os.environ["AITER_JIT_DIR"]) / (
        "module_moe_ck2stages_b16_b16_preshuffle_on_b16_silu_no_mulWeightStage2.so")
    if hashlib.sha256(binary.read_bytes()).hexdigest() != data["ck_binary_sha256"]:
        raise RuntimeError("CK HF K-order metadata does not match the loaded MoE binary")
    library = Path(data["library"])
    if hashlib.sha256(library.read_bytes()).hexdigest() != data["library_sha256"]:
        raise RuntimeError("CK HF K-order profile table does not match installed hipBLASLt")
    if os.getenv("HIPBLASLT_TENSILE_LIBPATH") or os.getenv("HIPBLASLT_TUNING_OVERRIDE_FILE"):
        raise RuntimeError("Regenerate CK HF K-order profiles for overridden hipBLASLt libraries")
    return torch.tensor(data["mi32_flags"], dtype=torch.int32,
                        device=torch.device("cuda", device_index))


def ck_hf_korder_plan(topk_ids, num_valid):
    """Extend the existing two-int sorting header with 64 per-expert flags."""
    if topk_ids.ndim != 2 or topk_ids.shape[1] != 8 or topk_ids.shape[0] > 32768:
        raise ValueError("CK HF K-order metadata requires Mellum top8 and <=32768 tokens")
    if os.getenv("CK_MELLUM_STAGE1_KORDER_PLAN") != "1":
        raise RuntimeError("CK HF K-order metadata requires matching CK binary mode")
    if num_valid.numel() != 2 or not topk_ids.is_contiguous():
        raise ValueError("Unexpected CK sorting metadata layout")
    table = _profile_table(topk_ids.device.index)
    counts = torch.zeros(64, dtype=torch.int32, device=topk_ids.device)
    plan = torch.empty(66, dtype=torch.int32, device=topk_ids.device)
    _count_expert_rows[(tr.cdiv(topk_ids.numel(), 1024),)](
        topk_ids, counts, topk_ids.numel(), 1024)
    _build_plan[(1,)](counts, table, num_valid, plan)
    return plan

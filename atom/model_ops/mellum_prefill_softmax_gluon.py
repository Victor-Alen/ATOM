"""Experimental prefill scale/mask/softmax fusion based on accepted Gluon decode reduction.
Original QK/PV GEMMs unchanged. Enabled only by the opt-in hybrid dispatch.
"""
import torch
import triton as tr
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.extra.hip import libdevice as hip_libdevice

@gluon.jit
def _large_exp_ftz(x):
    # ATen large-row fast exp flushes only exponential underflow. Keep the
    # subsequent normalization product in IEEE mode, including subnormals.
    e = tl.exp(x)
    return tl.where(e < 1.1754943508222875e-38, 0.0, e)

@gluon.jit
def _load_scaled_masked(scores, offset, pos, valid, qpos, key_start, WINDOW: gl.constexpr, SCALE: gl.constexpr):
    allowed = valid & (pos >= 0) & (pos + key_start <= qpos)
    if WINDOW > 0:
        allowed = allowed & (pos + key_start > qpos - WINDOW)
    raw = gl.load(scores + offset + pos, allowed, other=0).to(gl.float32)
    scaled = (raw * SCALE).to(gl.bfloat16).to(gl.float32)
    return gl.where(allowed, scaled, float('-inf'))

@gluon.jit
def _prefill_softmax_streaming(scores, probs, n, query_count, context_length, WINDOW: gl.constexpr, SCALE: gl.constexpr, SMALL: gl.constexpr):
    row = gl.program_id(0)
    offset = row.to(gl.int64) * n
    qpos = row % query_count + context_length - query_count
    key_start = context_length - n
    active_start = gl.maximum(0, qpos - key_start - WINDOW + 1) if WINDOW > 0 else 0
    active_end = gl.minimum(n, qpos - key_start + 1)
    if SMALL:
        lanes = gl.arange(0, 64, layout=gl.BlockedLayout([1], [64], [1], [0]))
        local_max = gl.full([64], float('-inf'), gl.float32, gl.BlockedLayout([1], [64], [1], [0]))
        for block in range(active_start // 64, gl.cdiv(active_end, 64)):
            pos = block * 64 + lanes
            x = _load_scaled_masked(scores, offset, pos, pos < n, qpos, key_start, WINDOW, SCALE)
            local_max = gl.maximum(local_max, x)
        mx = gl.max(local_max, 0)
        acc = gl.full([64], 0.0, gl.float32, gl.BlockedLayout([1], [64], [1], [0]))
        for block in range(active_start // 64, gl.cdiv(active_end, 64)):
            pos = block * 64 + lanes
            x = _load_scaled_masked(scores, offset, pos, pos < n, qpos, key_start, WINDOW, SCALE)
            acc += hip_libdevice.exp(x - mx)
        for step in gl.static_range(6):
            acc += gl.gather(acc, lanes ^ (32 >> step), 0)
        denom = gl.sum(gl.where(lanes == 0, acc, 0.0), 0)
        for block in range(gl.cdiv(n, 64)):
            pos = block * 64 + lanes
            if block * 64 < active_end and (block + 1) * 64 > active_start:
                x = _load_scaled_masked(scores, offset, pos, pos < n, qpos, key_start, WINDOW, SCALE)
                p = gl.div_rn(hip_libdevice.exp(x - mx), denom)
                gl.store(probs + offset + pos, p.to(gl.bfloat16), pos < n)
            else:
                gl.store(probs + offset + pos, 0.0, pos < n)
    else:
        large_lanes = gl.arange(0, 512, layout=gl.BlockedLayout([1], [64], [8], [0]))
        large_local_max = gl.full([512], float('-inf'), gl.float32, gl.BlockedLayout([1], [64], [8], [0]))
        for large_block in range(active_start // 512, gl.cdiv(active_end, 512)):
            large_pos = large_block * 512 + large_lanes
            large_x = _load_scaled_masked(scores, offset, large_pos, large_pos < n, qpos, key_start, WINDOW, SCALE)
            large_local_max = gl.maximum(large_local_max, large_x)
        large_mx = gl.max(large_local_max, 0)
        large_shift = ((row % 4) * (n % 4)) % 4
        large_start = gl.where(large_shift > 0, 512 - large_shift, 0)
        large_bulk = gl.where(n % 4 == 0, gl.cdiv(n, 2048) * 2048, ((n - large_start) // 2048) * 2048)
        large_prefix = large_lanes - large_shift
        large_x = _load_scaled_masked(scores, offset, large_prefix, (large_shift > 0) & (large_prefix >= 0) & (large_prefix < n), qpos, key_start, WINDOW, SCALE)
        large_acc = _large_exp_ftz(large_x - large_mx)
        for large_block in range(gl.maximum(0, (active_start - large_start) // 2048), gl.maximum(0, gl.cdiv(active_end - large_start, 2048))):
            for large_j in gl.static_range(4):
                large_pos = large_start + large_block * 2048 + large_lanes * 4 + large_j
                large_x = _load_scaled_masked(scores, offset, large_pos, (large_block * 2048 < large_bulk) & (large_pos < n), qpos, key_start, WINDOW, SCALE)
                large_acc += _large_exp_ftz(large_x - large_mx)
        for large_j in gl.static_range(4):
            large_pos = large_start + large_bulk + large_lanes + large_j * 512
            large_x = _load_scaled_masked(scores, offset, large_pos, large_pos < n, qpos, key_start, WINDOW, SCALE)
            large_acc += _large_exp_ftz(large_x - large_mx)
        for large_step in gl.static_range(6):
            large_src = large_lanes // 64 * 64 + gl.minimum(large_lanes % 64 + (32 >> large_step), 63)
            large_acc += gl.gather(large_acc, large_src, 0)
        large_warp_sums = gl.gather(large_acc, gl.minimum(large_lanes, 7) * 64, 0)
        large_warp_sums = gl.where(large_lanes < 8, large_warp_sums, 0.0)
        for large_step in gl.static_range(6):
            large_warp_sums += gl.gather(large_warp_sums, gl.minimum(large_lanes + (32 >> large_step), 63), 0)
        large_denom = gl.sum(gl.where(large_lanes == 0, large_warp_sums, 0.0), 0)
        large_inv = gl.div_rn(1.0, large_denom)
        for large_block in range(gl.cdiv(n, 512)):
            large_pos = large_block * 512 + large_lanes
            if large_block * 512 < active_end and (large_block + 1) * 512 > active_start:
                large_x = _load_scaled_masked(scores, offset, large_pos, large_pos < n, qpos, key_start, WINDOW, SCALE)
                gl.store(probs + offset + large_pos, (_large_exp_ftz(large_x - large_mx) * large_inv).to(gl.bfloat16), large_pos < n)
            else:
                gl.store(probs + offset + large_pos, 0.0, large_pos < n)


def softmax(scores, query_count, context_length, window, scale):
    assert scores.dtype == torch.bfloat16 and scores.is_contiguous()
    n = scores.shape[-1]
    out = torch.empty_like(scores)
    _prefill_softmax_streaming[(scores.numel() // n,)](scores, out, n, query_count, context_length, window or -1, scale, n <= 2048, num_warps=1 if n <= 2048 else 8, enable_fp_fusion=False)
    return out

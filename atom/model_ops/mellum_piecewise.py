"""Bounded tensor-only compilation and decode request-shaped CUDA graph regions.

Attention/CPU metadata remain outside these regions. This is independent of
the model-level level=3 backend: no decorated model is changed.
"""
import logging
import os

import torch
import torch.nn.functional as F

LOG = logging.getLogger(__name__)
COMPILE = os.environ.get("ATOM_MELLUM_COMPILE_PIECES", "0") == "1"
GRAPH = os.environ.get("ATOM_MELLUM_GRAPH_PIECES", "0") == "1"
GRAPH_SIZES = frozenset(
    map(int, os.environ.get("ATOM_MELLUM_GRAPH_SIZES", "1,8,16,32").split(","))
)


def norm_tensor(x, weight, eps):
    values = x.float()
    variance = values.square().mean(-1, keepdim=True)
    normalized = (values * torch.rsqrt(variance + eps)).to(x.dtype)
    return weight * normalized


def rope_tensor(q, k, cos, sin):
    def apply(x):
        a, b = x.chunk(2, -1)
        return x * cos + torch.cat((-b, a), -1) * sin

    return apply(q), apply(k)


def gate_tensor(values):
    gate, up = values.chunk(2, -1)
    return F.silu(gate) * up


def compile_piece(fn):
    if not COMPILE:
        return fn
    # Respect explicit BF16 casts rather than silently removing rounding
    # boundaries when fusing pointwise work. Reduction parity is tested, not assumed.
    return torch.compile(
        fn,
        fullgraph=True,
        dynamic=True,
        options={"emulate_precision_casts": True, "triton.cudagraphs": False},
    )


rope = compile_piece(rope_tensor)
gate = compile_piece(gate_tensor)
_GRAPHS = {}


def request_region(x, weight, bias, parts, decode, *, norm_eps=None):
    def eager(values):
        if norm_eps is not None:
            if len(parts) == 1:
                return norm_tensor(values, weight, norm_eps)
            return torch.cat(
                [norm_tensor(values[a:b], weight, norm_eps) for a, b in parts], 0
            )
        if len(parts) == 1:
            return F.linear(values, weight, bias)
        return torch.cat([F.linear(values[a:b], weight, bias) for a, b in parts], 0)

    # Never pad a request or capture request metadata. Only actual unit-row
    # decode regions qualify. Other shapes retain the original implementation.
    eligible = (
        GRAPH
        and decode
        and x.is_cuda
        and x.shape[0] in GRAPH_SIZES
        and all(b - a == 1 for a, b in parts)
        and not torch.cuda.is_current_stream_capturing()
    )
    if not eligible:
        return eager(x)
    key = (
        weight.data_ptr(),
        tuple(weight.shape),
        tuple(weight.stride()),
        None if bias is None else bias.data_ptr(),
        tuple(x.shape),
        tuple(x.stride()),
        x.dtype,
        x.device,
        tuple(parts),
        norm_eps,
    )
    entry = _GRAPHS.get(key)
    if entry is None and len(_GRAPHS) >= 1024:
        return eager(x)
    if entry is None:
        # Static inputs own their storage. Parameter references prevent pointer
        # recycling; service weights are immutable. Independent graph pools keep
        # captures safe when scheduler batch sizes change their replay order.
        static = torch.empty_strided(
            x.shape, x.stride(), device=x.device, dtype=x.dtype
        )
        static.copy_(x)
        stream = torch.cuda.Stream(device=x.device)
        stream.wait_stream(torch.cuda.current_stream(x.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                eager(static)
        torch.cuda.current_stream(x.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = eager(static)
        entry = (static, graph, output, weight, bias)
        _GRAPHS[key] = entry
        LOG.warning(
            "Mellum piece graph captured: rows=%s weight=%s entries=%s",
            x.shape[0],
            tuple(weight.shape),
            len(_GRAPHS),
        )
    static, graph, output, _, _ = entry
    static.copy_(x)
    graph.replay()
    # Do not expose reusable storage: Q/K/V or different pipeline consumers
    # can keep outputs alive across later calls to this same graph.
    return output.clone()

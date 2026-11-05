"""Experimental fused prefill softmax dispatch from same-GPU screening."""
from atom.model_ops.mellum_prefill_softmax_gluon import softmax as _streaming
from atom.model_ops.mellum_prefill_softmax_runtime import softmax as _cached

def softmax(scores, query_count, context_length, window, scale):
    n = scores.shape[-1]
    full_window = window is None or window <= 0
    use_cached = 2048 < n <= 4096 and (n % 4 == 0 or (full_window and n > 3072))
    fn = _cached if use_cached else _streaming
    return fn(scores, query_count, context_length, window, scale)

"""Experimental fused prefill softmax dispatch from same-GPU screening."""
from atom.model_ops.mellum_prefill_softmax_gluon import softmax as _streaming
from atom.model_ops.mellum_prefill_softmax_runtime import softmax as _cached

def softmax(scores, query_count, context_length, window, scale):
    n = scores.shape[-1]
    if 2048 < n <= 4096 and window == 1024 and __import__('os').environ.get('ATOM_MELLUM_PREFILL_WINDOW_CACHE') == '1':
        from atom.model_ops.mellum_prefill_softmax_window_cache import softmax as cached_window
        return cached_window(scores, query_count, context_length, window, scale, rows=1)
    if 2048 < n <= 4096 and __import__('os').environ.get('ATOM_MELLUM_PREFILL_LARGE_CACHE') == '1':
        from atom.model_ops.mellum_prefill_softmax_large_cache import softmax as adaptive
        return adaptive(scores, query_count, context_length, window, scale, rows=1)
    if n <= 2048 and __import__('os').environ.get('ATOM_MELLUM_PREFILL_ADAPTIVE_CACHE') == '1':
        from atom.model_ops.mellum_prefill_softmax_adaptive import softmax as adaptive
        return adaptive(scores, query_count, context_length, window, scale, rows=1)
    full_window = window is None or window <= 0
    use_cached = 2048 < n <= 4096 and (n % 4 == 0 or (full_window and n > 3072))
    fn = _cached if use_cached else _streaming
    return fn(scores, query_count, context_length, window, scale)

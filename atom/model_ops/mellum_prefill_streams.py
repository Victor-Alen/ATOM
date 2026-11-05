"""Bounded prefill request concurrency using unchanged native CK operators."""
from functools import lru_cache
import torch
from atom.model_ops.mellum_native_prefill_moe import experts

@lru_cache(maxsize=8)
def _streams(device, count):
    return tuple(torch.cuda.Stream(device=device) for _ in range(count))

def experts_parallel(layer, reference, x, ids, weights, parts, count=2):
    if count not in (2,4):
        raise ValueError('Screened stream counts are 2 and 4')
    # First use initializes immutable weight shuffles and profile tables on
    # the caller stream; later children wait for those writes before use.
    initialized = (getattr(layer, '_mellum_native_prefill_stage1', None) is not None
                   and getattr(layer, '_mellum_native_prefill_stage2', None) is not None)
    if len(parts)<2 or not initialized or torch.cuda.is_current_stream_capturing():
        return torch.cat([experts(layer,reference,x[a:b],ids[a:b],weights[a:b]) for a,b in parts],0)
    parent=torch.cuda.current_stream(x.device)
    streams=_streams(x.device,min(count,len(parts)))
    for stream in streams:
        stream.wait_stream(parent)
        x.record_stream(stream);ids.record_stream(stream);weights.record_stream(stream)
    outputs=[]
    for i,(a,b) in enumerate(parts):
        with torch.cuda.stream(streams[i%len(streams)]):
            outputs.append(experts(layer,reference,x[a:b],ids[a:b],weights[a:b]))
    for stream in streams:
        parent.wait_stream(stream)
    for output in outputs:
        output.record_stream(parent)
    return torch.cat(outputs,0)

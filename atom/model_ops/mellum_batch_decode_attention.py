"""Batch equal-length decode GEMMs without padding the reduction dimension."""
from collections import defaultdict
import os

import torch
import torch.nn.functional as F

from atom.utils.forward_context import get_forward_context


def attention(q, kc, vc, metadata, lengths, scale, window, cache_format):
    ctx = get_forward_context()
    plans = getattr(ctx, '_mellum_decode_attention_plans', None)
    if plans is None:
        plans = {}
        ctx._mellum_decode_attention_plans = plans
    key = (id(metadata), window, kc.shape[3], q.device)
    plan = plans.get(key)
    if plan is None:
        groups = defaultdict(list)
        for row, length in enumerate(lengths):
            effective = min(length, window) if window is not None and window > 0 else length
            groups[effective].append(row)
        plan = []
        for length, members in groups.items():
            rows = torch.tensor(members, device=q.device, dtype=torch.long)
            starts = torch.tensor([lengths[i]-length for i in members], device=q.device, dtype=torch.long)
            positions = starts[:, None] + torch.arange(length, device=q.device)[None, :]
            block_size = kc.shape[3]
            blocks = metadata.block_tables[rows[:, None], positions // block_size].long().flatten()
            slots = (positions % block_size).flatten()
            plan.append((length, rows, blocks, slots))
        plans[key] = plan
    heads, dim = q.shape[1:]
    kvheads = kc.shape[1]
    repeat = heads // kvheads
    out = torch.empty_like(q)
    for length, rows, blocks, slots in plan:
        batch = len(rows)
        keys = kc[blocks, :, :, slots, :].reshape(batch, length, kvheads, dim)
        if cache_format == 'SHUFFLE':
            pack = 16 // vc.element_size()
            layout = vc.view(vc.shape[0], kvheads, vc.shape[-1] // pack, dim, pack)
            values = layout[blocks, :, slots // pack, :, slots % pack].reshape(batch, length, kvheads, dim)
        else:
            assert cache_format == 'NHD', cache_format
            values = vc[blocks, :, :, slots].reshape(batch, length, kvheads, dim)
        keys = keys.transpose(1, 2)[:, :, None].expand(batch, kvheads, repeat, length, dim).reshape(batch, heads, length, dim)
        values = values.transpose(1, 2)[:, :, None].expand(batch, kvheads, repeat, length, dim).reshape(batch, heads, length, dim)
        queries = q.index_select(0, rows).unsqueeze(2)
        if os.environ.get('ATOM_MELLUM_BATCH_ATTENTION_GEMMS', '0') == '1':
            scores = torch.matmul(queries, keys.transpose(2, 3)) * scale
            probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            result = torch.matmul(probabilities, values).squeeze(2)
        else:
            # Keep HF's head-batch size for GEMM dispatch; batch only KV gathering.
            results = []
            for row in range(batch):
                scores = torch.matmul(queries[row:row+1], keys[row:row+1].transpose(2, 3)) * scale
                probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
                results.append(torch.matmul(probabilities, values[row:row+1]).squeeze(2))
            result = torch.cat(results, dim=0)
        out.index_copy_(0, rows, result)
    return out

"""Mellum MTP single-token draft with explicit checkpoint validation.

Forward structure follows llama.cpp Mellum graph_mtp and conversion/mellum.py:
post-final-norm target hidden, concat [embedding, hidden], MTP attention defaults
to sliding_attention unless mtp_layer_types says otherwise. The supported scope
is unquantized BF16 with TP=PP=1; deployment evidence is recorded separately.
"""
from __future__ import annotations

import copy
import json
import os
import struct
from glob import glob
from pathlib import Path
from typing import Any

import torch
from torch import nn
from aiter import QuantType
from aiter.dist.parallel_state import get_pp_group, get_tp_group
from atom.config import Config
from atom.model_ops.layernorm import RMSNorm
from atom.model_ops.linear import ReplicatedLinear
from atom.model_ops.moe import FusedMoE
from atom.models.mellum import MellumDecoderLayer, MellumForCausalLM
from atom.models.utils import maybe_prefix


def expected_mtp_shapes(config) -> dict[str, tuple[int, ...]]:
    """Exact source manifest expected for the known unquantized single MTP head."""
    hidden = config.hidden_size
    dim = config.head_dim
    intermediate = config.moe_intermediate_size
    shapes = {
        'mtp.fc.weight': (hidden, 2 * hidden),
        'mtp.pre_fc_norm_embedding.weight': (hidden,),
        'mtp.pre_fc_norm_hidden.weight': (hidden,),
        'mtp.norm.weight': (hidden,),
        'mtp.layers.0.input_layernorm.weight': (hidden,),
        'mtp.layers.0.post_attention_layernorm.weight': (hidden,),
        'mtp.layers.0.self_attn.q_proj.weight': (config.num_attention_heads * dim, hidden),
        'mtp.layers.0.self_attn.k_proj.weight': (config.num_key_value_heads * dim, hidden),
        'mtp.layers.0.self_attn.v_proj.weight': (config.num_key_value_heads * dim, hidden),
        'mtp.layers.0.self_attn.o_proj.weight': (hidden, config.num_attention_heads * dim),
        'mtp.layers.0.self_attn.q_norm.weight': (dim,),
        'mtp.layers.0.self_attn.k_norm.weight': (dim,),
        'mtp.layers.0.mlp.gate.weight': (config.num_experts, hidden),
    }
    for expert in range(config.num_experts):
        for projection, shape in (
            ('gate_proj', (intermediate, hidden)),
            ('up_proj', (intermediate, hidden)),
            ('down_proj', (hidden, intermediate)),
        ):
            shapes[f'mtp.layers.0.mlp.experts.{expert}.{projection}.weight'] = shape
    return shapes


def validate_mtp_manifest(manifest: dict[str, Any], config) -> None:
    """Call on the safetensors header BEFORE invoking generic load_model.

    Header entries must have shape and dtype fields. This proves layout and
    source coverage, not the learned forward semantics or numerical correctness.
    """
    expected = expected_mtp_shapes(config)
    actual = {name: info for name, info in manifest.items() if name.startswith('mtp.')}
    if set(actual) != set(expected):
        raise ValueError(f'MTP manifest mismatch: missing={sorted(set(expected)-set(actual))}; '
                         f'extra={sorted(set(actual)-set(expected))}')
    for name, shape in expected.items():
        if tuple(actual[name]['shape']) != shape or actual[name]['dtype'] != 'BF16':
            raise ValueError(f'Unexpected MTP tensor: {name}: {actual[name]}, expected BF16 {shape}')


def make_local_draft_config(atom_config):
    """Copy only HF config; preserve runtime registries owned by ATOM."""
    spec = atom_config.speculative_config
    if spec is None or spec.method != 'mtp' or spec.num_speculative_tokens != 1:
        raise ValueError('Mellum MTP only supports method=mtp and K=1')
    if any(os.environ.get(name) == '1' for name in (
        'ATOM_REFERENCE_MODE', 'ATOM_REFERENCE_RECOMPUTE', 'ATOM_REFERENCE_OWN_CACHE'
    )):
        raise ValueError('Mellum MTP review candidate does not support reference cache modes')
    if atom_config.load_dummy:
        raise ValueError('Mellum MTP requires real checkpoint weights')
    target = atom_config.hf_config
    draft_spec = spec.draft_model_hf_config
    kinds = getattr(draft_spec, 'mtp_layer_types', None) or ['sliding_attention']
    if len(kinds) != 1 or kinds[0] not in target.rope_parameters:
        raise ValueError('Expected one supported MTP attention type')
    kind = kinds[0]
    mlp_kinds = getattr(draft_spec, 'mtp_mlp_layer_types', None) or ['sparse']
    if mlp_kinds != ['sparse']:
        raise ValueError('Only the inspected sparse MTP head is supported')
    if getattr(draft_spec, 'num_nextn_predict_layers', 1) != 1:
        raise ValueError('Only one MTP decoder layer supported')
    config = copy.deepcopy(target)
    start = target.num_hidden_layers
    if len(target.layer_types) != start:
        raise ValueError('Expected target-only layer_types; do not mutate target for MTP')
    config.layer_types = list(target.layer_types) + [kind]
    if kind == 'sliding_attention':
        rope = config.rope_parameters[kind]
        if rope.get('rope_type') != 'default' or rope.get('rope_theta') != 500000.0:
            raise ValueError('Inspected MTP sliding RoPE requires default theta=500000')
    config.num_nextn_predict_layers = 1
    if config.num_experts != 64 or config.hidden_size != 2304 or config.head_dim != 128:
        raise ValueError('Mellum MTP is restricted to the inspected Mellum 12B MTP head')
    if getattr(config, 'attention_bias', False):
        raise ValueError('Inspected checkpoint has no attention bias')
    if start in getattr(config, 'mlp_only_layers', []) or (start + 1) % config.decoder_sparse_step:
        raise ValueError('Draft layer must use MoE; check draft local sparse-layer configuration')
    local = copy.copy(atom_config)
    local.hf_config = config
    return local, start


class _MellumMTPBody(nn.Module):
    def __init__(self, atom_config, start: int, prefix: str):
        super().__init__()
        self.mtp_start_layer_idx = start
        # No temporary full-vocabulary allocation; share_with_target binds these
        # before any forward. The generic proposer invokes that hook first.
        self.embed_tokens = nn.Identity()
        layer = MellumDecoderLayer(
            atom_config=atom_config, layer_num=start,
            prefix=maybe_prefix(prefix, f'layers.{start}'),
        )
        layer.self_attn.attn.impl.hf_rope_config = atom_config.hf_config
        self.layers = nn.ModuleDict({str(start): layer})


class _MellumMTPProjection(nn.Module):
    def __init__(self, atom_config, prefix: str):
        super().__init__()
        config = atom_config.hf_config
        self.pre_fc_norm_embedding = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc = ReplicatedLinear(config.hidden_size * 2, config.hidden_size,
                                   bias=False, quant_config=None,
                                   prefix=maybe_prefix(prefix, 'fc'))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class MellumMTP(nn.Module):
    packed_modules_mapping = dict(MellumForCausalLM.packed_modules_mapping)
    supports_selected_tail = True

    def __init__(self, atom_config: Config, prefix: str = ''):
        super().__init__()
        if get_tp_group().world_size != 1 or get_pp_group().world_size != 1:
            raise ValueError('Mellum MTP is restricted to TP=PP=1')
        if (atom_config.quant_config is not None
                and atom_config.quant_config.quant_type != QuantType.No):
            raise ValueError('Mellum MTP only supports the inspected BF16 head')
        local, start = make_local_draft_config(atom_config)
        self.config = local.hf_config
        self.atom_config = local
        self.model = _MellumMTPBody(local, start, maybe_prefix(prefix, 'model'))
        self.mtp = _MellumMTPProjection(local, maybe_prefix(prefix, 'mtp'))
        self.lm_head = nn.Identity()
        self._source_seen: set[str] = set()
        self._shared = False
        self._preflight_complete = False
        self._expected_sources = set(expected_mtp_shapes(self.config))
        if len(self._expected_sources) != 205:
            raise ValueError('Expected exactly 205 MTP source weights')

    def preflight_checkpoint(self, checkpoint_path: str) -> None:
        """Header-only audit using the SAME shard selection as the generic loader."""
        from atom.model_loader.weight_utils import filter_duplicate_safetensors_files
        from atom.utils import envs
        from transformers.utils import SAFE_WEIGHTS_INDEX_NAME
        if envs.ATOM_LOADER_USE_THREADPOOL:
            raise ValueError('Set ATOM_LOADER_USE_THREADPOOL=0 so weight-write errors propagate')
        root = Path(checkpoint_path)
        if not root.is_dir():
            raise ValueError('Mellum MTP preflight requires an already-downloaded checkpoint directory')
        shards = filter_duplicate_safetensors_files(
            glob(str(root / '*.safetensors')), str(root), SAFE_WEIGHTS_INDEX_NAME,
        )
        manifest = {}
        for path in shards:
            with open(path, 'rb') as handle:
                length_bytes = handle.read(8)
                if len(length_bytes) != 8:
                    raise ValueError(f'Truncated safetensors file: {path}')
                length = struct.unpack('<Q', length_bytes)[0]
                if not 0 < length <= 100_000_000:
                    raise ValueError(f'Invalid safetensors header length in {path}')
                raw = handle.read(length)
                if len(raw) != length:
                    raise ValueError(f'Truncated safetensors header in {path}')
                header = json.loads(raw)
                file_size = os.fstat(handle.fileno()).st_size
            for name, info in header.items():
                if not name.startswith('mtp.'):
                    continue
                if name in manifest:
                    raise ValueError(f'Duplicate MTP source in selected shards: {name}')
                begin, end = info['data_offsets']
                if not 0 <= begin <= end <= file_size - 8 - length:
                    raise ValueError(f'Truncated or invalid MTP tensor: {name}')
                elements = 1
                for dimension in info['shape']:
                    elements *= dimension
                if end - begin != 2 * elements:
                    raise ValueError(f'Invalid BF16 byte extent: {name}')
                manifest[name] = info
        validate_mtp_manifest(manifest, self.config)
        self._preflight_complete = True

    def remap_mtp_weight_name(self, name: str) -> str | None:
        if not name.startswith('mtp.'):
            return None  # Never load a target tensor into the MTP head.
        if name not in self._expected_sources:
            raise ValueError(f'Unknown MTP checkpoint tensor: {name}')
        if name in self._source_seen:
            raise ValueError(f'Duplicate MTP checkpoint tensor: {name}')
        self._source_seen.add(name)
        if name.startswith('mtp.layers.0.'):
            suffix = name[len('mtp.layers.0.'):]
            return f'model.layers.{self.model.mtp_start_layer_idx}.{suffix}'
        return name

    def get_expert_mapping(self):
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name='gate_proj', ckpt_down_proj_name='down_proj',
            ckpt_up_proj_name='up_proj', num_experts=self.config.num_experts,
        )

    def share_with_target(self, target, loaded: set[str]) -> None:
        # Called only after load_model returns successfully. A fused parameter
        # being present in loaded alone cannot prove every q/k/v or expert shard
        # was supplied; require source coverage AND destination coverage.
        if not self._preflight_complete:
            raise RuntimeError('MTP checkpoint preflight was not called')
        missing_sources = self._expected_sources - self._source_seen
        missing_destinations = set(dict(self.named_parameters())) - set(loaded)
        if missing_sources or missing_destinations:
            raise RuntimeError(f'Incomplete MTP load: sources={sorted(missing_sources)}; '
                               f'parameters={sorted(missing_destinations)}')
        if target.model.embed_tokens.weight.shape != (self.config.vocab_size, self.config.hidden_size):
            raise ValueError('Unexpected shared embedding shape')
        if target.lm_head.weight.shape != (self.config.vocab_size, self.config.hidden_size):
            raise ValueError('Unexpected shared head shape')
        self.model.embed_tokens = target.model.embed_tokens
        self.lm_head = target.lm_head
        self._shared = True

    def forward(self, input_ids, positions, hidden_states, intermediate_tensors=None,
                inputs_embeds=None, spec_step_idx: int = 0, last_token_indices=None):
        if not self._shared:
            raise RuntimeError('MTP weights must load and shared modules must bind before forward')
        if spec_step_idx != 0 or intermediate_tensors is not None:
            raise ValueError('K=1 and no pipeline intermediates supported')
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        # llama.cpp graph_mtp uses [embedding, post-final-norm target hidden].
        # EagleProposer has already shifted input IDs and incremented positions.
        projected_input = torch.cat((self.mtp.pre_fc_norm_embedding(inputs_embeds),
                                     self.mtp.pre_fc_norm_hidden(hidden_states)), dim=-1)
        from atom.utils.forward_context import get_forward_context
        draft_fc_mode = os.environ.get('ATOM_MELLUM_DRAFT_BATCHED_FC', '0')
        if (draft_fc_mode == '2'
                or (draft_fc_mode == '1' and not get_forward_context().context.is_prefill)):
            # This projection belongs only to the draft head. Its 2304x4608
            # weight has no native target GEMM solution; batching avoids one
            # GEMM per token row. Target verification arithmetic is unchanged.
            x = torch.nn.functional.linear(projected_input, self.mtp.fc.weight,
                                           self.mtp.fc.bias)
        else:
            x = self.mtp.fc(projected_input)
        layer = self.model.layers[str(self.model.mtp_start_layer_idx)]
        tail_mode = os.environ.get('ATOM_MELLUM_DRAFT_TAIL_MLP', '0')
        forward_context = get_forward_context()
        if (last_token_indices is not None and
                (tail_mode == '2' or (tail_mode == '1' and forward_context.context.is_prefill))):
            # This is the final and only draft layer. Attention still processes
            # every position and writes all required KV entries. The proposer
            # consumes only the selected last-valid row of each request, so
            # other positions need no post-attention MLP or output norm.
            if os.environ.get('ATOM_MELLUM_DECODE_UNIT_ROWS') != '1':
                raise ValueError('Selected draft tails require unit-row decode arithmetic')
            residual = x.index_select(0, last_token_indices)
            attended = layer.self_attn(positions=positions,
                                      hidden_states=layer.input_layernorm(x))
            selected = attended.index_select(0, last_token_indices)
            original_context = forward_context.context
            selected_context = copy.copy(original_context)
            selected_context.is_prefill = False
            selected_context.batch_size = last_token_indices.numel()
            selected_context.graph_bs = last_token_indices.numel()
            selected_context.positions = positions.index_select(0, last_token_indices)
            # No attention runs under this temporary context. It only selects
            # unit-row norm/router/MoE kernels for the selected draft positions.
            forward_context.context = selected_context
            try:
                selected, residual = layer.post_attention_layernorm(selected, residual)
                selected = layer.mlp(selected)
                selected = self.mtp.norm(selected + residual)
            finally:
                forward_context.context = original_context
            result = torch.zeros_like(x)
            result.index_copy_(0, last_token_indices, selected)
            return result
        x, residual = layer(positions=positions, hidden_states=x, residual=None)
        # Mellum decoder retains its last residual addition as a separate tensor.
        return self.mtp.norm(x + residual)

    def compute_logits(self, hidden_states, spec_step_idx: int = 0):
        if not self._shared or spec_step_idx != 0:
            raise RuntimeError('Shared modules and K=1 are required')
        return self.lm_head(hidden_states)

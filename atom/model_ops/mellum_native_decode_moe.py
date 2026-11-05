"""Native CK decode MoE preserving HF BF16 boundaries without weight gathering."""
import os
import torch
from aiter import ActivationType, QuantType
from aiter.fused_moe import get_2stage_cfgs, get_padded_M, moe_sorting

def experts(x, w13, w2, ids, weights):
    b, topk = ids.shape
    assert topk == 8 and x.shape == (b,2304) and w2.shape == (64,2304,896)
    # Decode HF uses independent batched GEMVs, not prefill expert-row profiles.
    if os.getenv('CK_MELLUM_STAGE1_KORDER_PLAN','0') != '0':
        raise RuntimeError('Native decode requires CK_MELLUM_STAGE1_KORDER_PLAN=0')
    if os.getenv('CK_MELLUM_STAGE1_MODE') != '1' or os.getenv('CK_MELLUM_STAGE2_ROUND_WEIGHT') != '1':
        raise RuntimeError('Native decode requires stage1 and stage2 BF16 rounding modes')
    meta=get_2stage_cfgs(get_padded_M(b),2304,896,64,8,x.dtype,w13.dtype,w13.dtype,QuantType.No,True,ActivationType.Silu,False,0,0,True)
    if meta.run_1stage or meta.ksplit != 0:
        raise RuntimeError('Unexpected native CK dispatch')
    si,sw,se,nv,_=moe_sorting(ids.int(),weights.float(),64,2304,x.dtype,meta.block_m)
    gated=torch.empty((b,8,896),device=x.device,dtype=x.dtype)
    stage1_options = {}
    stage1_ids, stage1_experts, stage1_valid = si, se, nv
    stage1_block_m = meta.block_m
    tile = os.getenv('ATOM_MELLUM_DECODE_STAGE1_TILE', '')
    if tile:
        # Explicit native CK template: fixed-input all-layer/batch checks required.
        if tile != 'm32n64k64v1':
            raise RuntimeError('Unvalidated Mellum native decode stage1 tile')
        stage1_options['kernelName'] = 'moe_ck2stages_gemm1_256x32x64x64_1x4_TypeCast_v1_Nswizzle0_Quant0_MulRoutedWeight0_silu_B16_B16_B16'
        stage1_block_m = 32
        if stage1_block_m != meta.block_m:
            stage1_ids, _, stage1_experts, stage1_valid, _ = moe_sorting(
                ids.int(), weights.float(), 64, 2304, x.dtype, stage1_block_m)
    meta.stage1(x,w13,w2,stage1_ids,stage1_experts,stage1_valid,gated,8,block_m=stage1_block_m,a1_scale=None,w1_scale=None,sorted_weights=None,**stage1_options)
    token=si&0xffffff;slot=(si>>24)&255
    flat=torch.where((token<b)&(slot<8),token*8+slot,b*8)
    per_slot=torch.zeros((b*8,2304),device=x.device,dtype=x.dtype)
    meta.stage2(gated.view(b*8,1,896),w13,w2,flat,se,nv,per_slot,1,block_m=meta.block_m,w2_scale=None,a2_scale=None,sorted_weights=sw)
    return per_slot.view(b,8,2304).sum(1)

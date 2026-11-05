"""Isolated native CK stage2 candidate; GPU block profiles, no model hook."""
from pathlib import Path
import json, os, sys, types, importlib.util
import torch
import triton
import triton.language as tl
from aiter.fused_moe import moe_sorting
from aiter.ops.shuffle import shuffle_weight
from atom.model_ops.mellum_native_prefill_hybrid_core import expert_counts

@triton.jit
def _plan(Counts, Tiles, Depths, Mis, Masks, Info, Nv, P:tl.constexpr, K:tl.constexpr, KEEP_MI:tl.constexpr, SP:tl.constexpr):
    e=tl.arange(0,64)
    n=tl.load(Counts+e)
    t=tl.load(Tiles+n);d=tl.load(Depths+n);mi=tl.load(Mis+n);mask=tl.load(Masks+n)
    source=tl.cdiv(n,SP)
    src=(tl.cumsum(source)-source)*SP
    keep=(d==32) if K==32 else (d>=64)
    if KEEP_MI:keep=keep&(mi==KEEP_MI)
    blocks=tl.where(keep,(n//t)*tl.cdiv(t,P)+tl.cdiv(n%t,P),0)
    prefix=tl.cumsum(blocks)-blocks
    tl.store(Info+e,n);tl.store(Info+64+e,t);tl.store(Info+128+e,d//K)
    tl.store(Info+192+e,mi);tl.store(Info+256+e,mask)
    tl.store(Info+320+e,src);tl.store(Info+384+e,blocks);tl.store(Info+448+e,prefix)
    tl.store(Nv,tl.sum(blocks)*P);tl.store(Nv+1,0)

@triton.jit
def _pack(Si, Sw, Info, OutSi, OutSw, Se, Nv, M, P:tl.constexpr, B:tl.constexpr):
    e=tl.program_id(0);chunk=tl.program_id(1)
    n=tl.load(Info+e);t=tl.load(Info+64+e);d=tl.load(Info+128+e)
    mi=tl.load(Info+192+e);mask=tl.load(Info+256+e)
    source=tl.load(Info+320+e);blocks=tl.load(Info+384+e);base=tl.load(Info+448+e)
    if chunk*B<blocks*P:
        i=chunk*B+tl.arange(0,B);pb=i//P;lane=i%P
        per_tile=tl.cdiv(t,P);tile=pb//per_tile;in_tile=(pb%per_tile)*P+lane
        row=tile*t+in_tile
        live=(pb<blocks)&(row<n)&(in_tile<t)
        v=tl.load(Si+source+row,live,M);w=tl.load(Sw+source+row,live,0.)
        tl.store(OutSi+base*P+i,v,pb<blocks);tl.store(OutSw+base*P+i,w,pb<blocks)
        is_block=(lane==0)&(pb<blocks)
        profile=((tile&mask)*d)|(d<<8)|((mi==32).to(tl.int32)<<16)
        tl.store(Se+base+pb,e,is_block);tl.store(Nv+2+base+pb,profile,is_block)

class Stage2:
    def __init__(self,w13,w2,root,raw_w2=None):
        self.w13=w13;self.w2=w2;self.root=Path(root)
        if raw_w2 is None:
            raw_w2=w2.view(64,144,28,4,16,8).permute(0,1,4,2,3,5).reshape(64,2304,896)
        self.w2_kpack4=shuffle_weight(raw_w2,layout=(16,4),is_guinterleave=True)
        self.w2_mi32=shuffle_weight(raw_w2,layout=(32,8),is_guinterleave=True)
        cfg=json.loads((self.root/'native_prefill_down_dispatch16k.json').read_text())
        assert cfg['complete']
        profiles=cfg['profiles'];profiles[0]=dict(row_tile=16,depth=512,mi=16,lsu=1)
        assert all(x['lsu']==1 for x in profiles)
        self.bounds=[]
        for p,k,mi in ((32,64,0),(64,32,16),(64,32,32)):
            bounds=[(0,0)];max_blocks=0;min_count=1<<30;ratio=1.
            for m in range(1,16385):
                f=profiles[m]
                keep=(f['depth']==32 if k==32 else f['depth']>=64) and (not mi or f['mi']==mi)
                if keep:
                    t=f['row_tile'];blocks=(m//t)*((t+p-1)//p)+(m%t+p-1)//p
                    max_blocks=max(max_blocks,blocks);min_count=min(min_count,m)
                    # Expert counts belonging to each profile interval are
                    # bounded independently; account for each final partial block.
                    ratio=max(ratio,((t+p-1)//p)*p/t)
                active=min(64,(m*8)//min_count)
                capacity=min(active*max_blocks,int(ratio*m*8/p)+active+1) if active else 0
                bounds.append((max_blocks,capacity))
            self.bounds.append(bounds)

        self.tables=tuple(torch.tensor(a,dtype=torch.int32,device=w2.device) for a in (
            [x['row_tile'] for x in profiles], [x['depth'] for x in profiles],
            [x['mi'] for x in profiles], [(1<<((896//x['depth']).bit_length()-1))-1 for x in profiles]))
        name='module_moe_ck2stages_b16_b16_preshuffle_on_b16_silu_no_mulWeightStage2'
        package='mellum_prefill_down_plan_n128_native';sys.modules.setdefault(package,types.ModuleType(package))
        qualified=package+'.'+name
        if qualified not in sys.modules:
            spec=importlib.util.spec_from_file_location(qualified,self.root/'native_moe_prefill_down_plan_n128_jit'/(name+'.so'))
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);sys.modules[qualified]=module
        self.native=sys.modules[qualified]
        package='mellum_prefill_down_plan_mi32_native';sys.modules.setdefault(package,types.ModuleType(package))
        qualified=package+'.'+name
        if qualified not in sys.modules:
            spec=importlib.util.spec_from_file_location(qualified,self.root/'native_moe_prefill_down_plan_mi32_jit'/(name+'.so'))
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);sys.modules[qualified]=module
        self.native_mi32=sys.modules[qualified]

    def prepare(self,ids,weights,shared=None):
        m=ids.shape[0]
        assert 0<m<=16384 and ids.shape[1]==8
        if shared is None:
            si,sw,_,_,_=moe_sorting(ids,weights.float(),64,2304,torch.bfloat16,32)
            counts=expert_counts(ids);source_physical=32
        else:
            si,sw,source_nv,counts=shared;source_physical=64
        plans=[]
        for g,(p,k,mi) in enumerate(((32,64,0),(64,32,16),(64,32,32))):
            max_blocks,capacity=self.bounds[g][m]
            if not capacity:
                plans.append(None)
                continue
            info=torch.empty(512,dtype=torch.int32,device=ids.device)
            nv=torch.empty(capacity+2,dtype=torch.int32,device=ids.device)
            osi=torch.empty(capacity*p,dtype=torch.int32,device=ids.device)
            osw=torch.empty(capacity*p,dtype=torch.float32,device=ids.device)
            se=torch.empty(capacity,dtype=torch.int32,device=ids.device)
            _plan[(1,)](counts,*self.tables,info,nv,p,k,mi,source_physical)
            _pack[(64,triton.cdiv(max_blocks*p,256))](si,sw,info,osi,osw,se,nv,m,p,256)
            plans.append((osi,osw,se,nv))
        return plans
    def run(self,gated,plans,out):
        assert os.getenv('CK_MELLUM_STAGE2_ROUTE_PLAN')=='1'
        assert os.getenv('CK_MELLUM_STAGE2_SLOT_OUTPUT')=='1'
        assert os.getenv('CK_MELLUM_STAGE2_ROUND_WEIGHT')=='1'
        for g,(p,plan) in enumerate(zip((32,64,64),plans)):
            if plan is None:continue
            si,sw,se,nv=plan
            name=f'moe_ck2stages_gemm2_256x{p}x64x64_1x4_TypeCastExpertWeight_v1_Nswizzle0_Quant0_MulRoutedWeight1_B16_B16_B16'
            module=self.native_mi32 if g==2 else self.native
            weight=(self.w2,self.w2_kpack4,self.w2_mi32)[g]
            module.ck_moe_stage2(gated,self.w13,weight,si,se,nv,out,8,block_m=p,
                w2_scale=None,a2_scale=None,sorted_weights=sw,kernelName=name)
        return out




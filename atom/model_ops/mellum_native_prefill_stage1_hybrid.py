"""Experimental native CK prefill stage1 split by MFMA/accumulation class."""
from pathlib import Path
import json,os,sys,types,importlib.util
import torch
from aiter import ActivationType
from aiter.fused_moe import moe_sorting
from atom.model_ops.mellum_native_prefill_hybrid_routes import prepare_all,expert_counts
ROOT=Path(os.environ['ATOM_MELLUM_PREFILL_NATIVE_ROOT'])
_MOD='module_moe_ck2stages_b16_b16_preshuffle_on_b16_silu_no_mulWeightStage2'
_MODULES={}
_BOUNDS=None
def _module(group):
 if group not in _MODULES:
  folder=['native_moe_prefill_long16_abi_jit','native_moe_prefill_long32_abi_jit','native_moe_prefill_lsu2_abi_jit','native_moe_prefill_lsu4_abi_jit','native_moe_prefill_long32k32_abi_jit','native_moe_prefill_wide_abi_jit','native_moe_prefill_wide192_abi_jit'][group]
  package='mellum_prefill_wide192_group_'+str(group);sys.modules[package]=types.ModuleType(package)
  spec=importlib.util.spec_from_file_location(package+'.'+_MOD,ROOT/folder/(_MOD+'.so'))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);_MODULES[group]=module
 return _MODULES[group]

class Stage1:
 def __init__(self,w16,w32,w2):
  self.w16,self.w32,self.w2=w16,w32,w2
  config=json.loads((ROOT/'native_prefill_stage1_config16k.json').read_text())
  self.flags=torch.tensor(config['flags'],device=w16.device,dtype=torch.int32)
  self.logical=torch.tensor(config['logical'],device=w16.device,dtype=torch.int32)
  self.logical_mi32=torch.tensor([128 if f==4 else v for f,v in zip(config['flags'],config['logical'])],device=w16.device,dtype=torch.int32)
  self.logical_lsu4=torch.full_like(self.logical,32)
  self.logical_wide=torch.tensor([{5:192,18:160,20:224,21:256}.get(f,l) for f,l in zip(config["flags"],config["logical"])],device=w16.device,dtype=torch.int32)
  self.logical_wide32=torch.tensor([{19:192,23:160}.get(f,l) for f,l in zip(config["flags"],config["logical"])],device=w16.device,dtype=torch.int32)
  self.modules=[_module(i) if i!=5 else None for i in range(7)]
  global _BOUNDS
  if _BOUNDS is None:
   # Bounds depend only on CPU shape and the verified dispatch table, never on
   # route values. They cap empty blocks without GPU synchronization.
   bounds=[[(0,0) for _ in range(16385)] for _ in range(7)]
   max_blocks=[0]*7;min_rows=[1<<30]*7;min_counts=[1<<30]*7
   for m in range(1,16385):
    kind=config['flags'][m];rows=config['logical'][m]
    g=6 if kind in (19,23) else 3 if kind==2 else 2 if kind in (3,10,11,12,16,17) else 4 if kind==22 else 1 if kind in (1,4,9,19,23) else 0
    if g==6:rows={19:192,23:160}[kind]
    if g==5:rows={5:192,18:160,20:224,21:256}[kind]
    if g==1 and kind==4:rows=128
    if g==3:rows=32
    max_blocks[g]=max(max_blocks[g],(m+rows-1)//rows)
    min_rows[g]=min(min_rows[g],rows);min_counts[g]=min(min_counts[g],m)
    for g in range(7):
     active=min(64,(m*8)//min_counts[g])
     cap=min(64*max_blocks[g],(m*8)//min_rows[g]+active)
     bounds[g][m]=(max_blocks[g],cap)
 
   _BOUNDS=bounds
  self.bounds=_BOUNDS
 def prepare(self,ids,weights):
  assert ids.shape[0]<=16384 and ids.shape[1]==8
  si,sw,se,nv,_=moe_sorting(ids,weights.float(),64,2304,torch.bfloat16,64)
  counts=expert_counts(ids)
  return prepare_all(ids,si,sw,nv,self.flags,self.logical,self.logical_mi32,self.logical_lsu4,self.logical_wide,self.logical_wide32,self.bounds,counts)

 def run(self,x,plans,out):
  for g,p in enumerate(plans):
   if p is None:continue
   si,sw,se,plan=p
   block=[64,128,64,32,128,256,192][g]
   name=f'moe_ck2stages_gemm1_256x{block}x128x64_1x4_TypeCast_v3_Nswizzle0_Quant0_MulRoutedWeight0_silu_B16_B16_B16'
   if g==6:name='moe_ck2stages_gemm1_256x256x128x64_1x4_TypeCast_v3_Nswizzle0_Quant0_MulRoutedWeight0_silu_B16_B16_B16'
   if g==4:name=name.replace('x128x64_1x4','x128x128_1x4')
   if g in (2,3,5):name=name.replace('x128x64_1x4_TypeCast_v3','x64x64_1x4_TypeCast_v1')
   self.modules[g].ck_moe_stage1(x,self.w32 if g in (1,4,6) else self.w16,self.w2,si,se,plan,out,8,kernelName=name,block_m=block,activation=int(ActivationType.Silu),splitk=1)
  return out

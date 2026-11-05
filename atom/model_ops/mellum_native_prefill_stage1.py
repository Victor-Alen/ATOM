"""Experimental native CK prefill stage1 split by MFMA/accumulation class."""
from pathlib import Path
import json,os,sys,types,importlib.util
import torch
from aiter import ActivationType
from aiter.fused_moe import moe_sorting
from atom.model_ops.mellum_native_prefill_routes import repack,expert_counts
ROOT=Path(os.environ['ATOM_MELLUM_PREFILL_NATIVE_ROOT'])
_MOD='module_moe_ck2stages_b16_b16_preshuffle_on_b16_silu_no_mulWeightStage2'
_MODULES={}
def _module(group):
 if group not in _MODULES:
  folder=['native_moe_prefill_long16_abi_jit','native_moe_prefill_long32_abi_jit','native_moe_prefill_lsu2_abi_jit','native_moe_prefill_lsu4_abi_jit','native_moe_prefill_long32k32_abi_jit'][group]
  package='mellum_prefill_abi_group_'+str(group);sys.modules[package]=types.ModuleType(package)
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
  self.modules=[_module(i) for i in range(5)]
 def prepare(self,ids,weights):
  assert ids.shape[0]<=16384 and ids.shape[1]==8
  si,sw,se,nv,_=moe_sorting(ids,weights.float(),64,2304,torch.bfloat16,64)
  counts=expert_counts(ids)
  result=[]
  for g in range(5):
   physical=128 if g in (1,4) else (32 if g==3 else 64)
   logical=self.logical_mi32 if g in (1,4) else (self.logical_lsu4 if g==3 else self.logical)
   result.append(repack(ids,si,sw,nv,self.flags,logical,lookup=True,physical=physical,source_physical=64,group=[0,1,4,3,5][g],given_counts=counts))
  return result

 def run(self,x,plans,out):
  for g,(si,sw,se,plan) in enumerate(plans):
   block=128 if g in (1,4) else (32 if g==3 else 64)
   name=f'moe_ck2stages_gemm1_256x{block}x128x64_1x4_TypeCast_v3_Nswizzle0_Quant0_MulRoutedWeight0_silu_B16_B16_B16'
   if g==4:name=name.replace('x128x64_1x4','x128x128_1x4')
   if g in (2,3):name=name.replace('x128x64_1x4_TypeCast_v3','x64x64_1x4_TypeCast_v1')
   self.modules[g].ck_moe_stage1(x,self.w32 if g in (1,4) else self.w16,self.w2,si,se,plan,out,8,kernelName=name,block_m=block,activation=int(ActivationType.Silu),splitk=1)
  return out

"""Opt-in original CK V1 M32 group0, with fused route preparation."""
from pathlib import Path
import os,json,types,sys,importlib.util
import torch
from aiter.fused_moe import moe_sorting
from atom.model_ops.mellum_native_prefill_stage1_hybrid import Stage1 as Parent
from atom.model_ops.mellum_native_prefill_hybrid_core import expert_counts
from atom.model_ops.mellum_native_prefill_rotv1m32_routes import prepare_all
ROOT=Path(os.environ['ATOM_MELLUM_PREFILL_NATIVE_ROOT'])
_MODULE=None
_NAME='moe_ck2stages_gemm1_256x32x64x64_1x4_TypeCast_v1_Nswizzle0_Quant0_MulRoutedWeight0_silu_B16_B16_B16'

def _native():
 global _MODULE
 if _MODULE is None:
  name='module_moe_ck2stages_b16_b16_preshuffle_on_b16_silu_no_mulWeightStage2';package='mellum_prefill_rotv1m32_native'
  sys.modules[package]=types.ModuleType(package)
  spec=importlib.util.spec_from_file_location(package+'.'+name,ROOT/'native_moe_prefill_rotv1m32_jit'/(name+'.so'))
  _MODULE=importlib.util.module_from_spec(spec);spec.loader.exec_module(_MODULE)
 return _MODULE

class _Group0:
 def ck_moe_stage1(self,*args,**kwargs):
  kwargs['kernelName']=_NAME;kwargs['block_m']=32
  return _native().ck_moe_stage1(*args,**kwargs)

class Stage1(Parent):
 def __init__(self,*args):
  super().__init__(*args)
  _native();self.modules=list(self.modules);self.modules[0]=_Group0()
  self.logical_group0=torch.full_like(self.logical,32)
  config=json.loads((ROOT/'native_prefill_stage1_config16k.json').read_text())
  bounds=[(0,0)];max_blocks=0;min_count=1<<30
  other={1,4,9,19,23,2,3,10,11,12,16,17,22}
  for m in range(1,16385):
   if config['flags'][m] not in other:
    max_blocks=max(max_blocks,(m+31)//32);min_count=min(min_count,m)
   active=min(64,(m*8)//min_count)
   bounds.append((max_blocks,min(64*max_blocks,(m*8)//32+active)))
  self.bounds=list(self.bounds);self.bounds[0]=bounds
 def prepare(self,ids,weights,shared=None):
  assert ids.shape[0]<=16384 and ids.shape[1]==8
  if shared is None:
   si,sw,se,nv,_=moe_sorting(ids,weights.float(),64,2304,torch.bfloat16,64)
   counts=expert_counts(ids)
  else:
   si,sw,nv,counts=shared
  return prepare_all(ids,si,sw,nv,self.flags,self.logical,self.logical_mi32,self.logical_lsu4,self.logical_wide,self.logical_wide32,self.bounds,counts,self.logical_group0)

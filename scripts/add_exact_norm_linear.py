from pathlib import Path
root=Path('/workspace/ATOM-general-norm/atom/model_ops')
p=root/'layernorm.py';s=p.read_text()
anchor='        if (envs.ATOM_MELLUM_HF_NORM_PREFILL and x.ndim==2'
start=s.index(anchor)
block='''        if (__import__('os').environ.get('ATOM_MELLUM_EXACT_NORM')=='1'
                and not self.use_fused_quant and self.tp_size==1
                and x.dtype==self.weight.dtype==torch.bfloat16
                and getattr(get_current_atom_config().hf_config,'model_type',None)=='mellum'):
            summed=x if residual is None else x+residual
            values=summed.float()
            variance=values.pow(2).mean(-1,keepdim=True)
            normalized=(values*torch.rsqrt(variance+self.eps)).to(summed.dtype)
            result=self.weight*normalized
            return result if residual is None else (result,summed)

'''
s=s[:start]+block+s[start:];p.write_text(s)
p=root/'linear.py';s=p.read_text()
anchor='''        if self.quant_type.value == QuantType.No.value:
            y = tgemm.mm('''
assert s.count(anchor)==1
s=s.replace(anchor,'''        if (__import__('os').environ.get('ATOM_MELLUM_EXACT_LINEAR')=='1'
                and self.quant_type.value==QuantType.No.value and self.tp_size==1
                and x.dtype==self.weight.dtype==torch.bfloat16
                and getattr(get_current_atom_config().hf_config,'model_type',None)=='mellum'):
            return torch.nn.functional.linear(x,self.weight,self.bias)
'''+anchor)
anchor='''    def forward(self,x:torch.Tensor,x_scale=None,otype=dtypes.bf16):
'''
assert s.count(anchor)==1
s=s.replace(anchor,anchor+'''        if (__import__('os').environ.get('ATOM_MELLUM_EXACT_LINEAR')=='1'
                and self.tp_size==1 and self.quant_type==QuantType.No
                and x.dtype==self.weight.dtype==torch.bfloat16
                and getattr(get_current_atom_config().hf_config,'model_type',None)=='mellum'):
            sizes=(self.num_heads*self.head_size,self.num_kv_heads*self.head_size,self.num_kv_heads*self.v_head_size)
            weights=self.weight.split(sizes,dim=0)
            biases=(None,None,None) if self.bias is None else self.bias.split(sizes)
            return torch.cat([torch.nn.functional.linear(x,w,b) for w,b in zip(weights,biases)],dim=-1)
''')
p.write_text(s)

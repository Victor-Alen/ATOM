"""Opt-in, armed capture of one real generation step; no arithmetic overrides."""
import os
from pathlib import Path
import torch

RECORDER=None

class TensorRecorder:
    def __init__(self,outdir,target):
        self.root=Path(outdir); self.root.mkdir(parents=True,exist_ok=True)
        self.target=target; self.active=False; self.data={}; self.detail={}; self.step=-1
    def begin(self,step):
        self.step=step; self.active=step==self.target
        if self.active: self.data={}
    def add(self,key,value):
        if not self.active: return
        if torch.is_tensor(value):
            t=value.detach()
            if t.ndim>=2: t=t.reshape(-1,t.shape[-1])[-1:]
            self.data[key]=t.cpu().clone()
    def pre(self,tag):
        def hook(module,args,kwargs):
            if not self.active: return
            for i,x in enumerate(args): self.add(f'{tag}.in{i}',x)
            for name in ('hidden_states','input_ids','positions','position_ids'):
                if name in kwargs: self.add(f'{tag}.kw.{name}',kwargs[name])
        return hook
    def post(self,tag):
        def hook(module,args,kwargs,output):
            if not self.active: return
            values=output if isinstance(output,(tuple,list)) else [output]
            for i,x in enumerate(values): self.add(f'{tag}.out{i}',x)
            if hasattr(module,'weight') and ('norm' in tag or tag.endswith('.router')):
                self.data[f'{tag}.weight']=module.weight.detach().cpu().clone()
        return hook
    def attach(self,model):
        self.handles=[]
        for i,layer in enumerate(model.layers):
            for label,name in [('input_norm','input_layernorm'),('attn','self_attn'),('post_norm','post_attention_layernorm'),('mlp','mlp')]:
                sub=getattr(layer,name); tag=f'L{i}.{label}'
                self.handles.append(sub.register_forward_pre_hook(self.pre(tag),with_kwargs=True))
                self.handles.append(sub.register_forward_hook(self.post(tag),with_kwargs=True))
            tag=f'L{i}.router'; sub=layer.mlp.gate
            self.handles.append(sub.register_forward_pre_hook(self.pre(tag),with_kwargs=True))
            self.handles.append(sub.register_forward_hook(self.post(tag),with_kwargs=True))
            tag=f'L{i}.layer'
            self.handles.append(layer.register_forward_hook(self.post(tag),with_kwargs=True))
        if os.environ.get('ATOM_STEP_TRACE_L0_DETAIL')=='1':
            for name,sub in [('input_norm',model.layers[0].input_layernorm),
                             ('post_norm',model.layers[0].post_attention_layernorm),
                             ('mlp',model.layers[0].mlp)]:
                def pre_full(module,args,kwargs,name=name):
                    if not self.active:return
                    for key,t in enumerate(args):
                        if torch.is_tensor(t):self.detail[f'L0.full.{name}.in{key}']=t.detach().cpu().clone()
                def post_full(module,args,kwargs,out,name=name):
                    if not self.active:return
                    for key,t in enumerate(out if isinstance(out,(tuple,list)) else [out]):
                        if torch.is_tensor(t):self.detail[f'L0.full.{name}.out{key}']=t.detach().cpu().clone()
                self.handles.append(sub.register_forward_pre_hook(pre_full,with_kwargs=True))
                self.handles.append(sub.register_forward_hook(post_full,with_kwargs=True))
            inner=model.layers[0].self_attn.attn.impl
            original_rope_cache=inner.rope_cache
            def rope_cache(*args,**kwargs):
                result=original_rope_cache(*args,**kwargs)
                if self.active:
                    ctx=kwargs.get('fwd_ctx',args[-1] if args else None)
                    am=ctx.attn_metadata
                    import inspect
                    self.detail["rope.class_file"]=inspect.getfile(type(inner.rotary_emb))
                    self.detail["rope.hf_inv_freq_flag"]=os.getenv("AITER_MELLUM_HF_ROPE_INV_FREQ")
                    if am.block_tables is None: return result
                    q,k,v,kc,vc,ks,vs=result
                    n=int(am.context_lens[0]);bs=kc.shape[3] if kc.ndim==5 else kc.shape[1]
                    blocks=am.block_tables[0,:(n+bs-1)//bs].long()
                    self.detail['kv.k_raw']=kc.index_select(0,blocks).detach().cpu().clone()
                    self.detail['kv.v_raw']=vc.index_select(0,blocks).detach().cpu().clone()
                    self.detail['kv.context_lens']=am.context_lens.detach().cpu().clone()
                    self.detail['kv.block_tables']=am.block_tables.detach().cpu().clone()
                    self.detail['kv.slot_mapping']=am.slot_mapping.detach().cpu().clone()
                    self.detail['kv.format']=getattr(inner,'_cache_format',None)
                    self.detail['kv.use_flash_layout']=inner.use_flash_layout
                    self.detail['kv.q']=q.detach().cpu().clone()
                    self.detail['kv.k']=k.detach().cpu().clone()
                    self.detail['kv.v']=v.detach().cpu().clone()
                return result
            inner.rope_cache=rope_cache
            for name in ('qkv_proj','q_norm','k_norm','rotary_emb','attn','o_proj'):
                sub=getattr(model.layers[0].self_attn,name)
                def before(module,args,kwargs,name=name):
                    if self.step<0 or self.step>self.target: return
                    for key,value in list(enumerate(args))+list(kwargs.items()):
                        if torch.is_tensor(value):
                            self.detail[f'L0.step{self.step}.{name}.in{key}']=value.detach().cpu().clone()
                def after(module,args,kwargs,value,name=name):
                    if self.step<0 or self.step>self.target: return
                    if name=='attn':
                        for key,t in kwargs.items():
                            if torch.is_tensor(t):
                                self.detail[f'L0.step{self.step}.{name}.after{key}']=t.detach().cpu().clone()
                    values=value if isinstance(value,(tuple,list)) else [value]
                    for key,t in enumerate(values):
                        if torch.is_tensor(t):
                            self.detail[f'L0.step{self.step}.{name}.out{key}']=t.detach().cpu().clone()
                self.handles.append(sub.register_forward_pre_hook(before,with_kwargs=True))
                self.handles.append(sub.register_forward_hook(after,with_kwargs=True))
        self.handles.append(model.norm.register_forward_pre_hook(self.pre('final_norm'),with_kwargs=True))
        self.handles.append(model.norm.register_forward_hook(self.post('final_norm'),with_kwargs=True))
    def save(self):
        if self.active:
            torch.save({'step':self.step,'tensors':{**self.detail,**self.data}},self.root/f'step_{self.step}.pt')

def install_step_trace(model):
    global RECORDER
    if RECORDER is not None: return
    rec=TensorRecorder(os.environ['ATOM_STEP_TRACE_DIR'],int(os.environ.get('ATOM_STEP_TRACE_STEP','27')))
    rec.attach(model); RECORDER=rec
    original=model.forward; counter=[0]
    def forward(*args,**kwargs):
        if not (rec.root/'ARMED').is_file(): return original(*args,**kwargs)
        rec.begin(counter[0]); counter[0]+=1
        rec.add('model.input_ids',kwargs.get('input_ids'))
        rec.add('model.positions',kwargs.get('positions'))
        try: return original(*args,**kwargs)
        finally: rec.save()
    model.forward=forward

def record_moe(layer_name,x,ids,weights,block_m):
    rec=RECORDER
    if rec is None or not rec.active: return
    import re
    idx=int(re.search(r'layers\.(\d+)\.',layer_name).group(1))
    rec.add(f'L{idx}.routing.input',x); rec.add(f'L{idx}.routing.ids',ids)
    rec.add(f'L{idx}.routing.weights',weights)
    rec.data[f'L{idx}.routing.block_m']=block_m
    if idx==0 and os.environ.get('ATOM_STEP_TRACE_L0_DETAIL')=='1':
        for name,t in [('input',x),('ids',ids),('weights',weights)]:
            rec.detail[f'L0.full.routing.{name}']=t.detach().cpu().clone()


def capture_head(logits,x,weight):
    rec=RECORDER
    if rec is None or not rec.active: return
    rec.add('head.input',x); rec.add('head.logits',logits)
    candidates=[int(v) for v in os.environ.get('ATOM_STEP_TRACE_CANDIDATES','').split(',') if v]
    candidates=sorted(set(candidates+logits[-1].topk(10).indices.tolist()))
    rec.data['head.weight_ids']=torch.tensor(candidates)
    rec.data['head.weight_rows']=weight[candidates].detach().cpu().clone()
    rec.save()

def capture_sampler(logits,temperatures):
    rec=RECORDER
    if rec is None or not rec.active: return
    rec.add('sampler.logits',logits); rec.add('sampler.temperatures',temperatures);rec.save()


def record_moe_intermediate(layer_name,name,value):
    rec=RECORDER
    if rec is None or not rec.active or os.environ.get('ATOM_STEP_TRACE_L0_DETAIL')!='1':return
    import re
    idx=int(re.search(r'layers\.(\d+)\.',layer_name).group(1))
    if idx==0:rec.detail[f'L0.full.moe.{name}']=value.detach().cpu().clone()


def record_moe_weights(layer_name,w13,w2):
    rec=RECORDER
    if rec is None or not rec.active or os.environ.get('ATOM_STEP_TRACE_L0_DETAIL')!='1':return
    import re
    idx=int(re.search(r'layers\.(\d+)\.',layer_name).group(1))
    if idx!=0:return
    path=rec.root/'L0_weights.pt'
    if path.exists():return
    torch.save({'w13':w13.detach().cpu(),'w2':w2.detach().cpu(),
                'w13_shuffled':getattr(w13,'is_shuffled',False),
                'w2_shuffled':getattr(w2,'is_shuffled',False)},path)

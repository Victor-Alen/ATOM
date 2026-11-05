"""Opt-in synchronized phase diagnostics; never enable for throughput scores."""
import functools,json,os,time
import torch
_ENABLED=os.environ.get("ATOM_MTP_PHASE_TIMING","0")=="1"
_STACK=[]
def phase(name):
 def decorate(fn):
  if not _ENABLED:return fn
  @functools.wraps(fn)
  def wrapped(self,*args,**kwargs):
   batch=kwargs.get("batch")
   if batch is None:
    for x in args:
     if hasattr(x,"total_seqs_num"):batch=x;break
   row={"phase":name,"batch":getattr(batch,"total_seqs_num",None),"tokens":getattr(batch,"total_tokens_num",None),"prefill_seqs":getattr(batch,"total_seqs_num_prefill",None)}
   torch.cuda.synchronize()
   start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
   frame={"children_gpu_ms":0.,"children_wall_ms":0.};_STACK.append(frame)
   wall=time.perf_counter();start.record()
   try:return fn(self,*args,**kwargs)
   finally:
    end.record();end.synchronize();wall_ms=(time.perf_counter()-wall)*1000;gpu_ms=start.elapsed_time(end);_STACK.pop()
    row.update(gpu_envelope_ms=gpu_ms,wall_ms=wall_ms,exclusive_gpu_envelope_ms=max(0.,gpu_ms-frame["children_gpu_ms"]),exclusive_wall_ms=max(0.,wall_ms-frame["children_wall_ms"]))
    if _STACK:_STACK[-1]["children_gpu_ms"]+=gpu_ms;_STACK[-1]["children_wall_ms"]+=wall_ms
    path=os.environ["ATOM_MTP_PHASE_TIMING_PATH"]+"."+str(os.getpid())+".jsonl"
    with open(path,"a") as f:f.write(json.dumps(row)+"\n")
  return wrapped
 return decorate

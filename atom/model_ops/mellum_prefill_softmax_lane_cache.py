"""Full-attention softmax cached in the frozen ATen lane/ILP accumulation order."""
import torch
import triton as tr
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
def _scaled(S,base,pos,valid,qpos,shift,SCALE:gl.constexpr):
    ok=valid & (pos+shift<=qpos)
    raw=gl.load(S+base+pos,ok,0).to(gl.float32)
    return gl.where(ok,(raw*SCALE).to(gl.bfloat16).to(gl.float32),float('-inf'))

@gluon.jit
def _exp(x):
    y=gl.exp(x)
    return gl.where(y<1.1754943508222875e-38,0.0,y)

@gluon.jit
def _row(values,b:gl.constexpr,B:gl.constexpr):
    idx=gl.full((1,512),b,gl.int32,gl.BlockedLayout([1,1],[1,64],[1,8],[1,0]))
    return gl.sum(gl.gather(values,idx,axis=0),axis=0)

@gluon.jit
def _lane_cache(S,P,N,Q,CTX,SCALE:gl.constexpr,B:gl.constexpr,ALIGNED:gl.constexpr):
    row=gl.program_id(0);base=row.to(gl.int64)*N
    qpos=row%Q+CTX-Q;key_shift=CTX-N
    layout:gl.constexpr=gl.BlockedLayout([1,1],[1,64],[1,8],[1,0])
    lanes=gl.arange(0,512,layout=gl.SliceLayout(0,layout))
    blocks=gl.arange(0,B,layout=gl.SliceLayout(1,layout))
    if ALIGNED:
        start=0;bulk=gl.cdiv(N,2048)*2048
    else:
        shift=((row%4)*(N%4))%4
        start=gl.where(shift>0,512-shift,0)
        bulk=((N-start)//2048)*2048
    pos=start+blocks[:,None]*2048+lanes[None,:]*4
    ok=blocks[:,None]*2048<bulk
    x0=_scaled(S,base,pos+0,ok&(pos+0<N),qpos,key_shift,SCALE)
    x1=_scaled(S,base,pos+1,ok&(pos+1<N),qpos,key_shift,SCALE)
    x2=_scaled(S,base,pos+2,ok&(pos+2<N),qpos,key_shift,SCALE)
    x3=_scaled(S,base,pos+3,ok&(pos+3<N),qpos,key_shift,SCALE)
    m=gl.maximum(gl.maximum(gl.max(x0,0),gl.max(x1,0)),gl.maximum(gl.max(x2,0),gl.max(x3,0)))
    if not ALIGNED:
        prefix=lanes-shift
        xp=_scaled(S,base,prefix,(shift>0)&(prefix>=0)&(prefix<N),qpos,key_shift,SCALE)
        tail=start+bulk+lanes
        xt0=_scaled(S,base,tail,tail<N,qpos,key_shift,SCALE)
        xt1=_scaled(S,base,tail+512,tail+512<N,qpos,key_shift,SCALE)
        xt2=_scaled(S,base,tail+1024,tail+1024<N,qpos,key_shift,SCALE)
        xt3=_scaled(S,base,tail+1536,tail+1536<N,qpos,key_shift,SCALE)
        m=gl.maximum(m,gl.maximum(xp,gl.maximum(gl.maximum(xt0,xt1),gl.maximum(xt2,xt3))))
    mx=gl.max(m,0)
    e0=_exp(x0-mx);e1=_exp(x1-mx);e2=_exp(x2-mx);e3=_exp(x3-mx)
    if ALIGNED:
        acc=gl.full((512,),0.0,gl.float32,gl.SliceLayout(0,layout))
    else:
        ep=_exp(xp-mx);et0=_exp(xt0-mx);et1=_exp(xt1-mx);et2=_exp(xt2-mx);et3=_exp(xt3-mx)
        acc=ep
    for block in gl.static_range(B):
        acc=acc+_row(e0,block,B)
        acc=acc+_row(e1,block,B)
        acc=acc+_row(e2,block,B)
        acc=acc+_row(e3,block,B)
    if not ALIGNED:
        acc=acc+et0;acc=acc+et1;acc=acc+et2;acc=acc+et3
    for step in gl.static_range(6):
        src=lanes//64*64+gl.minimum(lanes%64+(32>>step),63)
        acc=acc+gl.gather(acc,src,axis=0)
    warp=gl.gather(acc,gl.minimum(lanes,7)*64,axis=0)
    warp=gl.where(lanes<8,warp,0.0)
    for step in gl.static_range(6):
        warp=warp+gl.gather(warp,gl.minimum(lanes+(32>>step),63),axis=0)
    den=gl.sum(gl.where(lanes==0,warp,0.0),0)
    inv=gl.div_rn(1.0,den)
    if ALIGNED:
        packed:gl.constexpr=gl.BlockedLayout([1,1,4],[1,64,1],[1,8,1],[2,1,0])
        e0p=gl.convert_layout(e0,gl.SliceLayout(2,packed))
        e1p=gl.convert_layout(e1,gl.SliceLayout(2,packed))
        e2p=gl.convert_layout(e2,gl.SliceLayout(2,packed))
        e3p=gl.convert_layout(e3,gl.SliceLayout(2,packed))
        js=gl.arange(0,4,layout=gl.SliceLayout(0,gl.SliceLayout(1,packed)))
        values=gl.where(js[None,None,:]==0,e0p[:,:,None],gl.where(js[None,None,:]==1,e1p[:,:,None],gl.where(js[None,None,:]==2,e2p[:,:,None],e3p[:,:,None])))
        posp=gl.convert_layout(pos,gl.SliceLayout(2,packed))[:,:,None]+js[None,None,:]
        okp=gl.convert_layout(ok & (pos>=0),gl.SliceLayout(2,packed))[:,:,None]&(posp<N)
        gl.store(P+base+posp,(values*inv).to(gl.bfloat16),okp)
    else:
        gl.store(P+base+pos+0,(e0*inv).to(gl.bfloat16),ok&(pos+0<N))
        gl.store(P+base+pos+1,(e1*inv).to(gl.bfloat16),ok&(pos+1<N))
        gl.store(P+base+pos+2,(e2*inv).to(gl.bfloat16),ok&(pos+2<N))
        gl.store(P+base+pos+3,(e3*inv).to(gl.bfloat16),ok&(pos+3<N))
    if not ALIGNED:
        gl.store(P+base+prefix,(ep*inv).to(gl.bfloat16),(shift>0)&(prefix>=0)&(prefix<N))
        gl.store(P+base+tail,(et0*inv).to(gl.bfloat16),tail<N)
        gl.store(P+base+tail+512,(et1*inv).to(gl.bfloat16),tail+512<N)
        gl.store(P+base+tail+1024,(et2*inv).to(gl.bfloat16),tail+1024<N)
        gl.store(P+base+tail+1536,(et3*inv).to(gl.bfloat16),tail+1536<N)

def softmax(scores,query_count,context_length,window,scale):
    assert scores.is_contiguous() and scores.dtype==torch.bfloat16 and not window
    n=scores.shape[-1];assert 2048<n<=16384
    out=torch.empty_like(scores)
    _lane_cache[(scores.numel()//n,)](scores,out,n,query_count,context_length,scale,tr.next_power_of_2(tr.cdiv(n,2048)),n%4==0,num_warps=8,enable_fp_fusion=False)
    return out


def try_softmax(scores,query_count,context_length,window,scale):
    # Keep the accepted short-row/SWA implementation and avoid under-filled caches.
    # This is a shape-only occupancy guard, independent of prompt content.
    n=scores.shape[-1]
    if window is not None and window>0:return None
    if not (4096<n<=16384 and query_count==context_length==n):return None
    if scores.dtype!=torch.bfloat16 or not scores.is_contiguous():return None
    capacity=tr.next_power_of_2(tr.cdiv(n,2048))*2048
    if n*3<capacity*2:return None
    return softmax(scores,query_count,context_length,None,scale)

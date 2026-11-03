from pathlib import Path
import subprocess,shutil
base=Path('/workspace/ck-mellum-up-fp64');ck=Path('/workspace/ck-mellum-hf-korder')
if not ck.exists():subprocess.run(['git','-C',str(base),'worktree','add','-b','mellum/hf-korder',str(ck),'14e5a4558'],check=True)
p=ck/'include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_gufusion_v3.hpp';s=(base/p.relative_to(ck)).read_text()
s=s.replace('is_same_v<BDataType, bhalf_t>;','is_same_v<BDataType, bhalf_t>\n#ifndef CK_MELLUM_HF_KORDER\n                                     ;\n#else\n                                     && false;\n#endif',1)
needle='''                            using mfma_input_type ='''
# Same helper used at every hot loop and tail, with no output-coordinate exceptions.
helper='''        // Experimental BF16 KPack=8, 16x16 MFMA source permutation.
        // HF MI32 LDS reads: K[0:4,8:12], then K[4:8,12:16].
        // CK packs K8 per lane: exchange register bit2 with KLane bit1.
        auto hf_korder = [&](auto& vec) {
#ifdef CK_MELLUM_HF_KORDER
            if constexpr(is_same_v<ADataType, bhalf_t> && is_same_v<BDataType, bhalf_t> &&
                         MPerXDL == 16 && NPerXDL == 16 && KPack == 8)
            {
                auto src = vec;
                const unsigned lane = get_thread_local_1d_id() % 64;
                static_for<0, 4, 1>{}([&](auto pair) {
                    constexpr unsigned reg = decltype(pair)::value;

                    // Select source register pair using the destination KLane high bit.
                    const unsigned lane_pair = (lane >> 5) * 2;
                    using pair_type = typename vector_type<ComputeDataType, 2>::type;
                    unsigned lo = bit_cast<unsigned>(src.template AsType<pair_type>()[Number<reg % 2>{}]);
                    unsigned hi = bit_cast<unsigned>(src.template AsType<pair_type>()[Number<reg % 2 + 2>{}]);
                    
                    const unsigned from = (lane & ~32u) | ((reg / 2) * 32u);
                    const unsigned lp = __builtin_amdgcn_ds_bpermute(from * 4, lo);
                    const unsigned hp = __builtin_amdgcn_ds_bpermute(from * 4, hi);
                    vec.template AsType<pair_type>()(pair) =
                        bit_cast<pair_type>(lane_pair ? hp : lp);
                });
            }
#endif
        };
'''
s=s.replace('        // main body\n',helper+'\n        // main body\n',1)
# Insert before every mfma_input_type declaration (indent varies).
import re
s,n=re.subn(r'(\s+using mfma_input_type\s*=)',r'\n                    hf_korder(a_thread_vec);\n                    hf_korder(b_thread_vec);\n                    hf_korder(b_thread_vec_up);\1',s)
assert n==4,n
p.write_text(s)
old=Path('/workspace/ck-mellum-up-fp64-jit');new=Path('/workspace/ck-mellum-hf-korder-jit');new.mkdir(exist_ok=True)
for f in old.glob('*.so'):
 t=new/f.name
 if not t.exists():t.symlink_to(f)
mod='module_moe_ck2stages_b16_b16_preshuffle_on_b16_silu_no_mulWeightStage2';oldb=old/'build'/mod;newb=new/'build'/mod
if not newb.exists():shutil.copytree(oldb,newb,symlinks=True)
ninja=newb/'build/build.ninja';txt=(oldb/'build/build.ninja').read_text().replace(str(old),str(new)).replace(str(base),str(ck));txt=txt.replace('cuda_cflags = ','cuda_cflags = -DCK_MELLUM_HF_KORDER=1 ',1);ninja.write_text(txt)
# Only one selected stage1 object is rebuilt. Existing stage2 and other objects remain byte-identical.
obj='moe_ck2stages_gemm1_256x128x128x64_1x4_TypeCast_v3_Nswizzle0_Quant0_MulRoutedWeight0_silu_B16_B16_B16.cuda.o'
(newb/'build'/obj).unlink(missing_ok=True)
print('READY',ck,newb,obj,flush=True)
subprocess.run(['ninja','-v',obj],cwd=newb/'build',check=True)
# Link explicitly to avoid implicitly rebuilding every unrelated object.
lines=txt.splitlines();link=next(x for x in lines if x.startswith('build '+mod+'.so: link '));objects=link.split(': link ',1)[1].split();ldflags=next(x.split(' = ',1)[1] for x in lines if x.startswith('ldflags = '));out=new/(mod+'.so')
if out.is_symlink():out.unlink()
subprocess.run(['c++',*objects,*__import__('shlex').split(ldflags),'-o',str(out)],cwd=newb/'build',check=True)
print('BUILT',out,flush=True)





from pathlib import Path
import subprocess,shutil,shlex,json,hashlib
oldck=Path('/workspace/ck-mellum-hf-korder');ck=Path('/workspace/ck-mellum-hf-profile')
if not ck.exists():subprocess.run(['git','-C',str(oldck),'worktree','add','-b','mellum/hf-profile-korder',str(ck),'80f4da570'],check=True)
rel='include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_gufusion_v3.hpp';p=ck/rel;s=(oldck/rel).read_text();s=s.replace('index_t num_loop) const','index_t num_loop, bool hf_mi32 = false) const',1);s=s.replace('''            {
                auto src = vec;''','''            {
                if(!hf_mi32) return;
                auto src = vec;''',1);p.write_text(s)
rel='include/ck/tensor_operation/gpu/grid/gridwise_moe_gemm.hpp';p=ck/rel;s=(oldck/rel).read_text();s=s.replace('''    {
        if(mode == 0)''','''    {
        mode &= 0xff; // High bits carry opt-in stage1 metadata ABI flags.
        if(mode == 0)''',1);s=s.replace('''                    this->MellumStage1Mode = mode ? std::atoi(mode) : 0;''','''                    this->MellumStage1Mode = mode ? std::atoi(mode) : 0;
#ifdef CK_MELLUM_HF_KORDER
                    const char* plan = std::getenv("CK_MELLUM_STAGE1_KORDER_PLAN");
                    if(NumTokens_ > 1 && plan && std::atoi(plan) != 0) this->MellumStage1Mode |= 0x100;
#endif''',1)
# No Problem layout or argument ABI changes. Stage1 metadata extends num_valid_ids[0:2].
needle='''                c_thread_buf_up,
                num_k_block_main_loop);''';repl='''                c_thread_buf_up,
                num_k_block_main_loop
#ifdef CK_MELLUM_HF_KORDER
                , ((problem.MellumStage1Mode & 0x100) != 0 &&
                   p_max_token_id[2 + expert_id] != 0)
#endif
                );''';assert s.count(needle)==2,s.count(needle);s=s.replace(needle,repl);p.write_text(s)
old=Path('/workspace/ck-mellum-hf-korder-jit');new=Path('/workspace/ck-mellum-hf-profile-jit');new.mkdir(exist_ok=True);prod=Path('/workspace/aiter-mellum-norm-hf-rne-jit')
for f in prod.glob('*.so'):
 t=new/f.name
 if not t.exists():t.symlink_to(f)
mod='module_moe_ck2stages_b16_b16_preshuffle_on_b16_silu_no_mulWeightStage2';oldb=old/'build'/mod;newb=new/'build'/mod
if not newb.exists():shutil.copytree(oldb,newb,symlinks=True)
ninja=newb/'build/build.ninja';txt=(oldb/'build/build.ninja').read_text().replace(str(old),str(new)).replace(str(oldck),str(ck));ninja.write_text(txt)
objects_to_build = [line.split()[1].rstrip(":") for line in txt.splitlines()
                    if line.startswith("build moe_ck2stages_gemm1_") and "_v3_" in line]
assert len(objects_to_build) == 5
for obj in objects_to_build:
    (newb / "build" / obj).unlink(missing_ok=True)
print("BUILDING_ALL_V3_STAGE1", len(objects_to_build), flush=True)
subprocess.run(["ninja", "-j4", *objects_to_build], cwd=newb / "build", check=True)
link=next(x for x in txt.splitlines() if x.startswith('build '+mod+'.so: link '));objects=link.split(': link ',1)[1].split();ldflags=next(x.split(' = ',1)[1] for x in txt.splitlines() if x.startswith('ldflags = '));out=new/(mod+'.so')
if out.is_symlink():out.unlink()
temporary = out.with_suffix('.so.next')
with (new/'link.log').open('w') as f:subprocess.run(['c++',*objects,*shlex.split(ldflags),'-o',str(temporary)],cwd=newb/'build',stdout=f,stderr=f,check=True)
temporary.replace(out)

profile_path = Path('/workspace/ATOM-tritonmoe/atom/model_ops/mellum_hf_gateup_profiles.json')
profile_data = json.loads(profile_path.read_text())
profile_data['ck_stage1_metadata_abi'] = 1
profile_data['ck_binary_sha256'] = hashlib.sha256(out.read_bytes()).hexdigest()
profile_data['ck_v3_stage1_instances'] = len(objects_to_build)
profile_data['ck_source_sha256'] = {rel: hashlib.sha256((ck / rel).read_bytes()).hexdigest()
    for rel in ['include/ck/tensor_operation/gpu/grid/gridwise_moe_gemm.hpp',
                'include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_gufusion_v3.hpp']}
profile_path.write_text(json.dumps(profile_data,separators=(',',':'))+'\n')

print('BUILT',out,flush=True)

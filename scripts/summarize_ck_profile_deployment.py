from pathlib import Path
import json,hashlib,sys
sys.path[:0]=['/workspace/ATOM-tritonmoe','/workspace/aiter-mellum-norm-hf']
from atom.model_engine.llm_engine import _load_tokenizer
r=Path('/workspace/logs/step0_continuation/ck_hf_profile_prod');tokenizer=_load_tokenizer('/data/models/Mellum2.1-12B-A2.5B-Thinking');input_checks=[]
for s in ['A','C']:
 for line in (r/'fresh_hf'/f'{s}10_64.jsonl').read_text().splitlines():
  row=json.loads(line);same=tokenizer.encode(row['prompt'])==row['prompt_token_ids'];input_checks.append({'id':row['id'],'equal':same,'prompt_tokens':row['prompt_tokens']});assert same,row['id']
base=Path('/workspace/ck-mellum-up-fp64-jit/build/module_moe_ck2stages_b16_b16_preshuffle_on_b16_silu_no_mulWeightStage2/build');new=Path('/workspace/ck-mellum-hf-profile-jit/build/module_moe_ck2stages_b16_b16_preshuffle_on_b16_silu_no_mulWeightStage2/build');stage2=[]
for f in base.glob('moe_ck2stages_gemm2_*.o'):
 same=hashlib.sha256(f.read_bytes()).digest()==hashlib.sha256((new/f.name).read_bytes()).digest();assert same,f.name;stage2.append({'object':f.name,'equal':same})
old=json.loads((r/'baseline_pair.json').read_text());candidate=json.loads((r/'candidate_final_pair.json').read_text());perf=[]
for b,n in zip(old,candidate):
 assert b['id']==n['id'];perf.append({'id':b['id'],'baseline_seconds':b['elapsed_seconds'],'candidate_seconds':n['elapsed_seconds'],'baseline_tok_s':b['end_to_end_output_tokens_per_second'],'candidate_tok_s':n['end_to_end_output_tokens_per_second'],'throughput_change_percent':100*(n['end_to_end_output_tokens_per_second']/b['end_to_end_output_tokens_per_second']-1),'baseline_matches':b['matches'],'candidate_matches':n['matches'],'tokens':64})
report={'input_tokenization_checks':input_checks,'input_tokenization_all_equal':all(x['equal'] for x in input_checks),'stage2_objects_count':len(stage2),'stage2_objects_identical':all(x['equal'] for x in stage2),'paired_performance':perf,'A10':json.loads((r/'A10_64/summary.json').read_text()),'C10':json.loads((r/'C10_64/summary.json').read_text()),'fresh_hf_manifest':json.loads((r/'fresh_hf/manifest.json').read_text()),'actual_kernel':json.loads((r/'service_final/actual_kernel_validation.json').read_text()),'fixed_gpu0':json.loads((r/'fixed_validation_gpu0.json').read_text()),'fixed_gpu1':json.loads((r/'fixed_validation_gpu1.json').read_text())};(r/'final_summary.json').write_text(json.dumps(report,indent=2)+'\n');print('TOKENS_MATCH',len(input_checks),'STAGE2_IDENTICAL',len(stage2));print(json.dumps(perf,indent=2));print('A10',report['A10']['token_position_agreement'],'C10',report['C10']['token_position_agreement'])

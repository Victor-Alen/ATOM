import json,sys,resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
from human_eval.execution import check_correctness
if __name__=='__main__':
 resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
 d=json.loads(Path(sys.argv[1]).read_text());task=d['task'].copy();task['prompt']=''
 r=check_correctness(task,d['code'],timeout=3.0)
 print(json.dumps(r))

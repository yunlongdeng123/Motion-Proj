"""只测身份数据合同的关键反例，不启动GPU或导入训练模型。"""
from pathlib import Path
import sys,ast,json
import numpy as np
W=Path(__file__).resolve().parent;CODE=W/('bundle' if (W/'bundle').is_dir() else 'code');sys.path.insert(0,str(CODE))
from identity_common import compose_masks,ACTORS,OLD_ACTORS,NEW_ACTORS
old=np.zeros((4,5,3),bool);old[0,0,0]=True;old[0,1,1]=True;old[0,2,2]=True
new=np.zeros((4,5,2),bool);new[1,0,0]=True;new[1,1,1]=True
known=np.ones(2,bool);valid=np.ones_like(new)
checks=[]
labels,r=compose_masks(old,new,known,valid)
assert r['ready'] and np.array_equal(labels[0,:3],[1,2,3]) and np.array_equal(labels[1,:2],[4,5])
checks.append('three old IDs preserved; new IDs map to4/5')
unknown=valid.copy();unknown[3,4,0]=False
labels,r=compose_masks(old,new,known,unknown)
assert labels is None and not r['ready'] and r['missing'][0]['unknown_pixels']==1
checks.append('one unknown negative pixel blocks full-image baseline training')
empty=np.zeros_like(new)
labels,r=compose_masks(old,empty,known,valid)
assert r['ready'] and (labels<=3).all()
checks.append('explicitly known empty observation remains distinguishable from missing mask')
labels,r=compose_masks(old,empty,np.array([False,True]),valid)
assert labels is None and not r['ready']
checks.append('unknown empty observation is not zero-target supervision')
overlap=new.copy();overlap[0,0,0]=True
labels,r=compose_masks(old,overlap,known,valid)
assert labels is None and r['overlap_pixels']==1
checks.append('old/new overlap has no arbitrary identity winner')
bad=valid.copy();bad[1,0,0]=False
try:compose_masks(old,new,known,bad)
except ValueError:pass
else:raise AssertionError('positive outside valid domain accepted')
checks.append('invalid positive pixels rejected')
assert ACTORS==OLD_ACTORS+NEW_ACTORS==[5,26,18,6,37]
files=[]
for p in CODE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'));files.append(p.name)
assert 'torch' not in sys.modules
result=dict(status='CPU_DATA_CONTRACT_PASSED_GPU_NOT_RUN',checks=checks,python_AST=files,
    remote_input_coverage_checked_by_this_CPU_script=False,CUDA_renderer_tested_by_this_CPU_script=False,
    intended_training_roles=dict(GN11='existing30000',old_five_instances='fresh30000_same_old_method',R15='frozen4000_no_updates'))
(W/'cpu_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))

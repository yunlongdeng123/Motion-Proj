"""r49只增加参考空间绑定；复用r47输入与r48同预算控制。"""
from pathlib import Path
import json, os, sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
REPO = Path('/root/autodl-tmp/motion_proj_v77')
S = REPO/'scripts/worldsim_v77/target_protected'
T = Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')
PARENT = T/'r47'
CONTROL = T/'r48'
O = T/'r49'
sys.path[:0] = [str(Path(__file__).parent), str(S), str(S/'iteration14'),
               str(S/'iteration13'), str(S/'iteration12'), str(S/'iteration7'), str(S.parent)]
TRAIN_IDS = ['M010', 'M013', 'M018', 'M042']
EVAL_IDS = ['A034', 'A061_w08', 'A022', 'M003', 'M006']
DIAGNOSTIC_IDS = ['A034', 'A061_w08']
INITIAL = PARENT/'training/branch_0320.safetensors'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, data):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    tmp.replace(path)


def cases():
    return {c['case_id']:c for c in read(O/'manifest.json')['cases']}


def images(case, role):
    from iteration15.common import images as existing
    return existing(case, role)


def request(case):
    import numpy as np
    from interface import DeletionRequest
    p = PARENT/'inputs'/case['case_id']
    with np.load(p/'condition.npz') as data:
        priors = {k:data[k].copy() for k in data.files}
    with np.load(O/'inputs'/case['case_id']/'routing.npz') as data:
        route = {k:data[k].copy() for k in data.files if k.startswith('routing_')}
    x = images(case,'rgb');h = images(case,'hole')>0
    alpha = images(case,'alpha').astype('float32')/255 if case['kind']=='real' else h.astype('float32')
    result = DeletionRequest(x,h,alpha,priors,read(p/'references.json'),route)
    result.validate();return result

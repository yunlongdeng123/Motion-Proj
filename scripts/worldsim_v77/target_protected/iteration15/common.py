"""r48：复用r47不可变输入，短循环检查、训练与真实DEV验证。"""
from pathlib import Path
import json, os, sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
REPO = Path('/root/autodl-tmp/motion_proj_v77')
S = REPO/'scripts/worldsim_v77/target_protected'
T = Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')
PARENT = T/'r47'
O = T/'r48'
sys.path[:0] = [str(Path(__file__).parent), str(S/'iteration14'), str(S/'iteration13'),
               str(S/'iteration12'), str(S/'iteration7'), str(S), str(S.parent)]
TRAIN_IDS = ['M010', 'M013', 'M018', 'M042']
EVAL_64 = ['A034', 'A061_w08', 'A022', 'M003', 'M006']
EVAL_128 = EVAL_64 + ['A048', 'A041_w10']
DIAGNOSTIC_IDS = ['A034', 'A061_w08']
INITIAL = PARENT/'training/branch_0320.safetensors'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    tmp.replace(path)


def cases():
    return {c['case_id']: c for c in read(O/'manifest.json')['cases']}


def images(case, role):
    from PIL import Image
    import numpy as np
    folders = {'rgb': 'X', 'hole': 'model_hole', 'target': 'Y'} if case['kind']=='synthetic' else {
        'rgb': 'rgb', 'hole': 'model_mask', 'alpha': 'alpha'}
    files = sorted((Path(case['folder'])/folders[role]).glob(
        '*.jpg' if case['kind']=='real' and role=='rgb' else '*.png'))
    return np.stack([np.asarray(Image.open(files[i]).convert(
        'RGB' if role in ('rgb', 'target') else 'L')) for i in case['frame_indices']])


def request(case):
    import numpy as np
    from interface import DeletionRequest
    p = PARENT/'inputs'/case['case_id']
    with np.load(p/'condition.npz') as data:
        priors = {k: data[k].copy() for k in data.files}
    x = images(case, 'rgb'); h = images(case, 'hole')>0
    alpha = images(case, 'alpha').astype('float32')/255 if case['kind']=='real' else h.astype('float32')
    req = DeletionRequest(x, h, alpha, priors, read(p/'references.json'))
    req.validate()
    return req

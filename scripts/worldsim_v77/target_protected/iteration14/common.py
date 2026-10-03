"""r47多先验接口重构；官方r46基线独立保留。"""
from pathlib import Path
import json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
REPO = Path('/root/autodl-tmp/motion_proj_v77')
S = REPO/'scripts/worldsim_v77/target_protected'
T = Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')
A = Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-DELETE-AUDIT-20260928/r1')
O = T/'r47'
sys.path[:0] = [str(Path(__file__).parent), str(S/'iteration13'), str(S/'iteration12'), str(S/'iteration7'), str(S), str(S.parent)]


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def images(case, role):
    from PIL import Image
    import numpy as np
    folders = {'rgb': 'X', 'hole': 'model_hole', 'target': 'Y'} if case['kind']=='synthetic' else {
        'rgb': 'rgb', 'hole': 'model_mask', 'alpha': 'alpha'}
    files = sorted((Path(case['folder'])/folders[role]).glob('*.jpg' if case['kind']=='real' and role=='rgb' else '*.png'))
    return np.stack([np.asarray(Image.open(files[i]).convert('RGB' if role in ('rgb', 'target') else 'L'))
                     for i in case['frame_indices']])

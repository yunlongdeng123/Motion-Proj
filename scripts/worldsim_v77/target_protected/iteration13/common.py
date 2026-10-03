"""r45固定的小条件分支实验；不改旧工厂与旧产物。"""
from pathlib import Path
import json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
REPO = Path('/root/autodl-tmp/motion_proj_v77')
S = REPO/'scripts/worldsim_v77/target_protected'
T = Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')
A = Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-DELETE-AUDIT-20260928/r1')
O = T/'r45'
sys.path[:0] = [str(S), str(S/'iteration7'), str(S/'iteration12'), str(S.parent)]

def read(p):
    return json.loads(Path(p).read_text())

def dump(p, data):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix('.tmp'); q.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n'); q.replace(p)

def images(c, role):
    from PIL import Image
    import numpy as np
    if c['kind'] == 'synthetic':
        folder = Path(c['folder'])/({'rgb':'X','hole':'model_hole','target':'Y'}[role])
        paths = sorted(folder.glob('*.png'))
    else:
        folder = Path(c['folder'])/({'rgb':'rgb','hole':'model_mask','alpha':'alpha'}[role])
        paths = sorted(folder.glob('*.jpg' if role == 'rgb' else '*.png'))
    paths = [paths[i] for i in c['frame_indices']]
    return np.stack([np.asarray(Image.open(p).convert('RGB' if role in ('rgb','target') else 'L')) for p in paths])

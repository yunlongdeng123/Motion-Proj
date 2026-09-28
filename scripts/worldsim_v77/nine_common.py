from pathlib import Path
import sys
ROOT=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-NINE-FULL-20260928/r1')
REPO=Path('/root/autodl-tmp/motion_proj_v77')
sys.path.insert(0,str(REPO/'scripts/worldsim_v77'))
from repair_common import read,dump,camera,hull_mask,largest

def progress(stage,**kw):
    import datetime
    data=dict(stage=stage,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw)
    dump(ROOT/f'{stage}_progress.json',data)
    print(stage.upper(),kw,flush=True)

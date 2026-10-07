"""r50：先扩真实DELETE开发排查，暂不造数据或改时间层。"""
from pathlib import Path
import json, os, sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
REPO = Path('/root/autodl-tmp/motion_proj_v77')
T = Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')
O = T/'r50'
META = Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-DELETE-AUDIT-20260928/r1/metadata/v1.0-trainval')
TASK = 'WS-V77-TARGET-PROTECTED-20260929'
sys.path[:0] = [str(REPO/'scripts/worldsim_v77/delete_audit'), str(REPO/'scripts/worldsim_v77')]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    tmp.replace(path)


def ffmpeg_binary():
    import shutil
    executable=shutil.which('ffmpeg')
    if executable:return executable
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()

from pathlib import Path
import sys
from PIL import Image
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from repair_common import read,dump
from video_review import scene_frame
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPAND-EDIT-20260928');R=T/'r5';frames=read(T/'r2/frames.json');spec=frames[0];inst=read(Path(spec['root'])/'instances/instances_info.json');source=scene_frame(spec,46,inst);dump(R/'camera_frames.json',frames+[source])
for fr,c in [(frames[0],0),(source,2)]:
    Image.open(fr['views'][c]['image']).convert('RGB').resize((688,384),Image.Resampling.BICUBIC).save(R/f'source_f{fr["frame"]:03}_cam{c}.png')

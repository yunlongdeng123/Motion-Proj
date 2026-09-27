"""保存保留SUV的已观察原位几何控制输入，不复用隐藏输出当监督。"""
import sys
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from repair_common import read,dump
from video_review import scene_frame
from PIL import Image
BASE=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927');ROOT=BASE/'r41/observed_controls';assert not ROOT.exists();ROOT.mkdir();s=next(s for s in read(BASE/'r2/registration.json')['scenes'] if s['name']=='scene_0255');data=Path(s['data']);inst=read(data/'instances/instances_info.json');rows=[]
for f in [130,135,145]:
 fr=scene_frame(s['spec'],f,inst);b=next(b for b in fr['all_boxes'] if b['actor_id']=='52');v=fr['views'][3];rows.append(dict(frame=f,camera=3,actor=b,view=v,role='generation source ancestor' if f==130 else 'observed heldout image, but development scene not independent test'))
 Image.open(data/'images'/f'{f:03}_3.jpg').save(ROOT/f'original_f{f:03}.png')
dump(ROOT/'camera_frames.json',dict(rows=rows,human_verdict=None,scope='GT pose and scale only; no object movement, no background edit. Source130 is not heldout. Later views may have occluders.'))
print('POSE_CONTROLS_PREPARED')

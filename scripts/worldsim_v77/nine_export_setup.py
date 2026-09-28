"""导出本地渲染所需相机和固定参数，不选择生成结果。"""
from nine_common import *
import tarfile
reg=read(ROOT/'registration.json');render=dict(task_id=reg['task_id'],run_id=reg['run_id'],operation='FACTUAL_UNIFIED_NINE',scenes=[])
for s in reg['scenes']:
    streams=[dict(camera=v['camera'],active_frames=[f for f,a in enumerate(v['projected_areas']) if a>0]) for v in s['streams'] if v['active']]
    render['scenes'].append(dict(name=s['name'],actor=s['actor'],yaw=0,streams=streams))
dump(ROOT/'render_registration.json',render)
display=[]
for s in reg['scenes']:
    v=max(s['streams'],key=lambda v:(sum(a>0 for a in v['projected_areas']),sum(v['projected_areas'])))
    display.append(dict(scene=s['name'],camera=v['camera'],rule='max GT projected visible frame count, then sum area; before model results',human_verdict=None))
dump(ROOT/'display_registration.json',display)
with tarfile.open(ROOT/'render_setup.tar','w') as tar:
    tar.add(ROOT/'render_registration.json',arcname='registration.json')
    tar.add(ROOT/'display_registration.json',arcname='display_registration.json')
    tar.add(ROOT/'registration.json',arcname='experiment_registration.json')
    for s in reg['scenes']:
        tar.add(ROOT/s['name']/'camera_frames.json',arcname=s['name']+'/camera_frames.json')
print('RENDER_SETUP',[(s['name'],sum(len(v['active_frames']) for v in s['streams'])) for s in render['scenes']],flush=True)

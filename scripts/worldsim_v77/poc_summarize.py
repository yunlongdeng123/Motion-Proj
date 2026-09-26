"""Summarize immutable evidence counts for the v77 explicit-asset POC."""
import json
import pathlib
from PIL import Image
import numpy as np

ROOT=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPLICIT-POC-20260926/r1')
reg=json.loads((ROOT/'registration.json').read_text())
out={'task_id':reg['task_id'],'training_steps':0,'human_verdict':None,'scenes':[]}
for scene in reg['scenes']:
    name=scene['name'];base=ROOT/name
    selection=json.loads((base/'selection.json').read_text())
    shape=json.loads((base/'actor/shape_summary.json').read_text())
    paint=json.loads((base/'actor/paint_summary.json').read_text())
    keyframes=json.loads((base/'actor/keyframes.json').read_text())
    masks={}
    for cam in range(6):
        files=list((base/'masks'/f'cam{cam}').glob('*.png'))
        masks[str(cam)]={'files':len(files),'nonempty':sum(bool(np.asarray(Image.open(f)).any()) for f in files)}
    bg=[json.loads((base/'background'/f"{f['frame']:03d}"/'summary.json').read_text()) for f in scene['frames']]
    out['scenes'].append({'name':name,'actor_id':scene['actor'],'sampled_frames':[f['frame'] for f in scene['frames']],
      'sam2_masks':masks,'selected_keyframes':len(keyframes['selected']),'hunyuan_shape':shape,'hunyuan_paint':paint,
      'omega_bg_forwards':len(bg),'omega_bg_points_range':[min(x['bg_points'] for x in bg),max(x['bg_points'] for x in bg)],
      'omega_bg_peak_gib':max(x['peak_gpu_gib'] for x in bg),'human_verdict':None})
print(json.dumps(out,ensure_ascii=False,indent=2))

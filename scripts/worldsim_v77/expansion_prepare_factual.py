from pathlib import Path
import json,shutil
R=Path(__file__).resolve().parent;frames=json.loads((R/'actor12/camera_frames.json').read_text(encoding='utf-8'));scenes=[]
for yaw in [0,180]:
    name=f'factual12_yaw{yaw}';dest=R/name;dest.mkdir(exist_ok=False);shutil.copy2(R/'actor12/actor.glb',dest/'actor.glb');(dest/'camera_frames.json').write_text(json.dumps(frames,indent=2),encoding='utf-8');scenes.append(dict(name=name,actor='12',yaw=yaw,streams=[dict(camera=0,active_frames=[0]),dict(camera=2,active_frames=[46])]))
(R/'factual_registration.json').write_text(json.dumps(dict(task_id='WS-V77-EXPAND-EDIT-20260928',run_id='r5',operation='FACTUAL_ORIENTATION_CONTROL',scenes=scenes,human_verdict=None),indent=2),encoding='utf-8')

"""将已审核的空道路位置登记为独立INSERT，不冒充目标12的factual。"""
from pathlib import Path
import json,shutil,datetime
R=Path(__file__).resolve().parent
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
reg=load(R/'proposal_registration.json');chosen=next(p for p in reg['candidates'] if p['proposal']==2)
scene=R/'insert_official_000';scene.mkdir(exist_ok=False)
shutil.copy2(R/'actor_0255.glb',scene/'actor.glb')
frames=load(R/'insert_frames.json')
for fr in frames:
    fr['all_boxes'].append(dict(actor_id='inserted_0255',pose=chosen['pose_world'],size_lwh=chosen['size_lwh']))
(scene/'camera_frames.json').write_text(json.dumps(frames,indent=2),encoding='utf-8')
doc=dict(task_id=reg['task_id'],run_id='r3',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),operation='INSERT',selected=chosen,
    assistant_review='Proposal2 ground footprint stays in empty rear road lane f0/f9, unlike proposal9 across curb. GT/ego/static-point screening covers30frames; render only10. Adjacent CAM4 catches only image edge. No traffic-response or full road-law certificate.',
    scenes=[dict(name='insert_official_000',actor='inserted_0255',yaw=0,streams=[dict(camera=c,active_frames=list(range(10))) for c in [5,4]])],
    asset_source=reg['asset'],human_verdict=None)
(R/'registration.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8')

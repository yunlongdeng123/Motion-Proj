"""仅用首帧实测路面高度作固定z平移诊断，保留r5未校正对照。"""
from pathlib import Path
import json,shutil,datetime
R=Path(__file__).resolve().parent;g=json.loads((R/'ground_audit.json').read_text(encoding='utf-8'));shift=-g['rows'][0]['gt_bottom_minus_plane'];name='factual12_grounded';dest=R/name;dest.mkdir(exist_ok=False);shutil.copy2(R/'actor12/actor.glb',dest/'actor.glb');frames=json.loads((R/'actor12/camera_frames.json').read_text(encoding='utf-8'))
for fr in frames:
    actor=next(b for b in fr['all_boxes'] if b['actor_id']=='12');actor['pose'][2][3]+=shift
(dest/'camera_frames.json').write_text(json.dumps(frames,indent=2),encoding='utf-8')
reg=dict(task_id='WS-V77-EXPAND-EDIT-20260928',run_id='r7',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),operation='FACTUAL_GROUND_CONTROL',delta_z_m=shift,
    intervention='Only fixed z translation from f0 measured local road plane. Same r5 GLB, GT XY/yaw/size, r2 B_t, lights, camera and shader. No new generation.',
    evidence=g,limits='First-frame plane source nearest2.34m from target, later/source46 up to6.84m. Planar-road diagnostic, not direct observed target tire contact or global correct road surface.',
    scenes=[dict(name=name,actor='12',yaw=0,streams=[dict(camera=0,active_frames=list(range(10))),dict(camera=2,active_frames=[46])])],human_verdict=None,failure_ledger_refs=['V77-F02'])
(R/'ground_registration.json').write_text(json.dumps(reg,ensure_ascii=False,indent=2),encoding='utf-8')

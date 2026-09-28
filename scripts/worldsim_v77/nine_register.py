"""九例共同配置重跑登记；均为已接触开发例，旧结果只作参考。"""
from pathlib import Path
import sys,datetime
import numpy as np
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from repair_common import read,dump,camera
from video_review import scene_frame
from geometry import project_bbox

BASE=Path('/root/autodl-tmp/runs/worldsim_v77')
ROOT=BASE/'WS-V77-NINE-FULL-20260928/r1'
assert not ROOT.exists(), '唯一run已存在；先检查状态，禁止覆盖'
ROOT.mkdir(parents=True)
p0=read(BASE/'WS-V77-P0-24ACTOR-20260926/r1/registration.json')
exp=read(BASE/'WS-V77-EXPAND-EDIT-20260928/r1/registration.json')
items=[]
for name,actor,cam in [('scene_0230','22',5),('scene_0255','25',3),('official_000','12',0)]:
    spec=next(s for s in p0['scenes'] if s['name']==name)
    items.append(dict(name=name,actor=actor,primary_camera=cam,spec=spec,root=spec['root']))
for s in exp['new_scenes']:
    items.append({k:s[k] for k in ['name','actor','primary_camera','spec','root']})
for s in items:
    inst=read(Path(s['root'])/'instances/instances_info.json')
    frames=[scene_frame(s['spec'],f,inst) for f in range(30)]
    assert all(any(b['actor_id']==s['actor'] for b in f['all_boxes']) for f in frames)
    s['streams']=[]
    for c in range(6):
        boxes=[];areas=[]
        for fr in frames:
            b=next(b for b in fr['all_boxes'] if b['actor_id']==s['actor'])
            v,k=camera(fr,c);bb=project_bbox(b['pose'],b['size_lwh'],v,k,(576,1024))
            boxes.append(bb);areas.append(float((bb[2]-bb[0])*(bb[3]-bb[1])) if bb else 0)
        # 完整六相机输入：所有目标有投影的视角均尝试同一SAM规则，不只处理主视角。
        active=max(areas)>0
        s['streams'].append(dict(camera=c,active=active,boxes=boxes,projected_areas=areas,prompt_frame=int(np.argmax(areas)) if active else None))
    s.update(source_frames=list(range(30)),count=30,role='exposed_development_rerun',yaw=0)
    dump(ROOT/s['name']/'camera_frames.json',frames)
    print('SCENE',s['name'],s['actor'],[(x['camera'],round(max(x['projected_areas'])),sum(a>0 for a in x['projected_areas'])) for x in s['streams'] if x['active']],flush=True)
reg=dict(task_id='WS-V77-NINE-FULL-20260928',run_id='r1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scenes=items,
    request='Same latest frozen rules rerun all9 full background + explicit actor DELETE; three columns original/factual reconstruction/DELETE reconstruction.',
    fixed=dict(frames=list(range(30)),fps=10,model_resolution=[1024,576],render_resolution=[688,384],sam='SAM2.1-large, maxGTarea box prompt, bidirectional propagation',core='largest(SAM & GT hull dilated3px)',write='dilate3px, bounded by GT hull+3px',model_mask='write bbox pad8 left/right/top and24 bottom',alpha='rectangle smoothstep8px; write1; otherGT hull outsidewrite0',deletion_seed=42,deletion_steps=25,window_starts=[0,9,18,27],window_length=10,same_time_overlap=True,
        omega_checkpoint=p0['checkpoint'],omega_source=p0['omega_source'],omega_frozen=True,omega_resolution=512,
        asset='existing official Hunyuan3D-2.1 single best observed crop within same30frames; one shape andPBR, no result-dependent search',shape_seed=7740,shape_steps=50,guidance=7.5,octree=256,pbr_views=6,pbr_resolution=512,
        pose='GT center,size,yaw; native Blender longY->forwardX fixed -90deg; fixed yaw0 all9; no per-scene height/yaw fit',lighting='fixed world+sun; no contactshadow fitting',background='independent B_t; GT calibrated cameras/background LiDAR global scale; other actors baked',empty_masks='no invented mask or box fallback; unchanged RGB where no mask, logged as unresolved input failure; full downstream output is diagnostic not admitted'),
    input_roles='RGB GT camera/boxes and LiDAR are BUILD inputs; original GT actor trajectory drives factual placement. All9 exposed, no training/generalization claim. Source asset reference is within same30-frame BUILD window.',
    acceptance='Run all9 without replacing weak outputs. Flag missing masks/regenerated vehicles/identity or placement failures. Original/reconstruction/DELETE share times and primary view; provide six-view comparisons and pure geometry controls.',
    stage_semantics='Background completion then frozen Omega; factual=B+actor.glb, DELETE=same B without actor. Middle is not original RGB or direct rawOmega baseline. Query invokes no neural inference.',
    preserved='All r18/r21/r30 and prior nine outputs unchanged. No reuse generated videos/GLB as new unified rerun.',
    resource=dict(gpu='singleRTX3090',cpu_threads=4,model_sequence='SAM2 -> DriveEditor -> frozenOmega -> Hunyuan; localBlender CPU may run alongside GPU',max_drive_window_seconds=240),
    failure_ledger_refs=['V77-F02'],human_verdict=None)
dump(ROOT/'registration.json',reg)
print('REGISTERED',ROOT,flush=True)

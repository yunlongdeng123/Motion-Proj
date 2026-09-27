"""锁定六个新增scene和可迁移r30规则；留出不按生成结果调整。"""
from pathlib import Path
import sys,datetime,json
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from repair_common import read,dump,camera
from video_review import scene_frame
from geometry import project_bbox
ROOT=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPAND-EDIT-20260928/r1');inv=read(ROOT/'inventory.json')
choices=[('350','7',5,'development','清楚近车/道路与路缘'),('663','6',5,'development','湿路/施工护栏'),('191','12',2,'development','栏杆/树木遮挡'),('425','1',5,'heldout_v77','空旷道路/远车'),('382','4',0,'heldout_v77','邻车遮挡/行驶中尺度变化'),('756','3',2,'heldout_v77','夜间/反光/双视角')]
scenes=[]
for sid,aid,cam,split,reason in choices:
    row=next(c for c in inv['candidates'] if c['scene']==sid and c['actor']==aid and c['camera']==cam)
    name=f'processed_{sid}';dest=ROOT/name;assert not dest.exists();dest.mkdir();spec=dict(row['spec'],name=name);inst=read(Path(row['root'])/'instances/instances_info.json');frames=[scene_frame(spec,f,inst) for f in range(30)]
    streams=[]
    for c in range(6):
        boxes=[];areas=[]
        for fr in frames:
            b=next(b for b in fr['all_boxes'] if b['actor_id']==aid);v,k=camera(fr,c);bb=project_bbox(b['pose'],b['size_lwh'],v,k,(576,1024));boxes.append(bb);areas.append((bb[2]-bb[0])*(bb[3]-bb[1]) if bb else 0)
        if c!=cam and max(areas)<512:continue
        stream=dest/f'cam{c}';(stream/'rgb').mkdir(parents=True)
        for f in range(30):
            im=Image.open(Path(row['root'])/'images'/f'{f:03}_{c}.jpg').convert('RGB').resize((1024,576),Image.Resampling.BILINEAR);im.save(stream/'rgb'/f'{f:05}.jpg',quality=97)
        streams.append(dict(camera=c,boxes=boxes,projected_areas=areas,prompt_frame=int(np.argmax(areas)),role='primary' if c==cam else 'cross_view'))
    item=dict(name=name,processed_id=sid,root=row['root'],actor=aid,track_id=row['track_id'],primary_camera=cam,split=split,reason=reason,frames=list(range(30)),streams=streams,spec=spec)
    scenes.append(item);dump(dest/'frames.json',frames)
assert len(scenes)==6
# 去掉历史别名，新增scene不能靠重复同一段达到九例。
ids={r['root']:set(r.get('track_ids',[])) for r in inv['rows']}
for a in scenes:
    for b in scenes:
        if a['name']!=b['name']:assert not(ids[a['root']]&ids[b['root']])
dump(ROOT/'registration.json',dict(task_id='WS-V77-EXPAND-EDIT-20260928',run_id='r1',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),existing_scenes=['scene_0230','scene_0255','official_000'],new_scenes=scenes,total_unique_scenes=9,alias_excluded={'179':'scene_0230','204':'scene_0255'},
    frozen_recipe=dict(model='existing DriveEditor deletion',seed=42,steps=25,model_window=10,output_frames=30,fps=10,previous_segment_condition=True,window_starts=[0,9,18,27],sam='existing SAM2.1 large; GT box prompt at maximum projected area; bidirectional video propagation',write_core='largest SAM inside target GT hull +3px, dilate3px bounded by hull',model_mask='write mask bbox padded8 left/right/top and24 bottom',writeback='whole model rectangle smoothstep8px; precise write weight1; other GT envelopes outside write weight0',cpu_threads=4),
    exposure='Three old scenes are development. New six selected only from original images and source geometry, before new model outputs. Three heldout_v77 never tuned per scene; old V5/dynamic-editing source exposure and pretrained training overlap unknown, not an independent external benchmark.',
    gates='Inspect exact identity and masks before generation; failed stream stays failed in denominator. Same recipe across added scenes, no per-scene threshold/seed edits. No unsupported evidence warp.',
    downstream='r30 return to frozen Omega is separate short-window validation; INSERT independent; MOVE only after background and factual visual review. No traffic-interaction claim.',
    training='Prepare 200-500 candidate clean RGB clips and masks, no training until repeated module failure excluding engineering errors and fixed eval. Omega and Hunyuan remain frozen.',human_verdict=None,failure_ledger_refs=['V77-F02'],failure_ledger_delta='none'))
print('REGISTERED',[(s['name'],s['split'],[(v['camera'],v['prompt_frame']) for v in s['streams']]) for s in scenes],flush=True)

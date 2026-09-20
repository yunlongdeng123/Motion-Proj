"""构造初始图像一致的未来减速轨迹，并验证 reference 条件输入。"""
from datetime import datetime,timezone
import copy,json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation,Slerp
import torch

from closed_loop_bridge import upload_scene
from prepare_argoverse import project_track
from render_argoverse import render


ROOT=Path('/root/autodl-tmp/runs/worldsim_v75')
SOURCE=ROOT/'WS-V75-ACTOR-REMOVAL-G1-QUALIFY-01/20260921-r1'
OUT=ROOT/'WS-V75-ACTOR-SLOWDOWN-G1-QUALIFY-01/20260921-r1'
SAMPLES=[0,4,5,45,61,85,109,116];LATE=[45,61,85,109,116]


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    assert not OUT.exists();OUT.mkdir(parents=True)
    old=json.loads((SOURCE/'result.json').read_text());s=old['selected'];base=Path(s['base']);reference=json.loads(Path(s['state_inputs']['reference']['source']).read_text())
    edited=copy.deepcopy(reference);target=next(t for t in edited['tracks'] if t['id']==s['actor'] and 0 in t['frames'])
    source_target=next(t for t in reference['tracks'] if (t['id'],t['segment'])==(target['id'],target['segment']))
    frames=np.asarray(target['frames'],float);tau=np.where(frames>=5,4+.5*(frames-4),frames);centers=np.asarray(target['centers'])
    target['centers']=np.stack([np.interp(tau,frames,centers[:,i]) for i in range(3)],1).tolist()
    target['quaternions']=Slerp(frames,Rotation.from_quat(target['quaternions']))(tau).as_quat().tolist()
    source_path=OUT/'condition-reference.json';edited_path=OUT/'condition-reference-slowed-after-005.json';save(source_path,reference);save(edited_path,edited)
    tr=np.load(base/'trajectory.npz');separations=[]
    for frame in [5,13,21,29,37,45,61,85,109,116]:
        a=project_track(source_target,frame,tr['camera_world'],tr['K']);b=project_track(target,frame,tr['camera_world'],tr['K'])
        ac=(np.asarray(a['bounds'][:2])+a['bounds'][2:])/2;bc=(np.asarray(b['bounds'][:2])+b['bounds'][2:])/2
        separations.append({'frame':frame,'center_separation_px':float(np.linalg.norm(ac-bc)),'original_bounds':a['bounds'],'edited_bounds':b['bounds']})
    assert all(next(x for x in separations if x['frame']==f)['center_separation_px']>=20 for f in LATE)
    protocol={'task_id':'WS-V75-ACTOR-SLOWDOWN-G1-QUALIFY-01','run_id':OUT.name,'frozen_utc':datetime.now(timezone.utc).isoformat(),
      'source_revision':'7a5f22d2','source_qualification':str(SOURCE),'log':s['log'],'actor':s['actor'],
      'intervention':'frames0-4 unchanged; from frame5, actor pose follows original trajectory at tau=4+0.5*(frame-4), including SO(3) interpolation; declared future slowdown',
      'initial_consistency':'same real initial RGB/text and exact reference state/condition through frame4',
      'input_separability':separations,'late_evaluation_frames':LATE,
      'raster_gate':'conditions exact at frames0,4; changed at all late frames',
      'followup':'one edited generation reusing existing seed42 unedited reference; no reconstruction until G1 passes',
      'stop':'one fixed factor0.5 and one sequence; no factor/event/source/seed search; OOM/error stops','human_verdict':None,'failure_ledger_delta':'none'}
    save(OUT/'protocol.json',protocol);states={'reference':{'source':str(source_path),'edited':str(edited_path),'target_dimensions_m':target['dimensions'],
      'initial_center_world_m':target['centers'][target['frames'].index(0)],'other_tracks_exact_reference':True,'map_exact_reference':True}}
    result={'status':'passed_pending_raster','selected':{'log':s['log'],'actor':s['actor'],'behind':s['behind'],'event_frame':5,'state_inputs':states,
      'conditioning':s['conditioning'],'base':str(base)},'generation_admitted':False,'raster_probe_required':True,'new_reconstruction_calls':0,'world_model_generation_calls':0,'human_verdict':None,'failure_ledger_delta':'none'};save(OUT/'result.json',result)
    began=time.monotonic();terminal={'status':'started','human_verdict':None,'failure_ledger_delta':'none'}
    try:
        rendered={}
        for variant,scene in [('unedited',reference),('edited',edited)]:
            ctx,sid,fit=upload_scene(scene,tr['timestamps_us'],tr['K']);rendered[variant]=render(ctx,sid,tr['timestamps_us'][SAMPLES],tr['camera_world'][SAMPLES]);del ctx;torch.cuda.empty_cache()
        rows=[]
        for i,frame in enumerate(SAMPLES):
            mask=np.any(rendered['unedited'][i]!=rendered['edited'][i],axis=-1);rows.append({'frame':frame,'changed_pixels':int(mask.sum()),'exactly_equal':bool(not mask.any())})
        passed=all(next(x for x in rows if x['frame']==f)['exactly_equal'] for f in [0,4]) and all(next(x for x in rows if x['frame']==f)['changed_pixels']>=50 for f in LATE)
        terminal.update(status='passed' if passed else 'failed',rows=rows,generation_admitted=passed,condition_render_calls=2,rendered_condition_frames=len(SAMPLES)*2)
    except BaseException as exc:
        terminal.update(status='oom_stopped' if isinstance(exc,torch.OutOfMemoryError) else 'failed_stopped',error_type=type(exc).__name__,error=str(exc),generation_admitted=False);raise
    finally:
        terminal.update(wall_s=time.monotonic()-began,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,world_model_generation_calls=0);save(OUT/'raster_result.json',terminal)
    result.update(status='qualified' if terminal['generation_admitted'] else terminal['status'],generation_admitted=terminal['generation_admitted'],raster_result=str(OUT/'raster_result.json'));save(OUT/'result.json',result)
    print(json.dumps({'result':result,'raster':terminal,'separations':separations},ensure_ascii=False),flush=True)


if __name__=='__main__':main()

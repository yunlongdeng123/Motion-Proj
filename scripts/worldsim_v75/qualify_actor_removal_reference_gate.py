"""为中远距离新源构造 reference 未编辑/移除条件，并验证 raster。"""
from datetime import datetime,timezone
import copy,json
from pathlib import Path
import time

import numpy as np
import torch

from closed_loop_bridge import upload_scene
from render_argoverse import render


ROOT=Path('/root/autodl-tmp/runs/worldsim_v75')
SCREEN=ROOT/'WS-V75-ACTOR-REMOVAL-G1-SCREEN-01/20260921-r1'
OUT=ROOT/'WS-V75-ACTOR-REMOVAL-G1-QUALIFY-01/20260921-r1'
EVENT=5;FRAMES=[0,5,30,90]


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    assert not OUT.exists();OUT.mkdir(parents=True)
    screen=json.loads((SCREEN/'result.json').read_text());selected=screen['selected']
    assert screen['status']=='qualified' and screen['screened_logs']==5
    protocol={'task_id':'WS-V75-ACTOR-REMOVAL-G1-QUALIFY-01','run_id':OUT.name,'frozen_utc':datetime.now(timezone.utc).isoformat(),
      'source_revision':'61bff442','source_screen':str(SCREEN/'result.json'),'log':selected['log'],'actor':selected['actor'],'behind':selected['behind'],
      'question':'Does the reference state obey an actor removal for a lower-salience initial actor?',
      'intervention':'A present through frame4, absent from state at frame5 onward; declared instantaneous external removal',
      'raster_gate':'frame0 unedited/removed exactly equal; frames5,30,90 each change >=50 pixels',
      'followup':'encode once and run reference unedited/removed only; DVGT forbidden until reference generation gate passes',
      'stop':'raster failure or any error/OOM stops; no source/event/threshold/seed replacement','human_verdict':None,'failure_ledger_delta':'none'}
    save(OUT/'protocol.json',protocol)
    base=Path(selected['base']);reference=json.loads((base/'scene.json').read_text())
    target=next(t for t in reference['tracks'] if t['id']==selected['actor'] and 0 in t['frames'])
    edited=copy.deepcopy(reference);e=next(t for t in edited['tracks'] if (t['id'],t['segment'])==(target['id'],target['segment']))
    keep=[i for i,f in enumerate(e['frames']) if f<EVENT];assert keep and e['frames'][keep[-1]]==EVENT-1
    for key in ['frames','centers','quaternions']:e[key]=[e[key][i] for i in keep]
    source_path=OUT/'condition-reference.json';edited_path=OUT/'condition-reference-removed-after-005.json'
    save(source_path,reference);save(edited_path,edited)
    states={'reference':{'source':str(source_path),'edited':str(edited_path),'target_dimensions_m':target['dimensions'],
                         'initial_center_world_m':target['centers'][target['frames'].index(0)],'retained_target_frames':e['frames'],
                         'other_tracks_exact_reference':True,'map_exact_reference':True}}
    result={'status':'passed_pending_raster','selected':{'log':selected['log'],'actor':selected['actor'],'behind':selected['behind'],
            'event_frame':EVENT,'state_inputs':states,'conditioning':str(OUT/'conditioning'),'base':str(base)},
            'generation_admitted':False,'raster_probe_required':True,'new_reconstruction_calls':0,
            'world_model_generation_calls':0,'human_verdict':None,'failure_ledger_delta':'none'}
    save(OUT/'result.json',result);began=time.monotonic();terminal={'status':'started','human_verdict':None,'failure_ledger_delta':'none'}
    try:
        tr=np.load(base/'trajectory.npz');images={}
        for variant,scene in [('unedited',reference),('removed',edited)]:
            ctx,sid,fit=upload_scene(scene,tr['timestamps_us'],tr['K'])
            images[variant]=render(ctx,sid,tr['timestamps_us'][FRAMES],tr['camera_world'][FRAMES]);del ctx;torch.cuda.empty_cache()
        rows=[]
        for i,frame in enumerate(FRAMES):
            mask=np.any(images['unedited'][i]!=images['removed'][i],axis=-1)
            rows.append({'frame':frame,'changed_pixels':int(mask.sum()),'exactly_equal':bool(not mask.any())})
        passed=rows[0]['exactly_equal'] and all(x['changed_pixels']>=50 for x in rows[1:])
        terminal.update(status='passed' if passed else 'failed',rows=rows,generation_admitted=passed,condition_render_calls=2,rendered_condition_frames=8)
    except BaseException as exc:
        terminal.update(status='oom_stopped' if isinstance(exc,torch.OutOfMemoryError) else 'failed_stopped',error_type=type(exc).__name__,error=str(exc),generation_admitted=False);raise
    finally:
        terminal.update(wall_s=time.monotonic()-began,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,world_model_generation_calls=0)
        save(OUT/'raster_result.json',terminal)
    result.update(status='qualified' if terminal['generation_admitted'] else terminal['status'],generation_admitted=terminal['generation_admitted'],raster_result=str(OUT/'raster_result.json'))
    save(OUT/'result.json',result);print(json.dumps({'result':result,'raster':terminal},ensure_ascii=False),flush=True)


if __name__=='__main__':main()

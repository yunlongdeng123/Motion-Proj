"""已有六日志中的真实运动跟车窗口；先冻结任务条件，再读取模型误差。"""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
import torch
from prepare_argoverse import ROOT, CAMERA, POS, QUAT, poses, project_track, crop_image
from following_geometry import Route, scene, reference_lead, FRAMES
from evaluate_following_baseline import logged_state, oracle_lead
from rgb_idm_policy import RGBIDMPolicy, IDMPolicy
from audit_temporal_state import reference_rows, velocity

RUNS=Path('/root/autodl-tmp/runs/worldsim_v75')
PREVIOUS=RUNS/'WS-V75-BRAKING-DEV2-01/20260920-r1'
OUT=RUNS/'WS-V75-MOVING-FOLLOWING-SCREEN-01/20260920-r1'


def save(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not OUT.exists(); OUT.mkdir(parents=True)
    prior=json.loads((PREVIOUS/'protocol.json').read_text())
    protocol={'task_id':'WS-V75-MOVING-FOLLOWING-SCREEN-01','run_id':OUT.name,
              'frozen_utc':datetime.now(timezone.utc).isoformat(),
              'logs':prior['logs'],'start_offsets_seconds':prior['start_offsets_seconds'],
              'source_role':'reuse all18 windows in the existing frozen6-log development window; no new logs or offsets, not independent confirmation',
              'motivation':'previous qualified braking targets are nearly static; motion is a different state dimension, not a revision of previous screen terminal counts',
              'task':'following a genuinely moving front actor that changes ordinary driving actions; recorded ego deceleration is not required',
              'task_gate':{'initial_ego_speed_min_mps':2.,'first_route_leader_times_min':3,
                           'past_target_speed_min_mps':1.,'past_forward_velocity_min_mps':1.,
                           'target_displacement_over2s_min_m':2.,'target_full117frame_reference':True,
                           'initial_bbox_min_wh':[48,32],'past_3time_bbox_min_wh':[32,24],
                           'all_sampled_projections_fully_inside':True,
                           'leader_effect_vs_free_acceleration_max_mps2':-.5,'leader_effect_times_min':2},
              'past_reference':'past1s >=8 annotations, maximum adjacent gap150ms; evaluate past CV future error but never fit with future positions',
              'selection':'first qualifying start per log, first2 logs in frozen order; no model-error ranking or replacement after downstream failure',
              'next_scope':'real RGB fixed-policy and legal-past correspondence qualification first; no generation admission from this screen',
              'stop':'exact6x3 windows; no new logs, thresholds, offsets or synthetic velocities; zero candidates closes this window',
              'human_verdict':None,'failure_ledger_refs':['V74-H2-F22'],'failure_ledger_delta':'none'}
    save(OUT/'protocol.json',protocol)
    policy=object.__new__(RGBIDMPolicy)
    policy.idm=IDMPolicy(target_velocity=10.,min_gap_to_lead_agent=1.,headway_time=1.5,accel_max=1.,decel_max=3.)
    rows=[];qualified=[];began=time.monotonic()
    result={'status':'running','human_verdict':None,'failure_ledger_delta':'none'}
    try:
        for log in protocol['logs']:
            raw=ROOT/log; images=sorted((raw/'sensors/cameras'/CAMERA).glob('*.jpg'))
            image_ns=np.array([int(p.stem) for p in images],np.int64)
            ego=pd.read_feather(raw/'city_SE3_egovehicle.feather')
            log_cases=[]
            for offset in protocol['start_offsets_seconds']:
                index=int(np.searchsorted(image_ns,image_ns[0]+round(offset*1e9)))
                start=int(image_ns[index]); row={'log_id':log,'offset_seconds':offset,'cutoff_ns':start,'qualified':False}
                rows.append(row)
                if index<20 or start-1_000_000_000<ego.timestamp_ns.min() or start+7_866_666_667>min(image_ns[-1],ego.timestamp_ns.max()):
                    row['reason']='insufficient_history_or_future_reference'; continue
                matrices=poses(ego,[start-500_000_000,start]); v=(matrices[1,:3,3]-matrices[0,:3,3])/.5
                row['initial_ego_speed_mps']=float(v@matrices[1,:3,0])
                if row['initial_ego_speed_mps']<2:
                    row['reason']='ego_below_fixed_task_speed';continue
                old_base=PREVIOUS/'cases'/log/f'start-{offset:.1f}'/'base'
                folder=OUT/'cases'/log/f'start-{offset:.1f}';folder.mkdir(parents=True)
                base=old_base if (old_base/'input_manifest.json').exists() else folder/'base'
                if base!=old_base:
                    cmd=[sys.executable,str(Path(__file__).with_name('prepare_argoverse.py')),'--output',str(base),'--log-id',log,
                         '--start-offset-seconds',str(offset),'--task-id',protocol['task_id'],'--run-id',OUT.name]
                    with (folder/'prepare.log').open('w') as output:subprocess.run(cmd,stdout=output,stderr=subprocess.STDOUT,check=True)
                row.update(base=str(base),reused_prepared_base=base==old_base)
                data=scene(base);tr=np.load(base/'trajectory.npz');route=Route(tr['ego_world'][:,:3,3])
                manifest=json.loads((base/'input_manifest.json').read_text());assert manifest['first_timestamp_ns']==start
                leads=[reference_lead(data,tr,route,f) for f in FRAMES]
                first=next((r for r in leads if r),None)
                row['leaders']=leads
                if first is None:
                    row['reason']='no_route_leader';continue
                target=next(t for t in data['tracks'] if (t['id'],t['segment'])==(first['id'],first['segment']))
                count=sum(bool(x and (x['id'],x['segment'])==(target['id'],target['segment'])) for x in leads)
                row.update(target=target['id'],target_segment=target['segment'],target_leader_times=count)
                if count<3 or not all(f in target['frames'] for f in range(117)):
                    row['reason']='target_not_persistent_full_task';continue
                packed,_=reference_rows(raw,target['id'],np.array(manifest['city_origin']))
                past=packed[(packed.timestamp_ns<=start)&(packed.timestamp_ns>=start-1_000_000_000)]
                if len(past)<8 or np.diff(past.timestamp_ns).max()>150_000_000:
                    row['reason']='insufficient_past_target_reference';continue
                times=(past.timestamp_ns.to_numpy(np.int64)-start)/1e9
                v=velocity(times,past[POS].to_numpy())
                future=np.array([target['centers'][target['frames'].index(f)] for f in range(117)])
                future_t=(tr['timestamps_ns'][:117]-start)/1e9
                displacement=float(np.linalg.norm(future[60,:2]-future[0,:2]))
                forward=float(v@tr['ego_world'][0,:3,0])
                row.update(past_reference_count=len(past),past_velocity_world_mps=v.tolist(),past_speed_xy_mps=float(np.linalg.norm(v[:2])),
                           past_forward_velocity_mps=forward,target_displacement_over2s_m=displacement,
                           reference_past_cv_max_future_error_m=float(np.linalg.norm(future[0]+future_t[:,None]*v-future,axis=1).max()))
                if row['past_speed_xy_mps']<1 or forward<1 or displacement<2:
                    row['reason']='no_forward_moving_leader';continue
                projections=[project_track(target,f,tr['camera_world'],tr['K']) for f in FRAMES]
                box=projections[0]['bounds'] if projections[0] else [0,0,0,0]
                initial_ok=bool(projections[0] and box[2]-box[0]>=48 and box[3]-box[1]>=32)
                future_visible=all(p and p['fully_inside'] for p in projections)
                captures=[]
                for relative in [-1.,-.5,0.]:
                    request=start+round(relative*1e9)
                    ix=int(np.searchsorted(image_ns,request,side='right')-1)
                    captures.append(int(image_ns[ix]))
                current=poses(ego,captures);current[:,:3,3]-=np.array(manifest['city_origin'])
                extrinsic=np.linalg.inv(tr['ego_world'][0])@tr['camera_world'][0]
                target_pose=poses(packed,captures)
                past_track={'frames':[0,1,2],'centers':target_pose[:,:3,3].tolist(),
                            'quaternions':Rotation.from_matrix(target_pose[:,:3,:3]).as_quat().tolist(),'dimensions':target['dimensions']}
                past_projections=[project_track(past_track,f,current@extrinsic,tr['K']) for f in range(3)]
                past_visible=all(p and p['fully_inside'] and p['bounds'][2]-p['bounds'][0]>=32 and p['bounds'][3]-p['bounds'][1]>=24 for p in past_projections)
                actions=[]
                for f,lead in zip(FRAMES,leads):
                    s=logged_state(base,tr,f);o=oracle_lead(lead,data,tr,f,s['yaw_rad'],s['speed_mps'])
                    acc=policy.acceleration(s['speed_mps'],o);free=policy.acceleration(s['speed_mps'],None)
                    actions.append({'frame':f,'oracle_acceleration_mps2':acc,'leader_effect_mps2':acc-free,
                                    'matched_target':bool(lead and (lead['id'],lead['segment'])==(target['id'],target['segment']))})
                effect=sum(a['matched_target'] and a['leader_effect_mps2']<=-.5 for a in actions)
                row.update(initial_projection=projections[0],future_projections=projections,past_captures_ns=captures,past_projections=past_projections,
                           initial_bbox_ok=initial_ok,future_fully_visible=future_visible,past_observable=past_visible,
                           oracle_actions=actions,task_effect_times=effect)
                row['qualified']=bool(initial_ok and future_visible and past_visible and effect>=2)
                row['reason']='moving_following_geometry_qualified' if row['qualified'] else 'moving_leader_not_visible_or_not_action_relevant'
                save(folder/'geometry_result.json',row)
                if row['qualified']:log_cases.append(row)
            if log_cases:qualified.append(log_cases[0])
            save(OUT/'progress.json',{'status':'running','rows':rows,'qualified_logs':[x['log_id'] for x in qualified]})
            print(json.dumps({'log':log,'qualified_starts':len(log_cases)}),flush=True)
        selected=qualified[:2]
        for case in selected:
            base=Path(case['base']);m=json.loads((base/'input_manifest.json').read_text())
            folder=OUT/'selected'/case['log_id'];folder.mkdir(parents=True)
            save(folder/'source.json',{'base':str(base),'source_log':case['log_id'],'target':case['target'],'selection_protocol':str(OUT/'protocol.json')})
            sheet=Image.new('RGB',(1280,3*394),'#152839');d=ImageDraw.Draw(sheet)
            font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
            for i,(ns,projection) in enumerate(zip(case['past_captures_ns'],case['past_projections'])):
                rgb=crop_image(ROOT/case['log_id']/'sensors/cameras'/CAMERA/f'{ns}.jpg',m['crop_xyxy'])
                bounds=projection['bounds'];ImageDraw.Draw(rgb).rectangle(bounds,outline='yellow',width=3)
                d.text((8,i*394+5),f"{case['log_id'][:8]} +{case['offset_seconds']}s | legal RGB t={(ns-case['cutoff_ns'])/1e9:+.2f}s | reference target",font=font,fill='white')
                sheet.paste(rgb.resize((640,352)),(0,i*394+32))
                x,y=(bounds[0]+bounds[2])/2,(bounds[1]+bounds[3])/2
                sheet.paste(rgb.crop((int(x-160),int(y-88),int(x+160),int(y+88))).resize((640,352)),(640,i*394+32))
            sheet.save(folder/'past-task-review.jpg',quality=93)
        assert not torch.cuda.is_initialized()
        result.update(status='complete',window_count=len(rows),qualified_log_count=len(qualified),selected=selected,rows=rows,
                      new_gpu_calls=0,cuda_initialized=False)
    except BaseException as exc:
        result.update(status='failed_stopped',error_type=type(exc).__name__,error=str(exc),rows=rows);raise
    finally:
        result['wall_s']=time.monotonic()-began;save(OUT/'result.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ['rows','selected']}),flush=True)


if __name__=='__main__':main()

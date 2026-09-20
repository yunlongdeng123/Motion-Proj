"""冻结一次6日志18窗口任务发现；不读取重建误差，不回改旧48窗口终态。"""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw
from prepare_argoverse import ROOT,CAMERA,poses,project_track,crop_image
from following_geometry import Route,scene,reference_lead,FRAMES
from evaluate_following_baseline import logged_state,oracle_lead
from rgb_idm_policy import RGBIDMPolicy,IDMPolicy

OUT=Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-BRAKING-DEV2-01/20260920-r1')
PREVIOUS=Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-BRAKING-TASKS-01/20260920-r1')


def save(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def main():
    assert not OUT.exists();OUT.mkdir(parents=True)
    prior=json.loads((PREVIOUS/'protocol.json').read_text())['logs']
    complete=sorted(p for p in ROOT.iterdir() if p.is_dir() and (p/'annotations.feather').exists()
                    and (p/'city_SE3_egovehicle.feather').exists()
                    and len(list((p/'sensors/cameras'/CAMERA).glob('*.jpg')))>=200)
    logs=[p for p in complete if p.name not in prior][:6];assert len(logs)==6
    protocol={'task_id':'WS-V75-BRAKING-DEV2-01','run_id':OUT.name,'frozen_utc':datetime.now(timezone.utc).isoformat(),
              'logs':[p.name for p in logs],'source_selection':'first6 sorted locally complete logs outside prior12 braking screen; freeze before reading motion or model errors',
              'role':'additional finite development window; may overlap earlier non-braking research, NOT independent confirmation',
              'previous_screen_unchanged':str(PREVIOUS),'start_offsets_seconds':[2.5,4.5,6.5],'measurement_frames':FRAMES,
              'history':'at least20 actual captures before start; no0.5s cold-start windows',
              'motion_gate':{'initial_speed_min_mps':2.,'speed_drop_over_2s_min_mps':1.},
              'task_gate':{'target':'first route leader during first2s; it need not be inside40m at initial frame',
                           'target_leader_observations_min':3,'target_initial_bbox_min_wh':[48,32],
                           'initial_and_all_leader_projections_fully_inside':True,'oracle_leader_braking_times_min':2,
                           'oracle_acceleration_max_mps2':-.5,'leader_effect_vs_free_acceleration_max_mps2':-.5},
              'selection_order':'first qualifying start per log; first2 qualified logs for real RGB baseline, independent of reconstruction error',
              'same_next_policy':'original world-distance association, fixed20 past captures, official height raster, sameIDM; rejected IoU variant not adopted',
              'scope':'CPU screen only; task geometry qualification does not admit generation; real RGB then GT feedback still required',
              'stop':'exact6x3 windows, no replacement/additional offsets; close if none qualify; never change thresholds to rescue candidates',
              'human_verdict':None,'failure_ledger_delta':'none','failure_ledger_refs':['V74-H2-F22']}
    save(OUT/'protocol.json',protocol)
    policy=object.__new__(RGBIDMPolicy);policy.idm=IDMPolicy(target_velocity=10.,min_gap_to_lead_agent=1.,headway_time=1.5,accel_max=1.,decel_max=3.)
    rows=[];cases=[]
    for raw in logs:
        images=sorted((raw/'sensors/cameras'/CAMERA).glob('*.jpg'));times=np.array([int(p.stem) for p in images],np.int64)
        ego=pd.read_feather(raw/'city_SE3_egovehicle.feather');qualified=[]
        for offset in protocol['start_offsets_seconds']:
            index=np.searchsorted(times,times[0]+round(offset*1e9));start=int(times[index]);row={'log_id':raw.name,'offset_seconds':offset,'start_ns':start}
            if index<20 or start-500_000_000<ego.timestamp_ns.min() or start+7_866_666_667>min(times[-1],ego.timestamp_ns.max()):
                row['reason']='insufficient_fixed_input_or_history';rows.append(row);continue
            stamps=start+np.array([-.5,0,1.5,2])*1e9;E=poses(ego,stamps.astype(np.int64))
            speeds=[float((E[b,:3,3]-E[a,:3,3])@E[b,:3,0]/.5) for a,b in [(0,1),(2,3)]]
            row.update(recorded_speed_mps=speeds,recorded_speed_drop_mps=speeds[0]-speeds[1])
            if speeds[0]<2 or speeds[0]-speeds[1]<1:
                row['reason']='no_recorded_braking_demand';rows.append(row);continue
            base=OUT/'cases'/raw.name/f'start-{offset:.1f}'/'base';base.parent.mkdir(parents=True)
            cmd=[sys.executable,str(Path(__file__).with_name('prepare_argoverse.py')),'--output',str(base),'--log-id',raw.name,
                 '--start-offset-seconds',str(offset),'--task-id',protocol['task_id']]
            with (base.parent/'prepare.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            data=scene(base);tr=np.load(base/'trajectory.npz');route=Route(tr['ego_world'][:,:3,3])
            leads=[reference_lead(data,tr,route,f) for f in FRAMES]
            first=next((x for x in leads if x is not None),None)
            track=next((t for t in data['tracks'] if first and t['id']==first['id'] and t['segment']==first['segment']),None)
            target=first['id'] if first else None;initial=project_track(track,0,tr['camera_world'],tr['K']) if track and 0 in track['frames'] else None
            actions=[];projections=[]
            for f,lead in zip(FRAMES,leads):
                s=logged_state(base,tr,f);o=oracle_lead(lead,data,tr,f,s['yaw_rad'],s['speed_mps'])
                acc=policy.acceleration(s['speed_mps'],o);free=policy.acceleration(s['speed_mps'],None)
                actions.append({'frame':f,'oracle_acceleration_mps2':acc,'leader_effect_mps2':acc-free})
                t=next((t for t in data['tracks'] if lead and t['id']==lead['id'] and t['segment']==lead['segment']),None)
                projections.append(project_track(t,f,tr['camera_world'],tr['K']) if t else None)
            count=sum(bool(x and x['id']==target) for x in leads) if target else 0
            box=initial['bounds'] if initial else [0,0,0,0]
            visible=bool(initial and initial['fully_inside'] and all(p and p['fully_inside'] for lead,p in zip(leads,projections) if lead))
            braking=sum(a['oracle_acceleration_mps2']<=-.5 and a['leader_effect_mps2']<=-.5 for a in actions)
            okay=count>=3 and visible and box[2]-box[0]>=48 and box[3]-box[1]>=32 and braking>=2
            row.update(base=str(base),target=target,leaders=leads,initial_target_projection=initial,projections=projections,
                       oracle_actions=actions,target_times=count,fully_visible=visible,oracle_braking_times=braking,
                       qualified=bool(okay),reason='task_geometry_qualified' if okay else 'no_visible_leader_braking_task')
            rows.append(row)
            if okay:qualified.append(row)
        if qualified:cases.append(qualified[0])
        save(OUT/'progress.json',{'rows':rows,'qualified_logs':[x['log_id'] for x in cases]})
        print(json.dumps({'log':raw.name,'qualified_starts':len(qualified)}),flush=True)
    selected=cases[:2]
    for c in selected:
        base=Path(c['base']);m=json.loads((base/'input_manifest.json').read_text());tr=np.load(base/'trajectory.npz')
        files=sorted((Path(m['raw_path'])/'sensors/cameras'/CAMERA).glob('*.jpg'));ts=np.array([int(p.stem) for p in files])
        sheet=Image.new('RGB',(1280,5*378),'#152839')
        for i,f in enumerate(FRAMES):
            idx=int(np.searchsorted(ts,tr['timestamps_ns'][f],side='right')-1)
            rgb=crop_image(files[idx],m['crop_xyxy']);overlay=rgb.copy();p=c['projections'][i]
            if p:ImageDraw.Draw(overlay).rectangle(p['bounds'],outline='yellow',width=3)
            y=i*378;ImageDraw.Draw(sheet).text((8,y+4),f'{c["log_id"][:8]} +{c["offset_seconds"]}s | t={f/30:.1f}s | recorded RGB / reference leader',fill='white')
            sheet.paste(rgb.resize((640,352)),(0,y+24));sheet.paste(overlay.resize((640,352)),(640,y+24))
        sheet.save(base.parent/'real-task-review.jpg',quality=94)
    result={'status':'complete','window_count':len(rows),'recorded_braking_windows':sum('base' in r for r in rows),
            'qualified_log_count':len(cases),'selected':selected,'rows':rows,'new_gpu_calls':0,'human_verdict':None,'failure_ledger_delta':'none'}
    save(OUT/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['rows','selected']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()

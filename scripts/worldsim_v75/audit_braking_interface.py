"""冻结筛查之外的入口诊断：真实RGB是否能处理随后进入作用距离的车辆。"""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from evaluate_following_baseline import logged_state, oracle_lead
from evaluate_localization import iou
from following_geometry import FRAMES, scene, reference_lead
from prepare_argoverse import CAMERA, CORNERS, crop_image, project_track
from rgb_idm_policy import RGBIDMPolicy, IDMPolicy
from scipy.spatial.transform import Rotation
from screen_braking_tasks import OUT, save


def main():
    source = json.loads((OUT/'task_sources.json').read_text()); assert source['status']=='complete'
    assert not source['cases'], '仅诊断已关闭、无合格任务的筛查入口'
    candidate = next(r for r in source['rows'] if r.get('oracle_braking_times',0)>=2)
    path = OUT/'interface-audit'; assert not path.exists(); path.mkdir()
    target = next(x['id'] for x in candidate['leaders'] if x)
    save(path/'protocol.json', {'task_id':'WS-V75-BRAKING-INTERFACE-AUDIT-01','run_id':OUT.name,
        'frozen_utc':datetime.now(timezone.utc).isoformat(),'source_run':str(OUT),
        'source_log':candidate['log_id'],'base':candidate['base'],'target':target,
        'selection':'explicit post hoc engineering diagnosis of first existing window with >=2 oracle leader-induced braking times; original 48-window result unchanged',
        'frames':FRAMES,'policy':'unchanged cached FasterRCNN + calibrated ground contact + ordinary Kalman + official nuPlan IDM',
        'purpose':'separate initial 40m selection exclusion from actual RGB policy behavior; no world-model-error or performance claim',
        'stop':'five real observations only; no generation, threshold change or replacement task in this audit; any OOM stops',
        'human_verdict':None,'failure_ledger_delta':'none'})
    start=time.monotonic(); result={'status':'started','rows':[],'human_verdict':None,'failure_ledger_delta':'none',
                                   'generation_calls':0,'reconstruction_calls':0,'original_screening_admission':False}
    try:
        base=Path(candidate['base']); traj=np.load(base/'trajectory.npz'); data=scene(base)
        manifest=json.loads((base/'input_manifest.json').read_text())
        track=next(t for t in data['tracks'] if t['id']==target and 0 in t['frames'])
        extrinsic=np.linalg.inv(traj['ego_world'][0])@traj['camera_world'][0]
        policy=RGBIDMPolicy(traj['ego_world'][:,:3,3],traj['K'],extrinsic,candidate['ground_plane'])
        images=sorted((Path(manifest['raw_path'])/'sensors/cameras'/CAMERA).glob('*.jpg'))
        stamps=np.array([int(p.stem) for p in images],np.int64)
        sheet=Image.new('RGB',(1280,5*410),'#122335'); font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
        for i,f in enumerate(FRAMES):
            k=abs(stamps-traj['timestamps_ns'][f]).argmin(); assert abs(stamps[k]-traj['timestamps_ns'][f])<=1000
            rgb=crop_image(images[k],manifest['crop_xyxy']); rgb.save(path/f'real-{f:03d}.png')
            state=logged_state(base,traj,f)
            policy.step(rgb=np.array(rgb),frame_index=f,timestamp_us=int(traj['timestamps_us'][f]),
                        ego_state=state,camera_override=traj['camera_world'][f])
            ref=reference_lead(data,traj,policy.route,f); pred=policy.last['lead']
            oracle=oracle_lead(ref,data,traj,f,state['yaw_rad'],state['speed_mps'])
            oracle_acc=policy.acceleration(state['speed_mps'],oracle)
            projection=project_track(track,f,traj['camera_world'],traj['K'])
            j=track['frames'].index(f); rot=Rotation.from_quat(track['quaternions'][j]).as_matrix()
            corners=(CORNERS*np.array(track['dimensions']))@rot.T+track['centers'][j]
            ego_s=policy.route.project([state['x_m'],state['y_m']])[0]
            # 不改策略40m范围；额外目标gap只作为参考解释为何起点未入组。
            target_gap=min(policy.route.project(c)[0] for c in corners)-ego_s-2.4
            correct=(pred is None and ref is None) or (pred is not None and ref is not None and pred['box'] is not None and iou(pred['box'],projection['bounds'])>=.3)
            row={'frame':f,'target_projection':projection,'reference_target_gap_without_40m_filter_m':target_gap,
                 'reference_lead':ref,'oracle':oracle,'policy':policy.last,'correct_leader':bool(correct),
                 'oracle_acceleration_mps2':oracle_acc,'acceleration_difference_mps2':policy.last['acceleration_mps2']-oracle_acc,
                 'gap_error_m':abs(pred['gap_m']-ref['gap_m']) if correct and ref else None}
            result['rows'].append(row)
            image=rgb.copy(); draw=ImageDraw.Draw(image)
            if projection: draw.rectangle(projection['bounds'],outline='yellow',width=3)
            if pred: draw.rectangle(pred['box'],outline='#22e6b3',width=3)
            y=i*410; d=ImageDraw.Draw(sheet)
            d.text((8,y+3),f't={f/30:.1f}s | target GT gap={target_gap:.2f}m | policy gap={pred["gap_m"] if pred else None}',font=font,fill='white')
            d.text((8,y+25),f'accel RGB={policy.last["acceleration_mps2"]:.3f} / oracle={oracle_acc:.3f} m/s2 | yellow=target GT; green=policy lead',font=font,fill='white')
            sheet.paste(image.resize((640,352)),(0,y+56))
            sheet.paste(image.crop((320,220,1120,640)).resize((640,336)),(640,y+56))
        sheet.save(path/'real-policy-review.jpg',quality=94)
        errors=[abs(x['acceleration_difference_mps2']) for x in result['rows']]
        gaps=[x['gap_error_m'] for x in result['rows'] if x['gap_error_m'] is not None]
        result.update(status='complete',correct_leader_fraction=float(np.mean([x['correct_leader'] for x in result['rows']])),
                      median_abs_gap_error_m=float(np.median(gaps)) if gaps else None,
                      median_abs_acceleration_difference_mps2=float(np.median(errors)),
                      maximum_underbraking_mps2=float(max(max(x['acceleration_difference_mps2'],0) for x in result['rows'])),
                      maximum_overbraking_mps2=float(max(max(-x['acceleration_difference_mps2'],0) for x in result['rows'])),
                      original_screening_result_unchanged=True)
    except BaseException as exc:
        result.update(status='oom_stopped' if isinstance(exc,torch.OutOfMemoryError) else 'failed_stopped',
                      error_type=type(exc).__name__,error=str(exc)); raise
    finally:
        result.update(wall_s=time.monotonic()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        save(path/'result.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


def state_control():
    """相同IDM的离线状态替换；额外GT只用于解释，不称作策略改善。"""
    folder=OUT/'interface-audit'; dest=folder/'state_control_result.json'; assert not dest.exists()
    result=json.loads((folder/'result.json').read_text()); assert result['status']=='complete'
    policy=object.__new__(RGBIDMPolicy)
    policy.idm=IDMPolicy(target_velocity=10.,min_gap_to_lead_agent=1.,headway_time=1.5,accel_max=1.,decel_max=3.)
    rows=[]
    for r in result['rows']:
        if r['oracle'] is None or r['policy']['lead'] is None: continue
        gt,rgb=r['oracle'],r['policy']['lead']; speed=r['policy']['ego_state']['speed_mps']
        values={}
        for label,gap,velocity in [('reference',gt['gap_m'],gt['lead_speed_mps']),
                                   ('rgb_gap_only',rgb['gap_m'],gt['lead_speed_mps']),
                                   ('rgb_velocity_only',gt['gap_m'],rgb['lead_speed_mps']),
                                   ('both_rgb',rgb['gap_m'],rgb['lead_speed_mps'])]:
            values[label]=policy.acceleration(speed,{'gap_m':gap,'lead_speed_mps':velocity})
        assert abs(values['reference']-r['oracle_acceleration_mps2'])<1e-8
        assert abs(values['both_rgb']-r['policy']['acceleration_mps2'])<1e-8
        rows.append({'frame':r['frame'],'acceleration_mps2':values})
    value={'status':'complete','source_result':str(folder/'result.json'),'rows':rows,
           'role':'post hoc offline IDM state substitution with extra GT; not a generated rollout or policy improvement',
           'generation_calls':0,'human_verdict':None,'failure_ledger_delta':'none'}
    save(dest,value); print(json.dumps(value),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--state-control',action='store_true'); args=parser.parse_args()
    state_control() if args.state_control else main()

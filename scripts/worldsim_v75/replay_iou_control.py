"""一次固定IoU关联强控制：真实基线与四份已保存生成观测，不重跑检测。"""
import copy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from evaluate_localization import iou
from rgb_idm_policy import RGBIDMPolicy,MotionTracks
from iou_motion_tracks import ImageIoUMotionTracks

SOURCE = Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-APPROACH-BASELINE-01/20260920-association-r2')
GENERATED = Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-APPROACH-CLOSEDLOOP-01/20260920-association-r2')
STATE = Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-APPROACH-STATE-CONTROL-01/20260920-r1')
OUT = Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-IOU-ASSOCIATION-CONTROL-01/20260920-r1')
ARMS = ['gt_clean','dvgt_metric','dvgt_lidar_scaled','reference_lidar']


def save(path, obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')


def new_policy(base, tracker_type):
    tr=np.load(base/'trajectory.npz')
    p=RGBIDMPolicy(tr['ego_world'][:,:3,3],tr['K'],np.linalg.inv(tr['ego_world'][0])@tr['camera_world'][0],[0,0,0],detector=False)
    p.tracker=tracker_type();return p


def step(policy, prior):
    state=prior['ego_state'];det=copy.deepcopy(prior['detections'])
    for d in det:
        d.pop('track_id',None);d.pop('world_velocity',None)
    policy.tracker.update(det,prior['timestamp_us']/1e6,state['speed_mps']*np.array([np.cos(state['yaw_rad']),np.sin(state['yaw_rad'])]))
    lead=policy.choose_lead(det,state);acc=policy.acceleration(state['speed_mps'],lead)
    return {**prior,'detections':det,'lead':lead,'acceleration_mps2':acc}


def metrics(rows, key='acceleration_difference_mps2'):
    diffs=np.array([r[key] for r in rows])
    return {'mean_abs_action_error_mps2':float(np.mean(abs(diffs))),
            'median_abs_action_error_mps2':float(np.median(abs(diffs))),
            'max_underbraking_mps2':float(np.maximum(diffs,0).max()),
            'max_overbraking_mps2':float(np.maximum(-diffs,0).max())}


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not OUT.exists();OUT.mkdir(parents=True)
    source=json.loads((SOURCE/'protocol.json').read_text());real=json.loads((SOURCE/'result.json').read_text())
    base=Path(source['base']);assert real['status']=='complete' and real['real_gate_passed']
    protocol={'task_id':'WS-V75-IOU-ASSOCIATION-CONTROL-01','run_id':OUT.name,
              'frozen_utc':datetime.now(timezone.utc).isoformat(),'base':str(base),'source_log':source['source_log'],'target':source['target'],
              'real_source':str(SOURCE),'generated_source':str(GENERATED),'state_control_source':str(STATE),
              'variant':'last observed image bbox IoU >=0.3 before Hungarian; same .6s expiry and exact metric Kalman',
              'source':'https://github.com/abewley/sort/blob/master/sort.py','source_transfer':'ordinary IoU matching and default0.3; NOT full SORT or official tracker execution',
              'fixed':['saved detector candidates','ground-contact position','metric Kalman dynamics/noise','ego-speed cold start','IDM','route','time sequence'],
              'real_gate':source['real_gate'],
              'offline_gt_gate':{'median_abs_action_error_mps2':.5,'max_underbraking_mps2':2.,'max_overbraking_mps2':2.},
              'offline_improvement':'reference_lidar after shared startup: mean absolute action error and max underbraking both strictly decrease; max overbraking does not increase',
              'target_reference':'intended condition-state action at same saved ego; original fixed-physical reference also retained',
              'future_feedback_if_passed':'one paired GT/reference_lidar feedback trial, same117frames/seed42; GT full gate first; any OOM stops',
              'stop':'one fixed association variant; no IoU or world-Kalman tuning; if any gate fails close this control without new GPU work',
              'role':'exposed development control, offline changed actions do not prove closed-loop recovery',
              'human_verdict':None,'failure_ledger_delta':'none'}
    save(OUT/'protocol.json',protocol);start=time.monotonic();result={'status':'running','human_verdict':None,'failure_ledger_delta':'none'}
    try:
        # 先验证提取关联接口没有改变原策略，覆盖真实及全部生成输入。
        exact_count=0
        for arm in ['real']+ARMS:
            p=new_policy(base,MotionTracks)
            priors=[r['policy'] for r in real['warmup']]
            tail=real['rows'] if arm=='real' else json.loads((GENERATED/arm/'decisions.json').read_text())
            priors += [r['policy'] for r in tail]
            for prior in priors:
                q=step(p,prior)
                assert q['lead']==prior['lead'] and q['detections']==prior['detections']
                assert q['acceleration_mps2']==prior['acceleration_mps2']
                exact_count+=1
        save(OUT/'original_replay_check.json',{'status':'passed','observations_checked':exact_count,'unique_observations':95,'new_detector_calls':0})
        p=new_policy(base,ImageIoUMotionTracks);new_real=copy.deepcopy(real)
        for r in new_real['warmup']: r['policy']=step(p,r['policy'])
        snapshot={'previous_time':p.tracker.previous_time,'serial':p.tracker.serial,
                  'tracks':[{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in t.items()} for t in p.tracker.tracks]}
        save(OUT/'tracker_at_t0.json',snapshot)
        for r in new_real['rows']:
            r['policy']=step(p,r['policy']);lead=r['policy']['lead'];ref=r['reference_lead'];projection=r['projection']
            correct=bool(lead is None and ref is None or lead and ref and projection and iou(lead['box'],projection['bounds'])>=.3)
            r.update(correct_leader=correct,gap_error_m=abs(lead['gap_m']-ref['gap_m']) if correct and ref else None,
                     acceleration_difference_mps2=r['policy']['acceleration_mps2']-r['oracle_acceleration_mps2'])
        rm=metrics(new_real['rows']);gap=float(np.median([r['gap_error_m'] for r in new_real['rows'] if r['gap_error_m'] is not None]))
        frac=float(np.mean([r['correct_leader'] for r in new_real['rows']]))
        passed=frac>=.8 and gap<=3 and rm['median_abs_action_error_mps2']<=.5 and rm['max_underbraking_mps2']<=2 and rm['max_overbraking_mps2']<=2
        new_real.update(real_gate_passed=bool(passed),new_detector_calls=0,world_model_calls=0,wall_s=None,peak_allocated_gib=0,
                        correct_leader_fraction=frac,median_abs_gap_error_m=gap,
                        median_abs_acceleration_difference_mps2=rm['median_abs_action_error_mps2'],max_underbraking_mps2=rm['max_underbraking_mps2'],max_overbraking_mps2=rm['max_overbraking_mps2'])
        save(OUT/'real_result.json',new_real)
        condition_rows=json.loads((STATE/'offline_action_comparison.json').read_text())['rows'];summaries=[]
        for arm in ARMS:
            p=new_policy(base,ImageIoUMotionTracks)
            for r in real['warmup']:step(p,r['policy'])
            old=json.loads((GENERATED/arm/'decisions.json').read_text());rows=[]
            for r in old:
                q=step(p,r['policy']);f=r['observation_frame']
                c=next(c for c in condition_rows if c['arm']==arm and c['frame']==f)
                rows.append({'frame':f,'policy':q,'old_policy':r['policy'],
                             'condition_lead':c['condition_lead'],'condition_acceleration_mps2':c['condition_acceleration_at_own_ego_mps2'],
                             'acceleration_difference_mps2':q['acceleration_mps2']-c['condition_acceleration_at_own_ego_mps2'],
                             'old_acceleration_difference_mps2':r['policy']['acceleration_mps2']-c['condition_acceleration_at_own_ego_mps2'],
                             'physical_reference_acceleration_difference_mps2':q['acceleration_mps2']-r['oracle_acceleration_mps2']})
            save(OUT/f'{arm}.json',{'status':'complete','rows':rows,'role':'offline original-ego replay; no new pixels or executed trajectory'})
            summaries.append({'arm':arm,'original':metrics(rows[1:],'old_acceleration_difference_mps2'),
                              'iou':metrics(rows[1:]),'all_decision_physical_reference':metrics(rows,'physical_reference_acceleration_difference_mps2'),
                              'chosen_track_ids':[r['policy']['lead']['track_id'] if r['policy']['lead'] else None for r in rows]})
        gt=summaries[0]['all_decision_physical_reference']
        gt_pass=gt['median_abs_action_error_mps2']<=.5 and gt['max_underbraking_mps2']<=2 and gt['max_overbraking_mps2']<=2
        a,b=summaries[-1]['original'],summaries[-1]['iou']
        improvement=b['mean_abs_action_error_mps2']<a['mean_abs_action_error_mps2'] and b['max_underbraking_mps2']<a['max_underbraking_mps2'] and b['max_overbraking_mps2']<=a['max_overbraking_mps2']
        assert not torch.cuda.is_initialized()
        result.update(status='complete',real_gate_passed=bool(passed),real_metrics={**rm,'correct_leader_fraction':frac,'median_abs_gap_error_m':gap},
                      generated=summaries,offline_gt_gate_passed=bool(gt_pass),offline_reference_improved=bool(improvement),
                      generation_admitted=bool(passed and gt_pass and improvement),original_replay_observations=exact_count,
                      new_detector_calls=0,new_generator_calls=0,cuda_initialized=False)
    except BaseException as exc:
        result.update(status='failed_stopped',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        result['wall_s']=time.monotonic()-start;save(OUT/'result.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

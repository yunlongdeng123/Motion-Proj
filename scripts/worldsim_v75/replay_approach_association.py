"""在完整已保存真实感知序列上验证距离门控修复，不重复检测或改冻结输入。"""
from datetime import datetime,timezone
import copy,json
from pathlib import Path
import numpy as np
from rgb_idm_policy import RGBIDMPolicy,MotionTracks,IDMPolicy
from following_geometry import Route
from evaluate_localization import iou
from qualify_approach_policy import OUT as SOURCE
from qualify_raster_policy import save


def main():
    dest=SOURCE.parent/'20260920-association-r2'; assert not dest.exists(); dest.mkdir()
    source=json.loads((SOURCE/'protocol.json').read_text()); original=json.loads((SOURCE/'result.json').read_text())
    assert original['status']=='complete'
    protocol=copy.deepcopy(source); protocol.update(run_id=dest.name,frozen_utc=datetime.now(timezone.utc).isoformat(),
        replay_source=str(SOURCE),engineering_revision='distance gate before Hungarian; same10m cutoff,Kalman,perception inputs and IDM',
        role='engineering repair replay of same real RGB; not new source or independent sample')
    save(dest/'protocol.json',protocol)
    tr=np.load(Path(source['base'])/'trajectory.npz'); policy=object.__new__(RGBIDMPolicy)
    policy.route=Route(tr['ego_world'][:,:3,3]); policy.tracker=MotionTracks()
    policy.idm=IDMPolicy(target_velocity=10.,min_gap_to_lead_agent=1.,headway_time=1.5,accel_max=1.,decel_max=3.)
    output=copy.deepcopy(original)
    for section in ['warmup','rows']:
        for row in output[section]:
            prior=row['policy']; state=prior['ego_state']; detections=copy.deepcopy(prior['detections'])
            policy.tracker.update(detections,prior['timestamp_us']/1e6,state['speed_mps']*np.array([np.cos(state['yaw_rad']),np.sin(state['yaw_rad'])]))
            lead=policy.choose_lead(detections,state); acc=policy.acceleration(state['speed_mps'],lead)
            prior.update(detections=detections,lead=lead,acceleration_mps2=acc)
            if section=='rows':
                ref=row['reference_lead']; p=row['projection']
                correct=bool(lead is None and ref is None or lead and ref and p and iou(lead['box'],p['bounds'])>=.3)
                row.update(correct_leader=correct,gap_error_m=abs(lead['gap_m']-ref['gap_m']) if correct and ref else None,
                           acceleration_difference_mps2=acc-row['oracle_acceleration_mps2'])
        if section=='warmup':
            snapshot={'previous_time':policy.tracker.previous_time,'serial':policy.tracker.serial,
                      'tracks':[{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in t.items()} for t in policy.tracker.tracks]}
            save(dest/'tracker_at_t0.json',snapshot)
    rows=output['rows']; gaps=[r['gap_error_m'] for r in rows if r['gap_error_m'] is not None]
    gap=float(np.median(gaps)) if gaps else None; acc=float(np.median([abs(r['acceleration_difference_mps2']) for r in rows])); fraction=float(np.mean([r['correct_leader'] for r in rows]))
    under=max(max(r['acceleration_difference_mps2'],0) for r in rows); over=max(max(-r['acceleration_difference_mps2'],0) for r in rows)
    output.update(real_gate_passed=bool(fraction>=.8 and gap is not None and gap<=3 and acc<=.5 and under<=2 and over<=2),
                  correct_leader_fraction=fraction,median_abs_gap_error_m=gap,median_abs_acceleration_difference_mps2=acc,
                  max_underbraking_mps2=under,max_overbraking_mps2=over,new_detector_calls=0,saved_detector_calls_reused=35,
                  world_model_calls=0,wall_s=None,peak_allocated_gib=0.,role='offline replay, no new detector or generation')
    save(dest/'result.json',output)
    print(json.dumps({k:v for k,v in output.items() if k not in ['rows','warmup']}),flush=True)


if __name__=='__main__': main()

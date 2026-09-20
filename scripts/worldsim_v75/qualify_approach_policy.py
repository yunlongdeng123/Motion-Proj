"""单个已暴露接近任务的普通基线：高程地面＋合法真实历史，一次固定执行。"""
from dataclasses import asdict
from datetime import datetime,timezone
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation
import torch
from raster_rgb_policy import RasterRGBIDMPolicy
from closed_loop_bridge import VehicleState
from evaluate_following_baseline import oracle_lead
from evaluate_localization import iou
from prepare_argoverse import CAMERA,POS,QUAT,poses,crop_image,project_track
from following_geometry import scene,reference_lead,Route
from qualify_raster_policy import SOURCE,save

OUT=Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-APPROACH-BASELINE-01/20260920-r1')


def at_time(ego,stamp,origin):
    matrices=poses(ego,[stamp-50_000_000,stamp]); matrices[:,:3,3]-=origin
    pose=matrices[1]; velocity=(pose[:3,3]-matrices[0,:3,3])/.05
    roll,pitch,yaw=Rotation.from_matrix(pose[:3,:3]).as_euler('xyz')
    state=VehicleState(*map(float,pose[:3,3]),float(yaw),float(velocity@pose[:3,0]),0.,
                       pitch_rad=float(pitch),roll_rad=float(roll),velocity_x_mps=float(velocity[0]),velocity_y_mps=float(velocity[1]))
    return asdict(state),pose


def reference_at_times(base,timestamps,ego_matrices):
    data=scene(base); original=np.load(base/'trajectory.npz'); tracks=[]
    for t in data['tracks']:
        ns=original['timestamps_ns'][t['frames']]; good=np.flatnonzero((timestamps>=ns[0])&(timestamps<=ns[-1]))
        if len(ns)<2 or not len(good): continue
        frame=pd.DataFrame({'timestamp_ns':ns}); frame[POS]=t['centers']; frame[QUAT]=t['quaternions']
        p=poses(frame,timestamps[good]); entry=copy.deepcopy(t)
        entry.update(frames=good.tolist(),centers=p[:,:3,3].tolist(),quaternions=Rotation.from_matrix(p[:,:3,:3]).as_quat().tolist()); tracks.append(entry)
    data['tracks']=tracks
    tr={'ego_world':np.array(ego_matrices),'timestamps_us':np.rint((timestamps-timestamps[0])/1e3).astype(np.int64)+1_000_000}
    return data,tr


def main():
    global OUT
    parser=argparse.ArgumentParser(); parser.add_argument('--run-dir',type=Path,default=OUT)
    parser.add_argument('--source-protocol',type=Path,default=SOURCE/'protocol.json')
    parser.add_argument('--task-id',default='WS-V75-APPROACH-BASELINE-01')
    args=parser.parse_args(); OUT=args.run_dir
    assert not (OUT/'protocol.json').exists() and not (OUT/'result.json').exists()
    OUT.mkdir(parents=True,exist_ok=True)
    source=json.loads(args.source_protocol.read_text()); base=Path(source['base'])
    manifest=json.loads((base/'input_manifest.json').read_text()); raw=Path(manifest['raw_path']); origin=np.array(manifest['city_origin'])
    ego=pd.read_feather(raw/'city_SE3_egovehicle.feather'); tr=np.load(base/'trajectory.npz')
    images=sorted((raw/'sensors/cameras'/CAMERA).glob('*.jpg')); times=np.array([int(p.stem) for p in images],np.int64)
    cutoff=int(manifest['first_timestamp_ns']); history=np.flatnonzero(times<cutoff)[-20:]
    frames=[0]+[4+8*i for i in range(14)]; requested=tr['timestamps_ns'][frames]
    indices=np.searchsorted(times,requested,side='right')-1; captures=times[indices]
    # 官方20Hz文件含纳秒级抖动；固定20个先前capture，不用恰好1秒的浮点边界截断。
    assert len(history)==20 and cutoff-times[history[0]]<1_001_000_000
    assert np.unique(captures).size==15 and ((requested-captures)<50_001_000).all()
    protocol={'task_id':args.task_id,'run_id':OUT.name,'frozen_utc':datetime.now(timezone.utc).isoformat(),
              'base':str(base),'source_log':source['source_log'],'target':source['target'],
              'role':'one exposed development approach task; not re-admission of closed 48-window screening',
              'motivation':'known height raster fixes nonplanar-ground assumption; neutral-speed tracker cold start has no prior motion evidence',
              'change':'ordinary official ground-height map and fixed1s real RGB prehistory; same detector, Kalman parameters,40m range and IDM',
              'warmup_captures_ns':times[history].tolist(),'cutoff_ns':cutoff,'requested_frames':frames,
              'actual_captures_ns':captures.tolist(),'image_delay_ms':((requested-captures)/1e6).tolist(),
              'extra_information':['known raster ground map','20 actual RGB frames before initial generation timestamp'],
              'real_gate':{'correct_leader_fraction':.8,'median_abs_gap_error_m':3.,'median_abs_acceleration_difference_mps2':.5,
                           'max_underbraking_mps2':2.,'max_overbraking_mps2':2.},
              'generation_if_passed':'single seed42 GT-generated117-frame feedback baseline with shared warmup and terrain; before reconstruction arms',
              'stop':'one fixed preparation; no threshold/seed/source sweep; OOM stops immediately; failed real state or GT feedback blocks reconstruction comparison',
              'human_verdict':None,'failure_ledger_delta':'none','failure_ledger_refs':['V74-H2-F22']}
    if args.source_protocol!=SOURCE/'protocol.json':
        protocol.update(source_protocol=str(args.source_protocol),
                        role='fixed selected development task from BRAKING-DEV2; not independent confirmation',
                        motivation='apply frozen terrain-aware policy and20 legal past captures without tuning; task selected before model-error inspection')
    save(OUT/'protocol.json',protocol); result={'status':'started','rows':[],'warmup':[],'human_verdict':None,'failure_ledger_delta':'none'}
    began=time.monotonic()
    try:
        policy=RasterRGBIDMPolicy(base); matrices=[]; states=[]
        for stamp in captures:
            s,p=at_time(ego,int(stamp),origin); states.append(s); matrices.append(p)
        data,reference=reference_at_times(base,captures,matrices)
        camera=np.array(matrices)@policy.extrinsic
        for idx in history:
            stamp=int(times[idx]); state,pose=at_time(ego,stamp,origin)
            rgb=crop_image(images[idx],manifest['crop_xyxy'])
            policy.step(rgb=np.array(rgb),frame_index=-len(history)+len(result['warmup']),
                        timestamp_us=round((stamp-cutoff)/1e3)+1_000_000,ego_state=state,camera_override=pose@policy.extrinsic)
            result['warmup'].append({'timestamp_ns':stamp,'image':str(images[idx]),'policy':copy.deepcopy(policy.last)})
        save(OUT/'tracker_at_t0.json',policy.tracker_snapshot())
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
        sheet=Image.new('RGB',(1280,15*408),'#142335')
        for i,idx in enumerate(indices):
            stamp=int(times[idx]); state=states[i]; rgb=crop_image(images[idx],manifest['crop_xyxy'])
            rgb.save(OUT/f'real-{frames[i]:03d}.png')
            policy.step(rgb=np.array(rgb),frame_index=frames[i],timestamp_us=int(reference['timestamps_us'][i]),
                        ego_state=state,camera_override=camera[i])
            ref=reference_lead(data,reference,policy.route,i); pred=policy.last['lead']
            oracle=oracle_lead(ref,data,reference,i,state['yaw_rad'],state['speed_mps']); acc=policy.acceleration(state['speed_mps'],oracle)
            track=next((t for t in data['tracks'] if ref and t['id']==ref['id'] and t['segment']==ref['segment']),None)
            projection=project_track(track,i,camera,tr['K']) if track else None
            correct=bool(pred is None and ref is None or pred and projection and iou(pred['box'],projection['bounds'])>=.3)
            row={'frame':frames[i],'actual_timestamp_ns':stamp,'policy':copy.deepcopy(policy.last),'reference_lead':ref,'oracle':oracle,
                 'projection':projection,'correct_leader':correct,'gap_error_m':abs(pred['gap_m']-ref['gap_m']) if correct and ref else None,
                 'oracle_acceleration_mps2':acc,'acceleration_difference_mps2':policy.last['acceleration_mps2']-acc}
            result['rows'].append(row)
            overlay=rgb.copy(); d=ImageDraw.Draw(overlay)
            if projection: d.rectangle(projection['bounds'],outline='yellow',width=3)
            if pred: d.rectangle(pred['box'],outline='#15e0aa',width=3)
            y=i*408; d=ImageDraw.Draw(sheet)
            d.text((8,y+3),f't={(stamp-cutoff)/1e9:.2f}s | GT gap={ref["gap_m"] if ref else None} | RGB gap={pred["gap_m"] if pred else None}',font=font,fill='white')
            d.text((8,y+25),f'IDM RGB={policy.last["acceleration_mps2"]:.3f} / GT={acc:.3f} m/s2 | fixed1s past RGB warmup + known terrain',font=font,fill='white')
            sheet.paste(overlay.resize((640,352)),(0,y+54)); sheet.paste(overlay.crop((320,220,1120,640)).resize((640,336)),(640,y+54))
        sheet.save(OUT/'real-policy-review.jpg',quality=94)
        rows=result['rows']; errors=[r['gap_error_m'] for r in rows if r['gap_error_m'] is not None]
        gap=float(np.median(errors)) if errors else None; acc=float(np.median([abs(r['acceleration_difference_mps2']) for r in rows])); fraction=float(np.mean([r['correct_leader'] for r in rows]))
        under=max(max(r['acceleration_difference_mps2'],0) for r in rows); over=max(max(-r['acceleration_difference_mps2'],0) for r in rows)
        passed=fraction>=.8 and gap is not None and gap<=3 and acc<=.5 and under<=2 and over<=2
        result.update(status='complete',real_gate_passed=bool(passed),correct_leader_fraction=fraction,median_abs_gap_error_m=gap,
                      median_abs_acceleration_difference_mps2=acc,max_underbraking_mps2=under,max_overbraking_mps2=over,
                      new_detector_calls=len(history)+len(rows),world_model_calls=0)
    except BaseException as exc:
        result.update(status='oom_stopped' if isinstance(exc,torch.OutOfMemoryError) else 'failed_stopped',error_type=type(exc).__name__,error=str(exc)); raise
    finally:
        result.update(wall_s=time.monotonic()-began,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        save(OUT/'result.json',result); print(json.dumps({k:v for k,v in result.items() if k not in ['rows','warmup']}),flush=True)


if __name__=='__main__': main()

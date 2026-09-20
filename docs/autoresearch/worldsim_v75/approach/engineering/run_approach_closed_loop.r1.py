"""接近车辆任务的正式生成反馈：先GT基线，任何OOM或基线失败均停止后续对照。"""
import argparse
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import time
import av
import numpy as np
from PIL import Image
import torch
from common import CAMERA,config
from closed_loop_bridge import FeedbackBridge,run_feedback
from interactive_drive.simulation.ground_snap import GroundSnapper
from evaluate_following_baseline import oracle_lead
from following_geometry import Route,scene,reference_lead
from raster_rgb_policy import RasterRGBIDMPolicy
from raster_ground import RasterGround
from run_following_closed_loop import reference_clearance
from qualify_approach_policy import OUT as SOURCE
from qualify_raster_policy import save

OUT=Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-APPROACH-CLOSEDLOOP-01/20260920-r1')
ARMS=['gt_clean','dvgt_metric','dvgt_lidar_scaled','reference_lidar']


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--phase',choices=['encode','generate'],required=True)
    parser.add_argument('--arm',choices=ARMS,default='gt_clean'); args=parser.parse_args()
    source=json.loads((SOURCE/'protocol.json').read_text()); qualification=json.loads((SOURCE/'result.json').read_text())
    assert qualification['status']=='complete' and qualification['real_gate_passed']
    base=Path(source['base']); OUT.mkdir(parents=True,exist_ok=True)
    protocol={'task_id':'WS-V75-APPROACH-CLOSEDLOOP-01','run_id':OUT.name,'source_run':str(SOURCE),
              'base':str(base),'source_log':source['source_log'],'target':source['target'],
              'frames':117,'blocks':15,'fps':30,'seed':42,'arms':ARMS,
              'role':'single exposed development approach task; state gate repaired using ordinary terrain and legal past images',
              'policy':'FasterRCNN ground-contact + Kalman + official nuPlan IDM;40m; frozen previous parameters',
              'shared_inputs':['initial RGB','generic daylight driving text','20 prior RGB frames for policy tracker only','known calibration/ego/route',
                               'known official height map for perception and official GroundSnapper','GT map and other actor states/trajectories'],
              'change_for_error_arms':'target initial translation readout only; same offset over actor trajectory; fixed physical reference',
              'gt_feedback_gate':{'max_route_lateral_m':1.,'no_reference_overlap':True,
                                  'median_abs_online_oracle_acceleration_difference_mps2':.5,'max_underbraking_mps2':2.,'max_overbraking_mps2':2.},
              'error_arm_admission':'GT feedback and dense reference pass, then reliable reference/model support; no residual-size ranking',
              'stop':'any OOM immediate stop without retry; GT feedback failure blocks reconstruction comparison; no seeds/threshold/source search',
              'human_verdict':None,'failure_ledger_delta':'none'}
    if (OUT/'protocol.json').exists(): assert json.loads((OUT/'protocol.json').read_text())==protocol
    else:
        save(OUT/'protocol.json',protocol); (OUT/'frozen_utc.txt').write_text(datetime.now(timezone.utc).isoformat()+'\n')
    conditioning=OUT/'conditioning'; conditioning.mkdir(exist_ok=True)
    if args.phase=='encode':
        path=conditioning/'result.json'; assert not path.exists()
        result={'status':'started','phase':'encode','human_verdict':None}; began=time.monotonic()
        try:
            shutil.copy2(base/'initial_rgb.png',conditioning/'initial_rgb.png')
            (conditioning/'prompt.txt').write_text('A forward-facing driving camera on an urban road in daylight. Vehicles, buildings, road markings and sidewalks are visible.\n')
            from run_prepared import encode
            result.update(encode(conditioning),status='complete')
        except BaseException as exc:
            result.update(status='oom_stopped' if isinstance(exc,torch.OutOfMemoryError) else 'failed_stopped',error_type=type(exc).__name__,error=str(exc)); raise
        finally:
            result.update(wall_s=time.monotonic()-began,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            save(path,result); print(json.dumps(result),flush=True)
        return
    assert json.loads((conditioning/'result.json').read_text())['status']=='complete'
    offset=np.zeros(3)
    if args.arm!='gt_clean':
        previous=json.loads((OUT/'gt_clean/result.json').read_text()); dense=json.loads((OUT/'gt_clean/dense_reference_result.json').read_text())
        assert previous['status']=='complete' and previous['baseline_admitted'] and dense['status']=='passed'
        read=json.loads((OUT/'reconstruction/readout_ray_control_result.json').read_text())
        assert read['generation_admitted'] and read['target']==source['target']
        offset=np.array(read['readouts'][args.arm]['offset_world_m'])
    dest=OUT/args.arm; assert not dest.exists(); dest.mkdir(); started=time.monotonic()
    result={'status':'started','arm':args.arm,'human_verdict':None,'failure_ledger_delta':'none'}
    try:
        reference=scene(base); condition=copy.deepcopy(reference)
        target=next(t for t in condition['tracks'] if t['id']==source['target'] and 0 in t['frames'])
        target['centers']=(np.array(target['centers'])+offset).tolist(); save(dest/'condition_scene.json',condition)
        tr=np.load(base/'trajectory.npz'); terrain=RasterGround(base); vertices,faces=terrain.mesh_for_route(tr['ego_world'][:,:2,3])
        snapper=GroundSnapper(vertices,faces); bridge=FeedbackBridge(base,condition,ground_snapper=snapper)
        policy=RasterRGBIDMPolicy(base); policy.restore_tracker(json.loads((SOURCE/'tracker_at_t0.json').read_text()))
        cfg=config(); cfg.text_encoder=None; cfg.image_encoder=None; cfg.diffusion_model.seed=42
        pipeline=cfg.setup().to('cuda').eval()
        embeddings=torch.load(conditioning/'embeddings.pt',weights_only=True,map_location='cpu')
        cache=pipeline.initialize_cache_from_embeddings(**embeddings,view_names=[CAMERA])
        generated_out=np.lib.format.open_memmap(dest/'generated.npy',mode='w+',dtype=np.uint8,shape=(117,704,1280,3))
        condition_out=np.lib.format.open_memmap(dest/'conditions.npy',mode='w+',dtype=np.uint8,shape=(117,704,1280,3))
        cameras=[]; decisions=[]; route=Route(tr['ego_world'][:,:3,3])
        with av.open(str(dest/'generated.mp4'),'w') as writer:
            stream=writer.add_stream('libx264',rate=30); stream.width=1280; stream.height=704; stream.pix_fmt='yuv420p'; stream.options={'crf':'18'}
            def consume(row,record,frames):
                ids=record['indices']; generated_out[ids]=frames; condition_out[ids]=record['conditions']; cameras.extend(record['camera_world'])
                f=row['observation_frame']; state=policy.last['ego_state']; gt=reference_lead(reference,tr,route,f,[state['x_m'],state['y_m']])
                oracle=oracle_lead(gt,reference,tr,f,state['yaw_rad'],state['speed_mps']); acc=policy.acceleration(state['speed_mps'],oracle)
                row.update(policy=copy.deepcopy(policy.last),oracle_lead=oracle,oracle_acceleration_mps2=acc,
                           acceleration_difference_mps2=policy.last['acceleration_mps2']-acc,
                           route_lateral_m=route.project([row['ego_state']['x_m'],row['ego_state']['y_m']])[1],
                           clearance=reference_clearance(reference,row['last_condition_frame'],row['ego_state']))
                decisions.append(row); save(dest/'decisions.json',decisions)
                for frame in frames:
                    for packet in stream.encode(av.VideoFrame.from_ndarray(frame,format='rgb24')): writer.mux(packet)
                Image.fromarray(frames[-1]).save(dest/f'generated-{ids[-1]:03d}.jpg',quality=94)
                print(json.dumps({'arm':args.arm,'block':row['block'],'frames':int(ids[-1])+1,'acceleration':policy.last['acceleration_mps2'],
                                  'gap':policy.last['lead']['gap_m'] if policy.last['lead'] else None}),flush=True)
            run_feedback(pipeline,cache,bridge,policy,np.array(Image.open(conditioning/'initial_rgb.png')),15,consume)
            for packet in stream.encode(): writer.mux(packet)
        generated_out.flush(); condition_out.flush(); np.savez(dest/'camera_trajectory.npz',camera_world=np.array(cameras),timestamps_us=tr['timestamps_us'][:117])
        with av.open(str(dest/'generated.mp4')) as video: decoded=sum(1 for _ in video.decode(video=0))
        assert decoded==117 and reference==scene(base)
        acc=float(np.median([abs(r['acceleration_difference_mps2']) for r in decisions])); under=max(max(r['acceleration_difference_mps2'],0) for r in decisions)
        over=max(max(-r['acceleration_difference_mps2'],0) for r in decisions); lateral=max(abs(r['route_lateral_m']) for r in decisions)
        overlap=sum(bool(r['clearance'] and r['clearance']['overlap_area_m2']>0) for r in decisions)
        admitted=acc<=.5 and under<=2 and over<=2 and lateral<=1 and overlap==0
        result.update(status='complete',frames=117,decoded_frames=decoded,decisions=len(decisions),actual_policy_feedback=True,
                      median_abs_online_oracle_acceleration_difference_mps2=acc,max_underbraking_mps2=under,max_overbraking_mps2=over,
                      max_route_lateral_m=lateral,reference_overlap_at_chunk_boundaries=overlap,
                      baseline_admitted=bool(admitted) if args.arm=='gt_clean' else None,offset_world_m=offset.tolist(),
                      final_speed_mps=bridge.state.speed_mps,terrain_vertices=len(vertices),terrain_faces=len(faces),
                      min_reference_boundary_clearance_m=min(r['clearance']['distance_m'] for r in decisions if r['clearance']))
    except BaseException as exc:
        result.update(status='oom_stopped' if isinstance(exc,torch.OutOfMemoryError) else 'failed_stopped',error_type=type(exc).__name__,error=str(exc)); raise
    finally:
        result.update(wall_s=time.monotonic()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        save(dest/'result.json',result); print(json.dumps(result),flush=True)


if __name__=='__main__': main()

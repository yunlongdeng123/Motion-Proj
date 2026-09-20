"""对官方场景池中的单条轨迹平移；保留其余张量与官方渲染路径。"""
import argparse
import dataclasses
import json
import time
from datetime import datetime, timezone
import numpy as np
from PIL import Image, ImageDraw
import torch
from interactive_drive.config import RasterConfig
from interactive_drive.rasterizer import LudusConditionRasterizer
from ludus_renderer import PRIM_OBSTACLE
from select_target import P1, box_projection, scene_and_camera
from common import RUN

def freeze():
    path = P1 / 'protocol.json'
    if path.exists():
        raise FileExistsError(path)
    scene, trajectory, camera = scene_and_camera()
    candidates = json.loads((P1 / 'target_candidates.json').read_text())
    eligible = [r for r in candidates['candidates'] if r['track_id'] != '99' and
                r['max_nearer_bbox_overlap_fraction'] < 0.1 and
                all(p is not None and p['fully_in_image'] for f, p in
                    zip([0,5,13,21,29,36,37,53,69,85,101,117,149,181,213,236],r['projections']) if f <= 53)]
    eligible.sort(key=lambda r:(-r['initial_area'],r['track_id']))
    target_id = eligible[0]['track_id']
    assert target_id == '81', '必须重新检查真实初帧，不自动接受未知目标'
    target = next(t for t in scene.vehicle_bbox_tracks if t.track_id == target_id)
    direction = trajectory['rig_poses_world'][0, :3, 0].astype(np.float64)
    direction /= np.linalg.norm(direction)
    projections = {}
    for name, value in [('clean',0.0),('negative',-0.5),('positive',0.5)]:
        projections[name] = [box_projection(target,f,trajectory,camera,value*direction) for f in range(237)]
    protocol = {'task_id':'WS-V75-LOCALIZE-01','run_id':'20260920-r1',
                'frozen_utc':datetime.now(timezone.utc).isoformat(), 'target_id':target_id,
                'role':'engineering_development_and_synthetic_condition_sensitivity',
                'physical_reference_changed':False,'natural_reconstruction_error':False,
                'seed_values':[42,43], 'frames':237,'fps':30,'offset_m':[-0.5,0.5],
                'direction_world':direction.tolist(), 'start_frame':5,'restore_frame':37,
                'primary_measurement_frames':[5,53], 'post_restore_frames':[37,53],
                'selection_rule':'full projected box at prespecified samples through frame53, initial>=16x12px, nearer-box overlap<10%, largest initial area; original RGB silhouette inspected',
                'selection_amendment_before_intervention':'Initial 3.37s filter left track99, whose lower body is behind a hedge. Rejected on real RGB. Shortened required visibility to frame53, leaving at least16 frames after restore; no biased video viewed.',
                'rejected_targets':{'99':'vegetation occlusion in real initial RGB',
                                    '93':'occluded by covered vehicle16; initial verbal identification was corrected before freezing',
                                    '16':'covered vehicle leaves full view before frame53'},
                'target_description':'oncoming white car, isolated front silhouette in real initial RGB',
                'measurements':['raw uint8 paired RGB difference in fixed reference ROI and outside it',
                                'condition projection change in pixels',
                                'target localization only if an independent evaluator passes on clean'],
                'claim_boundary':'pixel response is not metric state error; short recovery window, no policy feedback, no future RGB GT',
                'cases':['clean','negative_persistent','positive_persistent','negative_restore','positive_restore'],
                'human_verdict':None}
    (P1/'target_projections.json').write_text(json.dumps(projections,indent=2)+'\n')
    path.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    image=Image.fromarray(scene.initial_rgb)
    draw=ImageDraw.Draw(image)
    b=projections['clean'][0]['bounds']
    draw.rectangle(b,outline='yellow',width=3)
    draw.text((b[0]-40,b[1]-18),'Target 81 (input selection)',fill='yellow')
    image.save(P1/'target-frozen.png')
    print(json.dumps(protocol,ensure_ascii=False),flush=True)

def render():
    protocol=json.loads((P1/'protocol.json').read_text())
    scene,trajectory,camera=scene_and_camera()
    target=next(t for t in scene.vehicle_bbox_tracks if t.track_id==protocol['target_id'])
    renderer=LudusConditionRasterizer(RasterConfig(),max_chunk_frames=8)
    renderer.load_scene(scene)
    # 所有GPU场景读写仍在官方渲染器所属线程执行。
    executor,impl=renderer._require_alive()
    def locate():
        base=impl._scene_data.clipgt_scene.timestamped_scene
        matches=[]
        for pi,pool in enumerate(base.cube_pools):
            if pool.prim_type_id != PRIM_OBSTACLE:
                continue
            ends=pool.cube_ts_prefix_sum.cpu().numpy()
            stamps=pool.track_timestamps_us.cpu().numpy()
            centers=pool.translations.cpu().numpy()
            for ci,(lo,hi) in enumerate(zip(np.r_[0,ends[:-1]],ends)):
                if hi-lo!=len(target.timestamps_us):
                    continue
                if np.array_equal(stamps[lo:hi],target.timestamps_us) and np.allclose(centers[lo:hi],target.centers_world,rtol=0,atol=1e-5):
                    matches.append((pi,ci,int(lo),int(hi),pool))
        assert len(matches)==1,len(matches)
        return matches[0]
    pi,ci,lo,hi,pool=executor.submit(locate).result()
    origin=pool.translations.clone()
    changed={}
    def apply_offset(offset):
        positions=origin.clone()
        positions[lo:hi]+=torch.as_tensor(offset,device=positions.device,dtype=positions.dtype)
        assert torch.equal(positions[:lo],origin[:lo]) and torch.equal(positions[hi:],origin[hi:])
        replacement=dataclasses.replace(pool,translations=positions)
        assert impl.ctx.update_cube_pool_at_index(impl._scene_id,pi,replacement)
    clean=np.load(RUN/'conditions.npy',mmap_mode='r')
    began=time.monotonic()
    try:
        for name,amount in [('zero',0.0),('negative',-0.5),('positive',0.5)]:
            outpath=P1/f'conditions-{name}.npy'
            if outpath.exists():
                raise FileExistsError(outpath)
            executor.submit(apply_offset,amount*np.asarray(protocol['direction_world'])).result()
            images=np.lib.format.open_memmap(outpath,mode='w+',dtype=np.uint8,shape=clean.shape)
            total_changed=0
            per_frame=[]
            for start in range(0,237,8):
                result=renderer.render_chunk(trajectory['rig_poses_world'][start:start+8],trajectory['timestamps_us'][start:start+8])
                for i,frame in enumerate(result.frames):
                    rgb=np.asarray(frame.rgb_host_uint8)
                    images[start+i]=rgb
                    count=int(np.any(rgb!=clean[start+i],axis=-1).sum())
                    total_changed+=count
                    per_frame.append(count)
            images.flush()
            for f in [0,5,21,36,37,53]:
                Image.fromarray(images[f]).save(P1/f'condition-{name}-{f:03d}.png')
            changed[name]={'changed_pixels':total_changed,'per_frame_changed_pixels':per_frame}
            print(json.dumps({'render':name,'changed_pixels':total_changed}),flush=True)
            if name=='zero':
                assert total_changed==0,'零偏置没有复现原始条件，禁止运行偏置实验'
    finally:
        renderer.cleanup()
    result={'status':'passed','pool_index':pi,'cube_index':ci,'track_pose_slice':[lo,hi],
            'target_track_id':target.track_id,'map_and_non_target_tensor_references_retained':True,
            'non_target_translations_exactly_equal':True,'zero_offset_exactly_equal':True,
            'variants':changed,'elapsed_s':time.monotonic()-began,
            'note':'偏移为整条world轨迹常量；生成时按预注册帧区间选用clean/偏置数组'}
    (P1/'condition_audit.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['freeze','render'])
    a=p.parse_args()
    {'freeze':freeze,'render':render}[a.phase]()

"""Explicit-bank native initialization; no scene/actor/frame-specific repairs.

Current native engine input supports uint8 instance IDs. Unknown vehicles retain
binary foreground but never receive a fabricated positive identity. Real source
returns are depth-compared only in native source pixel bins, not coarse cells.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,shutil,sys,time
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

# 绑定由命令行显式传入；不会复制上一场景的高斯、车辆或道路资产。
ROOT=PREP=BANK=ORIGINAL=ENGINE=API=None

def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def ply(p,xyz,times):
    fields=[(k,'<f4') for k in ['x','y','z','nx','ny','nz']]+[(k,'u1') for k in ['red','green','blue']]+[('time','<f4')]
    rows=np.zeros(len(xyz),dtype=fields)
    for i,k in enumerate(['x','y','z']):rows[k]=xyz[:,i]
    for k in ['nx','ny','nz','time']:rows[k]=times
    header='ply\nformat binary_little_endian 1.0\nelement vertex '+str(len(xyz))+'\n'+''.join('property '+('uchar' if k in ['red','green','blue'] else 'float')+' '+k+'\n' for k in rows.dtype.names)+'end_header\n'
    with p.open('wb') as f:f.write(header.encode());rows.tofile(f)

def main():
    import argparse
    global ROOT,PREP,BANK,ORIGINAL,ENGINE,API
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--prep',type=Path,required=True)
    ap.add_argument('--engine-source',type=Path,required=True)
    ap.add_argument('--config-source',type=Path,required=True)
    ap.add_argument('--api-source',type=Path,required=True)
    ap.add_argument('--mask-root',type=Path,required=True)
    ap.add_argument('--registry',type=Path,required=True)
    ap.add_argument('--anchor-frame',type=int,required=True)
    ap.add_argument('--reuse-code',action='store_true')
    ap.add_argument('--training-source',type=Path,required=True)
    ap.add_argument('--defer-ready',action='store_true')
    a=ap.parse_args()
    ROOT,PREP,BANK=a.root,a.prep,a.prep/'inputs'
    ENGINE,ORIGINAL,API=a.engine_source,a.config_source.parent,a.api_source
    selection=json.loads((PREP/'selected_case.json').read_text())
    receipt=json.loads((BANK/'receipt.json').read_text())
    assert receipt['status']=='PASS_CPU_OBSERVATIONS'
    assert json.loads((BANK/'verification.json').read_text())['status']=='PASS_CPU_BANK'
    # 分割尚未完成时，先拒绝，避免创建半套三维训练目录。
    for row in receipt['frames']:
        assert (a.mask_root/f"{row['raw']:03d}"/'labels.png').is_file()
        label=np.asarray(Image.open(a.mask_root/f"{row['raw']:03d}"/'labels.png'))
        foreground=np.asarray(Image.open(a.mask_root/f"{row['raw']:03d}"/'dynamic_union.png'))>0
        unknown=np.asarray(Image.open(a.mask_root/f"{row['raw']:03d}"/'unknown_vehicle_identity.png'))>0
        if (label.ndim!=2 or label.dtype.kind not in 'iu' or foreground.shape!=label.shape
                or unknown.shape!=label.shape or not np.array_equal(foreground,(label>0)|unknown)
                or ((label>0)&unknown).any()):raise ValueError('Foreground/instance/unknown contracts disagree')
        assert (BANK/'sky/sky_0'/f"{row['id']}.png").is_file()
    from .v44_inputs import IdentityRegistry
    registry=IdentityRegistry(json.loads(a.registry.read_text())['track_to_actor'])
    if max(registry.actor_to_track,default=0)>255:raise ValueError('Current native PNG interface requires IDs <=255')
    anchor_raw=a.anchor_frame
    if anchor_raw not in receipt['build_raws']:raise ValueError('Coordinate anchor must be an input frame')
    ROOT.mkdir(exist_ok=True);out=ROOT/'inputs/native';out.mkdir(parents=True,exist_ok=False)
    if not a.reuse_code:
        shutil.copytree(ENGINE,ROOT/'engine',ignore=shutil.ignore_patterns('__pycache__','smoke_*','tests'))
        shutil.copytree(API/'drivingeditor_dynamic_edit',ROOT/'api/drivingeditor_dynamic_edit',ignore=shutil.ignore_patterns('__pycache__'))
    elif not (ROOT/'engine/scene').is_dir() or not (ROOT/'api/drivingeditor_dynamic_edit/ownership.py').is_file():
        raise ValueError('Explicit reuse-code requested but frozen engine/API is missing')
    # engine-source 是已验证的通用原生引擎；只复制代码，不复制旧场景输入。
    for name in ['calib','pose','timestamp','velodyne','image','sky']:(out/name).symlink_to(BANK/name,target_is_directory=True)
    shutil.copy2(BANK/'explicit_timeline.json',out/'explicit_timeline.json')
    for name in ['mask/mask_0/Annotations','dynamic_image/dynamic_image_0','dynamic_masks/dynamic_mask_0']:(out/name).mkdir(parents=True)
    receipt=json.loads((BANK/'receipt.json').read_text());actor_meta=json.loads((BANK/'track_registry.json').read_text())['actors'];point_groups={};time_groups={};world_clouds=[];time_clouds=[];stats=[]
    for row in receipt['frames']:
        raw=row['raw'];name=row['id'];obs=json.loads((BANK/'observations'/f'{raw:03d}.json').read_text())
        cloud=np.fromfile(BANK/'velodyne'/f'{name}.bin',dtype='<f4').reshape(-1,6);xyz=cloud[:,:3];ego=np.array(obs['ego_world']).reshape(4,4);world=(xyz@ego[:3,:3].T+ego[:3,3]).astype(np.float32)
        world_clouds.append(world);time_clouds.append(np.full(len(world),row['model_time'],np.float32))
        labels=np.array(Image.open(a.mask_root/f'{raw:03d}'/'labels.png'));H,W=labels.shape
        if not set(np.unique(labels))<=({0}|set(registry.actor_to_track)):raise ValueError('Unregistered source label')
        native_w,native_h=obs['camera_size']
        foreground=np.array(Image.open(a.mask_root/f'{raw:03d}'/'dynamic_union.png'))>0
        Image.fromarray(np.uint8(foreground)*255).save(out/'dynamic_masks/dynamic_mask_0'/f'{name}.png')
        Image.fromarray(labels.astype(np.uint8)).resize((native_w,native_h),Image.Resampling.NEAREST).save(out/'mask/mask_0/Annotations'/f'{name}.png')
        rgb=np.array(Image.open(BANK/'image/image_0'/f'{name}.png'));union=np.array(Image.fromarray(np.uint8(foreground)).resize((native_w,native_h),Image.Resampling.NEAREST))>0
        rgb[~union]=0;Image.fromarray(rgb).save(out/'dynamic_image/dynamic_image_0'/f'{name}.png',compress_level=1)
        camera=np.array(obs['camera_pose_world']).reshape(4,4)@np.array(obs['camera_extrinsic']).reshape(4,4);inv=np.linalg.inv(camera);cam=world@inv[:3,:3].T+inv[:3,3];depth=cam[:,0]
        fx,fy,cx,cy,k1,k2,p1,p2,k3=obs['camera_intrinsic_Brown'];x=-cam[:,1]/np.maximum(depth,.1);y=-cam[:,2]/np.maximum(depth,.1);rr=x*x+y*y;rad=1+k1*rr+k2*rr**2+k3*rr**3
        uv_native=np.column_stack((fx*(x*rad+2*p1*x*y+p2*(rr+2*x*x))+cx,fy*(y*rad+p1*(rr+2*y*y)+2*p2*x*y)+cy))
        uv=uv_native*np.array([W/native_w,H/native_h])
        inside=(depth>.1)&(uv[:,0]>=0)&(uv[:,0]<W)&(uv[:,1]>=0)&(uv[:,1]<H)&(abs(x)<1.1)&(abs(y)<1.1)
        source=np.flatnonzero(inside);pixels=uv[source].astype(int);native_pixels=uv_native[source].astype(int);nearest=np.full((native_h,native_w),np.inf);np.minimum.at(nearest,(native_pixels[:,1],native_pixels[:,0]),depth[source]);keep=depth[source]<=nearest[native_pixels[:,1],native_pixels[:,0]]+.3
        assigned=labels[pixels[:,1],pixels[:,0]];source=source[keep];assigned=assigned[keep]
        frame_dynamic=0
        for box in obs['boxes']:
            if box['track'] not in registry.track_to_actor:continue
            actor=registry.track_to_actor[box['track']];ids=source[assigned==actor]
            if not len(ids):continue
            inv=np.linalg.inv(np.array(box['world_from_actor']).reshape(4,4));local=world[ids]@inv[:3,:3].T+inv[:3,3];half=np.array(box['dimensions'])/2
            ids=ids[(abs(local)<=half+.03).all(1)&(local[:,2]>-half[2]+.1)]
            if not len(ids):continue
            point_groups.setdefault(actor,[]).append(world[ids]);time_groups.setdefault(actor,[]).append(np.full(len(ids),row['model_time'],np.float32));frame_dynamic+=len(ids)
        stats.append(dict(raw=raw,lidar=len(world),dynamic=frame_dynamic));print(raw,frame_dynamic,flush=True)
    all_xyz=np.concatenate(world_clouds);all_t=np.concatenate(time_clouds);choice=np.random.RandomState(0).choice(len(all_xyz),600000,replace=True);sample=all_xyz[choice];times=all_t[choice]
    ply(out/'points3d.ply',sample,times)
    mapping=[];dyn=[]
    for actor in sorted(point_groups):
        xyz=np.concatenate(point_groups[actor]);t=np.concatenate(time_groups[actor]);p=out/f'points3d_dynamic_{actor:03d}.ply';ply(p,xyz,t);dyn.append(xyz)
        mapping.append(dict(actor_id=actor,track_id=registry.actor_to_track[actor],rows=len(xyz),ply=str(p)))
    if not dyn:raise ValueError('No measured registered actor points; cannot initialize editable native scene')
    nearest,_=cKDTree(np.concatenate(dyn)).query(sample,workers=1,distance_upper_bound=.1)
    expected_static=int((nearest>=.1).sum())
    write(ROOT/'identity_contract.json',dict(dynamic_actor_ids=sorted(point_groups),source_actor_rows=mapping,source_ply_order=[x['actor_id'] for x in mapping],expected_initial_static_rows=expected_static,expected_initial_dynamic_rows=sum(x['rows'] for x in mapping),native_identity='explicit stable track_to_actor registry',track_to_actor=dict(registry.track_to_actor)))
    cfg=json.loads(a.config_source.read_text());cfg.update(source_path=str(out),model_path=str(ROOT/'formal10k'),start_checkpoint=None,eval=False,start_frame=0,end_frame=selection['source_frame_count']-1,save_iterations=[10000],checkpoint_iterations=[10000],dynamic_actor_ids=sorted(point_groups),delete_actor_ids=[],time_duration=[-(selection['source_frame_count']-1)*.01,(selection['source_frame_count']-1)*.01])
    frames=receipt['frames'];anchor=next(x for x in frames if x['raw']==anchor_raw)
    cfg['followup_scene_spec']=dict(scene_id=selection['scene'],source_path=str(out),frame_ids=[x['id'] for x in frames],camera_directory='image_0',frame_ids_with_camera=['image_0/'+x['id'] for x in frames],model_timestamps=[x['model_time'] for x in frames],raw_frame_range=[0,selection['source_frame_count']-1],original_wh=[native_w,native_h],render_wh=[W,H],anchor_frame=anchor['id'],anchor_index=frames.index(anchor),anchor_timestamp=anchor['model_time'],actor_ply_paths=[x['ply'] for x in mapping])
    cfg['followup_static_anchor_frame']=anchor['id'];write(ROOT/'config.json',cfg)
    assert cfg['start_checkpoint'] is None and cfg['delete_actor_ids']==[]
    assert selection['target']['native_actor_id'] in point_groups
    write(ROOT/'arm_contract.json',dict(formal_stop=10000,learning_rate_horizon=30000,expected_initial_static_rows=expected_static,api_package_root=str(ROOT/'api'),smoke_seconds=600))
    source=ROOT/'source';source.mkdir(exist_ok=True)
    for name in ['train_scene.py','scene_binding.py','evaluate_native.py','render_v1.py']:
        shutil.copy2(a.training_source/name,source/name)
    receipt_path=ROOT/('preparation_receipt_pending_review.json' if a.defer_ready else 'preparation_receipt.json')
    write(receipt_path,dict(status='PASS_GENERIC_SCENE_INPUTS',frames=len(frames),actors=len(mapping),static=expected_static,dynamic=sum(x['rows'] for x in mapping),frame_stats=stats,holdout=receipt['holdout_raws'],point_source='real source frames, depth-visible instance masks and physical OBB',new_mask_preparation='accepted observations with independent occluder refinement; identity unknown foreground remains separate',registry=str(a.registry),native_pixel_depth_filter=True,method='Inherited v44 native GN11, global RGB=1, full dynamic opacity MSE=100, 30K LR horizon; fresh 10K formal'))
    print('NATIVE_INPUTS_READY',expected_static,sum(x['rows'] for x in mapping),len(mapping),flush=True)
if __name__=='__main__':main()

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
    a=ap.parse_args()
    ROOT,PREP,BANK=a.root,a.prep,a.prep/'inputs'
    ENGINE,ORIGINAL,API=a.engine_source,a.config_source.parent,a.api_source
    selection=json.loads((PREP/'selected_case.json').read_text())
    receipt=json.loads((BANK/'receipt.json').read_text())
    assert receipt['status']=='PASS_CPU_OBSERVATIONS'
    assert json.loads((BANK/'verification.json').read_text())['status']=='PASS_CPU_BANK'
    # 分割尚未完成时，先拒绝，避免创建半套三维训练目录。
    for row in receipt['frames']:
        assert (PREP/'masks_all_resolved_r2'/f"{row['raw']:03d}"/'labels.png').is_file()
        assert (BANK/'sky/sky_0'/f"{row['id']}.png").is_file()
    anchor_raw=60
    ROOT.mkdir(exist_ok=True);out=ROOT/'inputs/native';out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(ENGINE,ROOT/'engine',ignore=shutil.ignore_patterns('__pycache__','smoke_*','tests'))
    shutil.copytree(API/'drivingeditor_dynamic_edit',ROOT/'api/drivingeditor_dynamic_edit',ignore=shutil.ignore_patterns('__pycache__'))
    # engine-source 是已验证的通用原生引擎；只复制代码，不复制旧场景输入。
    for name in ['calib','pose','timestamp','velodyne','image','sky']:(out/name).symlink_to(BANK/name,target_is_directory=True)
    shutil.copy2(BANK/'explicit_timeline.json',out/'explicit_timeline.json')
    for name in ['mask/mask_0/Annotations','dynamic_image/dynamic_image_0','dynamic_masks/dynamic_mask_0']:(out/name).mkdir(parents=True)
    receipt=json.loads((BANK/'receipt.json').read_text());actor_meta=json.loads((BANK/'track_registry.json').read_text())['actors'];point_groups={};time_groups={};world_clouds=[];time_clouds=[];stats=[]
    for row in receipt['frames']:
        raw=row['raw'];name=row['id'];obs=json.loads((BANK/'observations'/f'{raw:03d}.json').read_text())
        cloud=np.fromfile(BANK/'velodyne'/f'{name}.bin',dtype='<f4').reshape(-1,6);xyz=cloud[:,:3];ego=np.array(obs['ego_world']).reshape(4,4);world=(xyz@ego[:3,:3].T+ego[:3,3]).astype(np.float32)
        world_clouds.append(world);time_clouds.append(np.full(len(world),row['model_time'],np.float32))
        labels=np.array(Image.open(PREP/'masks_all_resolved_r2'/f'{raw:03d}'/'labels.png'))
        Image.fromarray(np.uint8(labels>0)*255).save(out/'dynamic_masks/dynamic_mask_0'/f'{name}.png')
        Image.fromarray(labels.astype(np.uint8)).resize((1920,1280),Image.Resampling.NEAREST).save(out/'mask/mask_0/Annotations'/f'{name}.png')
        rgb=np.array(Image.open(BANK/'image/image_0'/f'{name}.png'));union=np.array(Image.fromarray(np.uint8(labels>0)).resize((1920,1280),Image.Resampling.NEAREST))>0
        rgb[~union]=0;Image.fromarray(rgb).save(out/'dynamic_image/dynamic_image_0'/f'{name}.png',compress_level=1)
        camera=np.array(obs['camera_pose_world']).reshape(4,4)@np.array(obs['camera_extrinsic']).reshape(4,4);inv=np.linalg.inv(camera);cam=world@inv[:3,:3].T+inv[:3,3];depth=cam[:,0]
        fx,fy,cx,cy,k1,k2,p1,p2,k3=obs['camera_intrinsic_Brown'];x=-cam[:,1]/np.maximum(depth,.1);y=-cam[:,2]/np.maximum(depth,.1);rr=x*x+y*y;rad=1+k1*rr+k2*rr**2+k3*rr**3
        uv=np.column_stack(((fx*(x*rad+2*p1*x*y+p2*(rr+2*x*x))+cx)/4,(fy*(y*rad+p1*(rr+2*y*y)+2*p2*x*y)+cy)/4))
        inside=(depth>.1)&(uv[:,0]>=0)&(uv[:,0]<480)&(uv[:,1]>=0)&(uv[:,1]<320)&(abs(x)<1.1)&(abs(y)<1.1)
        source=np.flatnonzero(inside);pixels=uv[source].astype(int);nearest=np.full((320,480),np.inf);np.minimum.at(nearest,(pixels[:,1],pixels[:,0]),depth[source]);keep=depth[source]<=nearest[pixels[:,1],pixels[:,0]]+.3
        assigned=labels[pixels[:,1],pixels[:,0]];source=source[keep];assigned=assigned[keep]
        frame_dynamic=0
        for box in obs['boxes']:
            alias=actor_meta[box['track']]['alias'];actor=int(alias[1:])+1;ids=source[assigned==actor]
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
        mapping.append(dict(actor_id=actor,alias=f'a{actor-1}',rows=len(xyz),ply=str(p)))
    nearest,_=cKDTree(np.concatenate(dyn)).query(sample,workers=1,distance_upper_bound=.1)
    expected_static=int((nearest>=.1).sum())
    write(ROOT/'identity_contract.json',dict(dynamic_actor_ids=sorted(point_groups),source_actor_rows=mapping,source_ply_order=[x['actor_id'] for x in mapping],expected_initial_static_rows=expected_static,expected_initial_dynamic_rows=sum(x['rows'] for x in mapping),native_identity='alias number + 1'))
    cfg=json.loads(a.config_source.read_text());cfg.update(source_path=str(out),model_path=str(ROOT/'formal10k'),start_checkpoint=None,start_frame=0,end_frame=selection['source_frame_count']-1,save_iterations=[10000],checkpoint_iterations=[10000],dynamic_actor_ids=sorted(point_groups),delete_actor_ids=[],time_duration=[-(selection['source_frame_count']-1)*.01,(selection['source_frame_count']-1)*.01])
    frames=receipt['frames'];anchor=next(x for x in frames if x['raw']==anchor_raw)
    cfg['followup_scene_spec']=dict(scene_id=selection['scene'],source_path=str(out),frame_ids=[x['id'] for x in frames],camera_directory='image_0',frame_ids_with_camera=['image_0/'+x['id'] for x in frames],model_timestamps=[x['model_time'] for x in frames],raw_frame_range=[0,selection['source_frame_count']-1],original_wh=[1920,1280],render_wh=[480,320],anchor_frame=anchor['id'],anchor_index=frames.index(anchor),anchor_timestamp=anchor['model_time'],actor_ply_paths=[x['ply'] for x in mapping])
    cfg['followup_static_anchor_frame']=anchor['id'];write(ROOT/'config.json',cfg)
    assert cfg['start_checkpoint'] is None and cfg['delete_actor_ids']==[]
    assert selection['target']['native_actor_id'] in point_groups
    write(ROOT/'arm_contract.json',dict(formal_stop=10000,learning_rate_horizon=30000,expected_initial_static_rows=expected_static,api_package_root=str(ROOT/'api'),smoke_seconds=600))
    source=ROOT/'source';source.mkdir(exist_ok=True)
    for name in ['train_scene.py','scene_binding.py','evaluate_native.py','render_v1.py']:
        shutil.copy2(Path(__file__).resolve().parent/name,source/name)
    write(ROOT/'preparation_receipt.json',dict(status='PASS_GENERIC_SCENE_INPUTS',frames=len(frames),actors=len(mapping),static=expected_static,dynamic=sum(x['rows'] for x in mapping),frame_stats=stats,holdout=receipt['holdout_raws'],point_source='real source frames, depth-visible instance masks and physical OBB',new_mask_preparation='SAM2 plus same-frame measured prompts and nearest cuboid depth overlap resolution',method='Inherited Full5/v44 native GN11, global RGB=1, full dynamic opacity MSE=100, 30K LR horizon; fresh 10K formal'))
    print('NATIVE_INPUTS_READY',expected_static,sum(x['rows'] for x in mapping),len(mapping),flush=True)
if __name__=='__main__':main()

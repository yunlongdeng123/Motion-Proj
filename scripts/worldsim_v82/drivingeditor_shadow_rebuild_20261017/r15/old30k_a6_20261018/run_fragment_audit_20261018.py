"""Frozen GN11 30k + rigid 30k inspection; no optimizer/parameter update."""
from pathlib import Path
import sys, json, time, math, gc, datetime
import numpy as np
from PIL import Image
import torch

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1')
OUT=ROOT/'old30k_fragment_audit_20261018'
R15=ROOT/'shadow_rebuild_20261017_v3/backend_restore_r15'
CK=ROOT/'rigid_runs/scene0008_089_137/full30k/checkpoint_030000.pt'
sys.path.insert(0,str(ROOT/'infra'))
from runtime import load_models, masks_for_frame
from renderer import render, prepare_instances, _screenspace, _rasterize, _payload, RasterizationSettings
from render_edits import frame_edits, verify_checkpoint_binding
from priority import require_gpu_slot, inspect_priority

def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def png(p,x):
    a=x.detach().float().cpu().numpy() if torch.is_tensor(x) else x
    if a.ndim==3 and a.shape[0] in (1,3):a=a.transpose(1,2,0)
    if a.ndim==3 and a.shape[-1]==1:a=a[...,0]
    Image.fromarray(np.rint(np.clip(a,0,1)*255).astype(np.uint8)).save(p)
def snap(p):
    s=p.stat();return {'path':str(p),'size':s.st_size,'mtime_ns':s.st_mtime_ns}

def old_isolated(camera,instances,config,actor):
    d,w=prepare_instances(camera,instances,config)
    chosen=w.point_ids==actor
    d={k:v[chosen] if torch.is_tensor(v) else v for k,v in d.items()}
    d['means2D']=_screenspace(d['means3D'])
    settings=RasterizationSettings(int(camera.image_height),int(camera.image_width),
        math.tan(-.5 if getattr(config,'neg_fov',False) else camera.FoVx*.5),
        math.tan(-.5 if getattr(config,'neg_fov',False) else camera.FoVy*.5),torch.zeros(3,device='cuda'),
        1.,camera.world_view_transform,camera.full_proj_transform,int(w.active_sh_degree),
        camera.camera_center,False,False)
    from diff_gaussian_rasterization import GaussianRasterizer
    r=GaussianRasterizer(raster_settings=settings)
    result=_rasterize(r,d,_payload(d,camera,d['means3D'].new_ones(int(chosen.sum())),
        w.point_ids[chosen],[]),settings,10)
    return result[1],result[2][1]

def main():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir()
    slot,admission=require_gpu_slot('/root/autodl-tmp/drivingeditor_repro/campaign/gpu_slot.lock')
    start=time.monotonic();torch.set_num_threads(2)
    snapshots=[snap(CK),snap(R15/'paused_checkpoint_004000_20261017/checkpoint.pt')]
    spec=json.loads((ROOT/'rigid_specs/scene0008_089_137.json').read_text())
    manifest=json.loads((R15/'data/manifest.json').read_text())
    current_ids=manifest['actor_ids']
    saved=torch.load(CK,map_location='cuda',weights_only=False)
    step=verify_checkpoint_binding(saved,spec,30000)
    args,scene,static,instances,opt,env,exclusion=load_models(spec,OUT,restore=saved)
    ids=list(instances.instance_ids)
    mapping=[{'old_index':i,'track_id':s,'r15_actor':current_ids.index(s)} for i,s in enumerate(ids)]
    assert [m['r15_actor'] for m in mapping]==[5,26,18]
    bg=torch.zeros(3,device='cuda');old_records=[];new_records=[]
    summary={'schema':'old30k_fragment_frozen_audit_v1','started':datetime.datetime.now().isoformat(),
        'old_checkpoint':snapshots[0],'old_actual_step':step,'old_spec':spec,
        'r15_checkpoint':snapshots[1],'mapping':mapping,'a6_track_id':current_ids[6],
        'a6_old_independent_asset_exists':current_ids[6] in ids,
        'a37_old_independent_asset_exists':current_ids[37] in ids,
        'a6_old_branch':'static_PVG','parameter_updates':0,'admission':admission,
        'background_exclusion':exclusion,'old_records':old_records,'r15_records':new_records,
        'limits':['same49 reconstruction views; no clean edited GT',
                  'isolated raw alpha outside visible body may include valid occluded surfaces',
                  'a6/a37 have no old independent instance; no fictitious OBB asset extraction']}
    seed=np.load(Path(spec['prepared'])/'instances.npz',allow_pickle=False)
    scale=float(seed['world_scale'])
    state=instances.state_dict();point_ids=state['point_ids'];xyz=state['_xyz']
    summary['old_geometry']=[]
    for k in range(len(ids)):
        keep=point_ids==k;half=instances.box_size[:,k].median(0).values/2
        summary['old_geometry'].append({'old_index':k,'r15_actor':current_ids.index(ids[k]),
            'points':int(keep.sum()),'seed_points':int((seed['point_ids']==k).sum()),
            'centers_outside_median_box':int((xyz[keep].abs()>half).any(-1).sum()),
            'box_half_physical_m':(half/scale).tolist(),
            'max_scale_p95_physical_m':float(torch.quantile(state['_scaling'][keep].exp().max(-1).values,.95)/scale)})
    old_dir=OUT/'old';old_dir.mkdir()
    with torch.no_grad():
        for i,camera in enumerate(scene.getTrainCameras()):
            check=inspect_priority()
            if not check['allowed']:raise RuntimeError(check)
            fid=Path(camera.image_name).stem;raw=int(fid[-3:])
            assert fid==manifest['frames'][i]['frame_id']
            folder=old_dir/fid;folder.mkdir()
            full=render(camera,static,instances,args,bg,env_map=env,instance_ids=[],render_separate=True)
            png(folder/'gt.png',camera.original_image);png(folder/'full.png',full['render'])
            png(folder/'static.png',full['render_s'])
            masks,known=masks_for_frame(spec,instances,camera)
            for k,track in enumerate(ids):
                actor=current_ids.index(track);rgb,alpha=old_isolated(camera,instances,args,k)
                png(folder/f'a{actor:02d}_rgb.png',rgb);png(folder/f'a{actor:02d}_alpha.png',alpha)
                png(folder/f'a{actor:02d}_mask.png',masks[k])
                body=masks[k]>0;outside=~body
                old_records.append({'raw':raw,'actor':actor,'body_pixels':int(body.sum()),
                    'alpha_mass_exterior':float(alpha[outside].sum()),
                    'exterior_alpha_pixels_gt_005':int(((alpha>.05)&outside).sum()),
                    'inside_alpha_mean':float(alpha[body].mean()) if body.any() else None})
                if raw in [101,105,113,121,137]:
                    edits,valid=frame_edits(instances,camera.timestamp,track,'translate',
                        translation_m=[0,2,1],scale=scale)
                    moved=render(camera,static,instances,args,bg,env_map=env,instance_ids=[track],
                        edits=edits,render_separate=False)
                    removed=render(camera,static,instances,args,bg,env_map=env,instance_ids=[track],
                        edits={track:{'delete':True}},render_separate=False)
                    assert removed['instance_visible_contribution'].count_nonzero()==0
                    png(folder/f'a{actor:02d}_translate_left2_up1m.png',moved['render'])
                    png(folder/f'a{actor:02d}_delete.png',removed['render'])
            if i%7==0:print(json.dumps({'old_frames':i+1,'seconds':time.monotonic()-start}),flush=True)
    assert snapshots[0]==snap(CK)
    del saved,args,scene,static,instances,opt,env,full,state,seed,xyz,point_ids
    gc.collect();torch.cuda.empty_cache()
    sys.path.insert(0,str(ROOT/'shadow_rebuild_20261017_v3/implementation_r15_train_v3/shadow_rebuild_v3'))
    from shadow_rebuild.config import Config,LossWeights
    from shadow_rebuild.data import Dataset
    from shadow_rebuild.model import SceneModel
    from shadow_rebuild.train import seed_from_state
    from shadow_rebuild.raster import cuda_raster
    from shadow_rebuild.math3d import linear_to_srgb
    saved=torch.load(R15/'paused_checkpoint_004000_20261017/checkpoint.pt',map_location='cpu',weights_only=False)
    raw=saved['config'].copy();raw['weights']=LossWeights(**raw['weights']);cfg=Config(**raw).validate()
    data=Dataset(R15/'data','cuda',True);model=SceneModel(seed_from_state(saved['state']),cfg).cuda()
    model.load_state_dict(saved['state']);model.requires_grad_(False);cfg.shadows=False
    actors=[5,6,18,26,37];new_dir=OUT/'r15';new_dir.mkdir()
    summary['r15_geometry']={str(a):{'points':int((model.field.owner==a).sum()),
        'seed_points':int((data.seed['owner']==a).sum())} for a in actors}
    summary['trajectory_displacement_m']={str(a):float(np.linalg.norm(data.poses[-1,a,:3,3]-data.poses[0,a,:3,3])) for a in actors}
    with torch.no_grad():
        for i in range(len(data.rows)):
            frame=data.frame(i);fid=frame.frame_id;raw=int(fid[-3:]);folder=new_dir/fid;folder.mkdir()
            g,p=model.buffers(frame,len(current_ids),'cuda',isolated=False)
            full=model.train_image(p,g);png(folder/'full_no_shadow.png',linear_to_srgb(full))
            for a in actors:
                selected=g['owner']==a
                feat=torch.cat((torch.ones_like(g['opacity']),g['appearance']),-1)
                r=cuda_raster(frame.camera,g['xyz'][selected],g['cov'][selected],g['opacity'][selected],feat[selected],10)
                alpha=r[...,0];rgb=r[...,1:4]
                png(folder/f'a{a:02d}_rgb.png',linear_to_srgb(rgb));png(folder/f'a{a:02d}_alpha.png',alpha)
                png(folder/f'a{a:02d}_mask.png',frame.masks[...,a].float())
                body=frame.masks[...,a]&frame.mask_valid[...,a]
                outside=(~frame.masks[...,a])&frame.mask_valid[...,a]
                new_records.append({'raw':raw,'actor':a,'body_pixels':int(body.sum()),
                    'alpha_mass_exterior':float(alpha[outside].sum()),
                    'exterior_alpha_pixels_gt_005':int(((alpha>.05)&outside).sum()),
                    'inside_alpha_mean':float(alpha[body].mean()) if body.any() else None})
            if i%7==0:print(json.dumps({'r15_frames':i+1,'seconds':time.monotonic()-start}),flush=True)
    assert snapshots[1]==snap(R15/'paused_checkpoint_004000_20261017/checkpoint.pt')
    summary.update(status='FROZEN_RENDER_COMPLETE_REVIEW_PENDING',frames=49,seconds=time.monotonic()-start,
        checkpoint_files_unchanged=True,original_R15_pause_marker_preserved=(R15/'PAUSE_REQUESTED.json').is_file())
    write(OUT/'audit.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['old_records','r15_records','old_spec']},ensure_ascii=False),flush=True)
    slot.close()

if __name__=='__main__':main()

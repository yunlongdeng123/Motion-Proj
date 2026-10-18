"""按真实身份输出49帧独立RGB/alpha、可见贡献与单车删除；不训练。"""
import argparse,sys,time,math
from identity_common import *

def save_png(path,value,linear=False):
    import torch
    from PIL import Image
    if linear:
        from shadow_rebuild.math3d import linear_to_srgb
        value=linear_to_srgb(value)
    a=value.detach().float().cpu().numpy() if torch.is_tensor(value) else value
    if a.ndim==3 and a.shape[0] in (1,3):a=a.transpose(1,2,0)
    if a.ndim==3 and a.shape[-1]==1:a=a[...,0]
    Image.fromarray(np.rint(np.clip(a,0,1)*255).astype(np.uint8)).save(path)

def isolate_old(camera,instances,config,idx):
    import torch
    from renderer import prepare_instances,_screenspace,_rasterize,_payload,RasterizationSettings
    from diff_gaussian_rasterization import GaussianRasterizer
    d,w=prepare_instances(camera,instances,config);take=w.point_ids==idx
    d={k:v[take] if torch.is_tensor(v) else v for k,v in d.items()}
    d['means2D']=_screenspace(d['means3D'])
    settings=RasterizationSettings(camera.image_height,camera.image_width,
        math.tan(-.5 if config.neg_fov else camera.FoVx*.5),
        math.tan(-.5 if config.neg_fov else camera.FoVy*.5),
        torch.zeros(3,device='cuda'),1.,camera.world_view_transform,camera.full_proj_transform,
        int(w.active_sh_degree),camera.camera_center,False,False)
    r=_rasterize(GaussianRasterizer(raster_settings=settings),d,
        _payload(d,camera,d['means3D'].new_ones(int(take.sum())),w.point_ids[take],[]),settings,10)
    return r[1],r[2][1]

def run_old(args,out):
    import torch
    from runtime import load_models
    from renderer import render
    from render_edits import verify_checkpoint_binding
    historical=args.model=='old3'
    ck=ROOT/'rigid_runs/scene0008_089_137/full30k/checkpoint_030000.pt' if historical else args.checkpoint or TASK/'train5/checkpoint_030000.pt'
    spec=read(ROOT/'rigid_specs/scene0008_089_137.json') if historical else read(TASK/'spec5.json')
    before=stat(ck);saved=torch.load(ck,map_location='cuda',weights_only=False)
    verify_checkpoint_binding(saved,spec,30000)
    config,scene,static,instances,opt,env,exclusion=load_models(spec,out,restore=saved)
    manifest=read(R15/'data/manifest.json');tracks=instances.instance_ids
    actors=OLD_ACTORS if historical else ACTORS
    assert list(tracks)==[manifest['actor_ids'][a] for a in actors]
    records=[];bg=torch.zeros(3,device='cuda')
    with torch.no_grad():
        for i,camera in enumerate(scene.getTrainCameras()):
            fid=Path(camera.image_name).stem;folder=out/fid;folder.mkdir()
            assert fid==manifest['frames'][i]['frame_id']
            full=render(camera,static,instances,config,bg,env_map=env,instance_ids=tracks)
            save_png(folder/'gt.png',camera.original_image);save_png(folder/'full.png',full['render'])
            save_png(folder/'static.png',full['render_s'])
            world=instances.evaluate(camera.timestamp)
            for idx,actor in enumerate(actors):
                track=tracks[idx];rgb,alpha=isolate_old(camera,instances,config,idx)
                deleted=render(camera,static,instances,config,bg,env_map=env,instance_ids=[track],
                    edits={track:{'delete':True}})
                assert deleted['instance_visible_contribution'].count_nonzero()==0
                assert torch.equal(full['render_s'],deleted['render_s'])
                edited_world=instances.evaluate(camera.timestamp,edits={track:{'delete':True}})
                old_keep=world.point_ids!=idx;new_keep=edited_world.point_ids!=idx
                assert not (edited_world.point_ids==idx).any()
                for key in ['xyz','rotation','scaling','features','opacity','row_ids']:
                    assert torch.equal(getattr(world,key)[old_keep],getattr(edited_world,key)[new_keep])
                save_png(folder/f'a{actor:02d}_isolated.png',rgb)
                save_png(folder/f'a{actor:02d}_alpha.png',alpha)
                save_png(folder/f'a{actor:02d}_visible.png',full['instance_visible_contribution'][idx])
                save_png(folder/f'a{actor:02d}_delete.png',deleted['render'])
                np.savez_compressed(folder/f'a{actor:02d}.npz',isolated_rgb=rgb.cpu().numpy(),
                    isolated_alpha=alpha.cpu().numpy(),visible_alpha=full['instance_visible_contribution'][idx].cpu().numpy(),
                    delete_rgb=deleted['render'].cpu().numpy())
                records.append(dict(frame_id=fid,actor=actor,track_id=track,
                    model_native_RGB_domain='sRGB-like original SH training domain',
                    delete_target_visible_alpha_exact_zero=True,other_actor_geometry_bitwise_unchanged=True,
                    isolated_static_RGB_bitwise_unchanged=True,isolated_render_mask_multiplied=False,
                    isolated_points=int((instances.point_ids==idx).sum())))
            if i%7==0:print(json.dumps(dict(model=args.model,frames=i+1)),flush=True)
    assert before==stat(ck)
    return dict(checkpoint=before,checkpoint_step=30000,actors=actors,records=records,
        actual_ID_ownership=True,physical_shadow_module=False,
        historical_3_ID_baseline=historical,new_5_ID_retrain=not historical)

def run_r15(args,out):
    import torch
    sys.path.insert(0,str(ROOT/'shadow_rebuild_20261017_v3/implementation_r15_train_v3/shadow_rebuild_v3'))
    from shadow_rebuild.config import Config,LossWeights
    from shadow_rebuild.data import Dataset
    from shadow_rebuild.model import SceneModel
    from shadow_rebuild.train import seed_from_state
    from shadow_rebuild.raster import render_buffers,cuda_raster
    from packed_shadow import install,factors_for_geometry
    ck=R15/'paused_checkpoint_004000_20261017/checkpoint.pt';before=stat(ck)
    saved=torch.load(ck,map_location='cpu',weights_only=False)
    raw=saved['config'].copy();raw['weights']=LossWeights(**raw['weights']);cfg=Config(**raw).validate()
    data=Dataset(R15/'data','cuda',True);model=SceneModel(seed_from_state(saved['state']),cfg).cuda()
    model.load_state_dict(saved['state']);model.set_stage('geometry');model.requires_grad_(False)
    install(model,dtype=torch.float64,block_size=256,pair_batch=256,recompute=False)
    assert cfg.sun_samples==5
    records=[]
    with torch.no_grad():
        for i in range(len(data.rows)):
            frame=data.frame(i);fid=frame.frame_id;folder=out/fid;folder.mkdir()
            g,p=model.buffers(frame,len(data.actor_ids),'cuda',isolated=False)
            cfg.shadows=False;full_off=model.train_image(p,g)
            cfg.shadows=True;full_on=model.train_image(p,g)
            save_png(folder/'gt.png',frame.rgb,True);save_png(folder/'full_no_shadow.png',full_off,True)
            save_png(folder/'full.png',full_on,True)
            for actor in ACTORS:
                take=g['owner']==actor
                feat=torch.cat((torch.ones_like(g['opacity']),g['appearance']),-1)
                r=cuda_raster(frame.camera,g['xyz'][take],g['cov'][take],g['opacity'][take],feat[take],10)
                keep=~take;dg=type(g)(**{k:v[keep] for k,v in g.items()})
                assert not (dg['owner']==actor).any()
                for key in ['rows','xyz','cov','opacity','appearance']:assert torch.equal(dg[key],g[key][keep])
                # 删除的同一几何同时从主相机和太阳caster集合消失，不能复用父缓存。
                dg.shadow_factors=factors_for_geometry(model,frame,dg)
                assert not (dg.shadow_factors.owners==actor).any()
                dp=render_buffers(frame.camera,dg,len(data.actor_ids),'cuda',cfg.max_features,isolated=False)
                dp['sky']=p['sky'];dp['canonical']=dp['appearance']+(1-dp['alpha'])[...,None]*p['sky']
                assert dp['visible_labels'][...,actor].count_nonzero()==0
                cfg.shadows=False;delete_off=model.train_image(dp,dg)
                cfg.shadows=True;delete_on=model.train_image(dp,dg)
                save_png(folder/f'a{actor:02d}_isolated.png',r[...,1:4],True)
                save_png(folder/f'a{actor:02d}_alpha.png',r[...,0])
                save_png(folder/f'a{actor:02d}_visible.png',p['visible_labels'][...,actor])
                save_png(folder/f'a{actor:02d}_mask.png',frame.masks[...,actor].float())
                save_png(folder/f'a{actor:02d}_delete.png',delete_on,True)
                save_png(folder/f'a{actor:02d}_delete_no_shadow.png',delete_off,True)
                np.savez_compressed(folder/f'a{actor:02d}.npz',isolated_rgb=r[...,1:4].cpu().numpy(),
                    isolated_alpha=r[...,0].cpu().numpy(),visible_alpha=p['visible_labels'][...,actor].cpu().numpy(),
                    delete_rgb=delete_on.cpu().numpy(),delete_no_shadow_rgb=delete_off.cpu().numpy())
                records.append(dict(frame_id=fid,actor=actor,track_id=data.actor_ids[actor],
                    model_native_RGB_domain='linear RGB',delete_target_visible_alpha_exact_zero=True,
                    target_solar_casters_removed=True,other_geometry_bitwise_unchanged=True,
                    isolated_points=int(take.sum()),isolated_render_mask_multiplied=False))
                del dg,dp,r
            if i%7==0:print(json.dumps(dict(model='r15',frames=i+1)),flush=True)
    assert before==stat(ck)
    return dict(checkpoint=before,checkpoint_step=4000,actors=ACTORS,records=records,
        actual_ID_ownership=True,shadow_implementation='validated FP64 factor adapter; original weights unchanged',
        physical_shadow_module=True,sun_samples=5)

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['r15','old3','old5'],required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path);args=p.parse_args()
    if args.model!='old5' and args.checkpoint:raise ValueError('现有两版使用固定检查点')
    if args.model=='old5' and args.checkpoint and TASK.resolve() not in args.checkpoint.resolve().parents:
        raise ValueError('新增五实例checkpoint必须属于本任务')
    out=args.output or TASK/('renders_'+args.model)
    if out.exists():raise FileExistsError(out)
    sys.path.insert(0,str(TASK/'infra'))
    from priority import require_gpu_slot
    slot,admission=require_gpu_slot(str(ROOT.parent/'gpu_slot.lock'))
    import torch
    torch.set_num_threads(2);out.mkdir(parents=True)
    start=time.monotonic()
    try:
        result=run_r15(args,out) if args.model=='r15' else run_old(args,out)
        result.update(task_id=TASK_ID,status='COMPLETE_REVIEW_PENDING',model=args.model,
            frames=49,parameter_updates=0,checkpoint_file_unchanged=True,seconds=time.monotonic()-start,
            admission=admission,human_verdict=None,paired_edited_GT_available=False)
        write(out/'render_receipt.json',result)
    except BaseException as e:
        write(out/'failure.json',dict(error=repr(e),assets_preserved=True));raise
    finally:slot.close()

if __name__=='__main__':main()

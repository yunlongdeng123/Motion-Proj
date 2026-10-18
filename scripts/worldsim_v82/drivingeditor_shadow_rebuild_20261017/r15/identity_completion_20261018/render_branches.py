"""Read-only full49 branch RGB/alpha before and after each real owner deletion."""
import argparse,sys,time
from identity_common import *
Q=R15/'repair_4k_20261018'

def run_old5(out):
    import torch
    from runtime import load_models
    from renderer import render
    from render_edits import verify_checkpoint_binding
    from render_identity_matrix import save_png
    from PIL import Image
    ck=TASK/'train5/checkpoint_030000.pt';before=stat(ck);saved=torch.load(ck,map_location='cuda',weights_only=False)
    spec=read(TASK/'spec5.json');verify_checkpoint_binding(saved,spec,30000)
    cfg,scene,static,instances,opt,env,exclusion=load_models(spec,out,restore=saved)
    tracks=instances.instance_ids;records=[];bg=torch.zeros(3,device='cuda')
    with torch.no_grad():
        for i,cam in enumerate(scene.getTrainCameras()):
            fid=Path(cam.image_name).stem;folder=out/fid;folder.mkdir()
            full=render(cam,static,instances,cfg,bg,env_map=env,instance_ids=tracks)
            old=np.asarray(Image.open(TASK/'renders_old5'/fid/'full.png'))
            now=np.rint(full['render'].permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8)
            assert np.array_equal(old,now),'branch replay differs from delivered full frame: '+fid
            save_png(folder/'dynamic.png',full['render_d']);save_png(folder/'dynamic_alpha.png',full['dynamic_alpha'])
            save_png(folder/'static.png',full['render_s']);save_png(folder/'static_alpha.png',full['static_alpha'])
            for idx,actor in enumerate(ACTORS):
                deleted=render(cam,static,instances,cfg,bg,env_map=env,instance_ids=[tracks[idx]],edits={tracks[idx]:{'delete':True}})
                assert deleted['instance_visible_contribution'].count_nonzero()==0
                assert torch.equal(deleted['render_s'],full['render_s'])
                save_png(folder/f'a{actor:02d}_dynamic_delete.png',deleted['render_d'])
                save_png(folder/f'a{actor:02d}_dynamic_delete_alpha.png',deleted['dynamic_alpha'])
                records.append(dict(frame_id=fid,actor=actor,target_visible_alpha_exact_zero=True,
                    static_before_after_bitwise_equal=True,static_after_path='static.png',full_replay_matches_existing_PNG=True))
            write(out/'progress.json',dict(model='old5',frames=i+1,total=49));print(json.dumps(dict(model='old5',frames=i+1)),flush=True)
    assert before==stat(ck)
    return dict(checkpoint=before,records=records,model='old5',static_shadow=False,
                branch_semantics='native isolated all-Actor and static/PVG RGB; black dynamic background; no mask multiplication')

def run_r15(model_name,out):
    import torch
    from render_identity_matrix import save_png
    repaired=model_name=='repair4k'
    source=Q/'implementation' if repaired else ROOT/'shadow_rebuild_20261017_v3/implementation_r15_train_v3/shadow_rebuild_v3'
    sys.path.insert(0,str(source))
    from shadow_rebuild.config import Config,LossWeights
    from shadow_rebuild.data import Dataset
    from shadow_rebuild.model import SceneModel
    from shadow_rebuild.train import seed_from_state
    from shadow_rebuild.raster import render_buffers,cuda_raster
    if repaired:from shadow_rebuild.factor_shadow import factors_for_geometry
    else:from packed_shadow import install,factors_for_geometry
    ck=Q/'train/checkpoint_004000.pt' if repaired else R15/'paused_checkpoint_004000_20261017/checkpoint.pt'
    before=stat(ck);saved=torch.load(ck,map_location='cpu',weights_only=False);assert saved['global_step']==4000
    raw=saved['config'].copy();raw['weights']=LossWeights(**raw['weights']);cfg=Config(**raw).validate()
    data=Dataset(R15/'data','cuda',True);m=SceneModel(seed_from_state(saved['state']),cfg).cuda();m.load_state_dict(saved['state'])
    m.set_stage('geometry');m.requires_grad_(False)
    if not repaired:install(m,recompute=False)
    records=[]
    def dynamic(frame,g):
        take=g['owner']>=0;feature=torch.cat((torch.ones_like(g['opacity']),g['appearance']),-1)
        return cuda_raster(frame.camera,g['xyz'][take],g['cov'][take],g['opacity'][take],feature[take],10)
    with torch.no_grad():
        for i in range(49):
            f=data.frame(i);folder=out/f.frame_id;folder.mkdir()
            # Static-only camera receivers; all real Actors remain solar casters.
            # Receiver geometry behind the car is NOT taken from an occluded full-camera G-buffer.
            g,_=m.buffers(f,len(data.actor_ids),'cuda',isolated=False)
            st=g['owner']<0;sg=type(g)(**{k:v[st] for k,v in g.items()})
            p=render_buffers(f.camera,sg,0,'cuda',cfg.max_features,isolated=False);p['sky']=m.sky(f.camera)
            cfg.shadows=False;static_off=m.train_image(p,g)
            cfg.shadows=True;static_on=m.train_image(p,g)
            d=dynamic(f,g)
            save_png(folder/'dynamic.png',d[...,1:4],True);save_png(folder/'dynamic_alpha.png',d[...,0])
            save_png(folder/'static.png',static_off,True);save_png(folder/'static_shadow.png',static_on,True)
            save_png(folder/'static_alpha.png',p['alpha'])
            for actor in ACTORS:
                keep=g['owner']!=actor;dg=type(g)(**{k:v[keep] for k,v in g.items()})
                assert not (dg['owner']==actor).any()
                for key in ['xyz','cov','opacity','appearance','rows']:
                    assert torch.equal(dg[key][dg['owner']<0],g[key][st])
                dg.shadow_factors=factors_for_geometry(m,f,dg)
                assert not (dg.shadow_factors.owners==actor).any()
                dd=dynamic(f,dg);cfg.shadows=True;deleted_static=m.train_image(p,dg)
                save_png(folder/f'a{actor:02d}_dynamic_delete.png',dd[...,1:4],True)
                save_png(folder/f'a{actor:02d}_dynamic_delete_alpha.png',dd[...,0])
                save_png(folder/f'a{actor:02d}_static_delete_shadow.png',deleted_static,True)
                records.append(dict(frame_id=f.frame_id,actor=actor,target_camera_and_solar_geometry_removed=True,
                    static_geometry_appearance_bitwise_unchanged=True,static_no_shadow_before_after_path='static.png',
                    static_with_shadow_after_path=f'a{actor:02d}_static_delete_shadow.png'))
            write(out/'progress.json',dict(model=model_name,frames=i+1,total=49));print(json.dumps(dict(model=model_name,frames=i+1)),flush=True)
    assert before==stat(ck)
    return dict(checkpoint=before,records=records,model=model_name,static_shadow=True,
                branch_semantics='actual isolated all-Actor RGB; static-only camera receivers and original full solar casters; no-shadow static is invariant under Actor deletion')

def execute(model_name,already_locked=False):
    sys.path.insert(0,str(TASK/'infra'));out=TASK/('branches_'+model_name)
    if (out/'branch_receipt.json').is_file():return read(out/'branch_receipt.json')
    if out.exists():raise FileExistsError('partial branch output needs explicit diagnosis: '+str(out))
    out.mkdir();slot=None;start=time.monotonic()
    try:
        if not already_locked:
            from priority import require_gpu_slot
            slot,admission=require_gpu_slot(str(ROOT.parent/'gpu_slot.lock'))
        import torch
        torch.set_num_threads(2)
        result=run_old5(out) if model_name=='old5' else run_r15(model_name,out)
        result.update(status='COMPLETE_REVIEW_PENDING',frames=49,actors=ACTORS,parameter_updates=0,
            checkpoint_file_unchanged=True,seconds=time.monotonic()-start,human_verdict=None)
        write(out/'branch_receipt.json',result);return result
    except BaseException as e:
        write(out/'failure.json',dict(error=repr(e),assets_preserved=True));raise
    finally:
        if slot:slot.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=['old5','r15','repair4k']);a=p.parse_args()
    execute(a.model)

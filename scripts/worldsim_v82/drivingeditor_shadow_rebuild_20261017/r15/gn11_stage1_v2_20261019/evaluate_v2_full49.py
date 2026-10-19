"""Only after actual v2 complete30K: strict restore and zero-update full-frame49."""
from pathlib import Path
import argparse,copy,datetime,fcntl,hashlib,json,os,sys,subprocess


def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False))


def main(campaign):
    C=campaign;run=C/'run30k';engine=C/'engine';out=C/'full49'
    status=read(run/'status.json')
    if status['status']!='complete' or status['iteration']!=30000:raise RuntimeError('v2 training not complete')
    if status['optimizer_step_counts']['static']!=30000 or status['optimizer_step_counts']['dynamic']!=30000:
        raise RuntimeError('actual update counts not30K')
    if out.exists():raise FileExistsError('preserve existing complete/partial evaluation before retry')
    lock=(C.parent/'gpu_slot.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    if apps:raise RuntimeError('GPU already occupied: '+apps)
    manifest=read(C/'source_bundle/candidate_manifest.json')
    for rel,item in manifest['files'].items():
        if hashlib.sha256((engine/rel).read_bytes()).hexdigest()!=item['candidate_sha256']:raise RuntimeError('active source differs: '+rel)
    sys.path.insert(0,str(engine));os.chdir(engine)
    import torch,numpy as np,random
    from PIL import Image
    from omegaconf import OmegaConf
    from scene import Scene,GaussianModel,Dynamic_GaussianModel,EnvLight
    from scene.paper_sgcm import create_sgcm_controller
    from gaussian_renderer import render
    from utils.hybrid30k_runtime import CameraSampler
    from utils.paper_checkpoint import restore_training_state,capture_training_state
    cfg=read(run/'effective_config.json');cfg['model_path']=str(out)
    out.mkdir();args=OmegaConf.create(cfg)
    s=GaussianModel(args);d=Dynamic_GaussianModel(args);scene=Scene(args,s,d,shuffle=False)
    e=EnvLight(resolution=args.env_map_res).cuda();models=dict(static=s,dynamic=d,environment=e)
    names=dict(static='chkpnt30000.pth',dynamic='dynamic_chkpnt30000.pth',environment='env_light_chkpnt30000.pth')
    originals={n:dict(bytes=(run/n).stat().st_size,mtime_ns=(run/n).stat().st_mtime_ns) for n in [*names.values(),'training_state30000.pth']}
    def equal(a,b):
        if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
        if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
        if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return type(a)==type(b) and a==b
    for name,model in models.items():
        state,step=torch.load(run/names[name],map_location='cuda',weights_only=False)
        if step!=30000:raise RuntimeError('wrong model step')
        model.restore(state,args)
        if not equal(state,model.capture()):raise RuntimeError('actual model/Adam restore differs: '+name)
        del state
    scene.hybrid_sampler=CameraSampler(args.seed,scene.getTrainCameras())
    sgcm=create_sgcm_controller(args.source_path,args,training_views=scene.getTrainCameras(),gaussians=s)
    stack,ema=restore_training_state(run/'training_state30000.pth',30000,scene,models,args,sgcm=sgcm)
    expected=torch.load(run/'training_state30000.pth',map_location='cpu',weights_only=False)
    actual=capture_training_state(30000,scene,stack,models,args,sgcm=sgcm,ema=ema)
    expected['config']['model_path']=str(out)
    if not equal(expected,actual):raise RuntimeError('actual RNG/sampler/SGCM continuation restore differs')
    s.enforce_static()
    def forbidden(*a,**k):raise RuntimeError('evaluation attempted optimizer update')
    for m in models.values():m.optimizer.step=forbidden
    write(out/'restore_receipt.json',dict(status='PASSED_ACTUAL_30K_MODEL_ADAM_RNG_SAMPLER_SGCM_RESTORE',
        source_run=str(run),parameter_updates=0,endpoint_training_continuation_authorized=False))
    def png(path,t):
        value=t.detach().clamp(0,1).cpu();value=value.permute(1,2,0) if value.shape[0]==3 else value[0]
        Image.fromarray(np.rint(value.numpy()*255).astype(np.uint8)).save(path)
    staticargs=copy.deepcopy(args);staticargs.including_dynamic=False
    bg=torch.zeros(3,device='cuda');rows=[]
    with torch.no_grad():
        cameras=sorted(scene.getTrainCameras(),key=lambda c:Path(c.image_name).stem)
        if {Path(c.image_name).stem for c in cameras}!={f'0008{i:03d}' for i in range(89,138)}:raise RuntimeError('wrong49 camera set')
        for cam in cameras:
            fid=Path(cam.image_name).stem;folder=out/fid;folder.mkdir()
            other=[s.get_scaling_t.clamp_max(2),s.get_inst_velocity]
            dynamic=[d.get_scaling_t.clamp_max(2),d.get_inst_velocity]
            full=render(cam,s,d,args,bg,env_map=e,other=other,dynamic_other=dynamic,is_training=False)
            removed=render(cam,s,None,staticargs,bg,env_map=e,other=other,is_training=False)
            difference=float((full['render_s']-removed['render']).abs().max())
            if difference>2e-4:raise RuntimeError('delete-all changes independent static layer')
            for name,key in [('full','render'),('dynamic','render_d'),('static','render_s'),('dynamic_alpha','dynamic_alpha'),('static_alpha','static_alpha')]:png(folder/f'{name}.png',full[key])
            png(folder/'gt.png',cam.original_image);png(folder/'delete_all.png',removed['render'])
            rows.append(dict(frame_id=fid,static_replay_max_abs=difference,removed_dynamic_points=0))
            write(out/'progress.json',dict(frames=len(rows),total=49,parameter_updates=0))
        fixed=copy.copy(cameras[0]);pictures=[]
        for t in [float(cameras[0].timestamp),float(cameras[-1].timestamp)]:
            fixed.timestamp=t
            pkg=render(fixed,s,None,staticargs,bg,env_map=e,other=other,is_training=False)
            pictures.append(pkg['render'].clone())
        delta=float((pictures[0]-pictures[1]).abs().max())
        if delta>1e-6:raise RuntimeError('final fixed-camera static time invariance failed')
    for n,info in originals.items():
        if info!=dict(bytes=(run/n).stat().st_size,mtime_ns=(run/n).stat().st_mtime_ns):raise RuntimeError('source checkpoint metadata changed')
    write(out/'receipt.json',dict(state='COMPLETE_V2_NATIVE_FIVE_ID_FULL49_REQUIRES_VISUAL_REVIEW',
        frames=49,records=rows,parameter_updates=0,all_full_frame=True,ROI=False,image_wh=[480,320],
        channels=['gt','full','dynamic','static','delete_all','dynamic_alpha','static_alpha'],
        individual_actor_assets=False,dynamic_scope=[6,37,5,26,18],all_inputs_are_training_views=True,
        fixed_camera_static_time_max_RGB_delta=delta,source_checkpoints_unchanged=True,
        source_components=originals,quality_accepted=False,human_verdict=None,
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(json.dumps(dict(status='COMPLETE_ZERO_UPDATE_V2_FULL49',frames=49,PNGs=343,static_time_delta=delta)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);a=p.parse_args();main(a.campaign)

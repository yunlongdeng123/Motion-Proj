import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import argparse,json,sys,time
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent.parent
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--iteration',type=int,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--all-build',action='store_true');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((a.model/'effective_config.json').read_text());sys.path[:0]=[str(ROOT/'engine'),str(ROOT/'api')]
    import torch
    from omegaconf import OmegaConf
    from scene import GaussianModel,Dynamic_GaussianModel,EnvLight
    from scene.cameras import Camera
    import gaussian_renderer
    from drivingeditor_dynamic_edit import install_identity_serialization,actor_ids,render_scene
    install_identity_serialization();torch.set_num_threads(1);args=OmegaConf.create(cfg)
    models=[GaussianModel(args),Dynamic_GaussianModel(args),EnvLight(resolution=args.env_map_res).cuda()]
    for model,prefix in zip(models,['chkpnt','dynamic_chkpnt','env_light_chkpnt']):
        payload,step=torch.load(str(a.model/f'{prefix}{a.iteration}.pth'),map_location='cpu',weights_only=False,mmap=True);assert step==a.iteration;payload=list(payload)
        if prefix=='env_light_chkpnt':payload[0]=torch.nn.Parameter(payload[0].detach().cuda(),requires_grad=False)
        else:
            for k in range(1,11):payload[k]=payload[k].detach().cuda()
            model.time_duration=tuple(cfg['time_duration'])
        model.restore(tuple(payload),training_args=None)
        assert getattr(model,'optimizer',None) is None
    static,dynamic,env=models;owners=actor_ids(dynamic);declared=list(dynamic._dynamic_actor_ids_declared);target=70
    assert target in declared
    cameras=json.loads((a.model/'cameras.json').read_text());headers={int(c['img_name'][-3:]):c for c in cameras}
    spec=cfg['followup_scene_spec'];times={int(name[-3:]):t for name,t in zip(spec['frame_ids'],spec['model_timestamps'])}
    raws=sorted(headers) if a.all_build else [130,155,160,190]
    bg=torch.tensor([0.,0.,0.],device='cuda');rows=[];start=time.monotonic()
    tag_s=torch.zeros((len(static.get_xyz),1),device='cuda');tag_d=(owners==target).float()[:,None]
    for raw in raws:
        c=headers[raw];pose=np.eye(4);pose[:3,:3]=c['rotation'];pose[:3,3]=c['position'];w=np.linalg.inv(pose);sx=c['width']/480;sy=c['height']/320
        camera=Camera(raw,w[:3,:3].T.copy(),w[:3,3].copy(),cx=c['cx']/sx,cy=c['cy']/sy,fx=c['fx']/sx,fy=c['fy']/sy,image=torch.zeros((3,320,480)),image_name=c['img_name'],resolution=(480,320),timestamp=times[raw])
        folder=a.output/f'{raw:03d}';folder.mkdir();images={};feature={}
        for name,delete in [('keep',[]),('delete',[target]),('keep_again',[])]:
            with torch.no_grad():pack=render_scene(camera,static,dynamic,args,bg,dynamic_actor_ids=declared,delete_actor_ids=delete,renderer=gaussian_renderer,env_map=env,other=[tag_s],dynamic_other=[tag_d],is_training=False)
            rgb=pack['render'].detach().cpu().numpy();assert np.isfinite(rgb).all();images[name]=rgb
            feature[name]=float(pack['feature'][0].abs().max())
            if name!='keep_again':
                Image.fromarray(np.uint8(np.clip(rgb.transpose(1,2,0),0,1)*255+.5)).save(folder/f'{name}.png')
                np.savez_compressed(folder/f'{name}.npz',rgb=rgb,depth_sum=pack['depth'].detach().cpu().numpy(),alpha=pack['alpha'].detach().cpu().numpy(),target_joint_alpha=pack['feature'][0].detach().cpu().numpy())
            del pack
        assert np.array_equal(images['keep'],images['keep_again'])
        assert feature['delete']<1e-6,feature
        image=Image.open(Path(cfg['source_path'])/'image/image_0'/f'{c["img_name"]}.png').resize((480,320),Image.Resampling.LANCZOS);image.save(folder/'observed.png');gt=np.array(image)/255
        mask=np.array(Image.open(Path(cfg['source_path'])/'mask/mask_0/Annotations'/f'{c["img_name"]}.png').resize((480,320),Image.Resampling.NEAREST))
        delta=abs(images['keep']-images['delete']).max(0);row=dict(raw=raw,keep_mae=float(abs(images['keep'].transpose(1,2,0)-gt).mean()),suv_keep_mae=float(abs(images['keep'].transpose(1,2,0)-gt)[mask==25].mean()) if (mask==25).any() else None,target_keep_alpha_max=feature['keep'],target_delete_alpha_max=feature['delete'],changed_pixels=int((delta>1/255).sum()),keep_again_maxabs=float(abs(images['keep']-images['keep_again']).max()))
        rows.append(row);print(json.dumps(row),flush=True)
    write(a.output/'summary.json',dict(status='PASS_READ_ONLY_KEEP_DELETE_KEEP_ENGINEERING',model=str(a.model),iteration=a.iteration,raws=raws,rows=rows,dynamic_actor_ids=declared,delete_actor_ids=[target],optimizer_created=False,updates=0,seconds=time.monotonic()-start,quality_accepted=None))
if __name__=='__main__':main()

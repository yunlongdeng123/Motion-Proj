"""重新载入单 a37 检查点，原生合成/动态图/删除结果及身份泄漏核验。"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import argparse,json,sys,gc,time
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parent.parent
ENGINE=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018/training_source/scene0008_089_137')
OLD=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/baseline_engineering_20261018/full5_pipeline/inputs/scene0008_089_137')
sys.path.insert(0,str(ENGINE))
from omegaconf import OmegaConf
from scene import GaussianModel,Dynamic_GaussianModel,EnvLight
from scene.cameras import Camera
from gaussian_renderer import render

def cpu(x):return x.detach().cpu().numpy()
def rgb8(x):return np.uint8(np.floor(np.clip(cpu(x).transpose(1,2,0),0,1)*255+.5))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def mask(path):return np.asarray(Image.open(path).convert('L').resize((480,320),Image.Resampling.NEAREST))>0

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--iteration',type=int,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--quick',action='store_true');opt=ap.parse_args()
    start=time.monotonic();out=opt.output;out.mkdir(parents=True,exist_ok=False)
    config=json.loads((opt.model/'effective_config.json').read_text());args=OmegaConf.create(config)
    static=GaussianModel(args);dynamic=Dynamic_GaussianModel(args);env=EnvLight(resolution=args.env_map_res).cuda()
    torch.set_num_threads(1)
    for prefix,model in [('chkpnt',static),('dynamic_chkpnt',dynamic),('env_light_chkpnt',env)]:
        payload,step=torch.load(str(opt.model/f'{prefix}{opt.iteration}.pth'),map_location='cpu',weights_only=False,mmap=True)
        assert step==opt.iteration;payload=list(payload)
        if prefix=='env_light_chkpnt':payload[0]=torch.nn.Parameter(payload[0].detach().cuda(),requires_grad=False)
        else:
            for i in range(1,11):payload[i]=payload[i].detach().cuda()
            model.time_duration=tuple(config['time_duration'])
        model.restore(tuple(payload),training_args=None)
        for v in vars(model).values():
            if torch.is_tensor(v):v.requires_grad_(False)
        del payload;gc.collect()
    event=json.loads((opt.model/f'checkpoint_events/iteration_{opt.iteration:06d}.json').read_text())
    cameras=json.loads((opt.model/'cameras.json').read_text());stamps=json.loads((opt.model/'actual_scene_binding.json').read_text())['timestamps']
    bg=torch.tensor([0.,0.,0.],device='cuda');rows=[];max_replay=0;alias=[5,26,18,6]
    rawlist=[89,105,137] if opt.quick else list(range(89,138))
    for raw in rawlist:
        fid=f'0008{raw:03d}';r=next(r for r in cameras if r['img_name']==fid)
        c2w=np.eye(4);c2w[:3,:3]=r['rotation'];c2w[:3,3]=r['position'];W=np.linalg.inv(c2w)
        cam=Camera(raw,W[:3,:3].T.copy(),W[:3,3].copy(),cx=r['cx']/4,cy=r['cy']/4,fx=r['fx']/4,fy=r['fy']/4,
            image=torch.zeros((3,320,480)),image_name=fid,resolution=(480,320),timestamp=stamps[raw-89])
        with torch.no_grad():pkg=render(cam,static,dynamic,args,bg,env_map=env,is_training=False)
        images={name:rgb8(pkg[key]) for name,key in [('composite','render'),('delete_a37','render_s'),('dynamic_a37','render_d')]}
        gt=np.asarray(Image.open(Path(config['source_path'])/'image/image_0'/f'{fid}.png').convert('RGB').resize((480,320),Image.Resampling.BILINEAR))
        images['gt']=gt;directory=out/f'{raw:03d}';directory.mkdir()
        for name,img in images.items():Image.fromarray(img).save(directory/(name+'.png'))
        a=cpu(pkg['dynamic_alpha'])[0];Image.fromarray(np.uint8(np.clip(a,0,1)*255+.5)).save(directory/'dynamic_alpha.png')
        target=mask(Path(config['source_path'])/'dynamic_masks/dynamic_mask_0'/f'{fid}.png')
        outside=~target
        frame=dict(raw=raw,target_pixels=int(target.sum()),outside_pixels=int(outside.sum()),
            alpha_inside_sum=float(a[target].sum()),alpha_outside_sum=float(a[outside].sum()),
            alpha_inside_mean=float(a[target].mean()),alpha_outside_mean=float(a[outside].mean()),
            alpha_outside_p99=float(np.quantile(a[outside],.99)),outside_alpha_over_half_pixels=int((a[outside]>.5).sum()),retained={})
        for i,actor in enumerate(alias):
            keep=mask(OLD/f'dynamic_actor_assets/actor_{i}/body_masks/{fid}.png')
            diff=np.abs(cpu(pkg['render_s']-pkg['render'])).mean(0)
            error=np.abs(images['delete_a37'].astype(float)-gt.astype(float)).mean(2)/255
            frame['retained'][str(actor)]=dict(pixels=int(keep.sum()),dynamic_alpha_sum=float(a[keep].sum()),
                dynamic_alpha_mean=float(a[keep].mean()) if keep.any() else None,
                deletion_MAE_sum=float(diff[keep].sum()),deletion_MAE=float(diff[keep].mean()) if keep.any() else None,
                delete_vs_observed_RGB_MAE_sum=float(error[keep].sum()))
        source_frame=next(f for f in event['frames'] if f['frame_id'].endswith(fid))
        frame['native_png_replay_max_uint8_difference']={}
        for name,key in [('composite','composite'),('delete_a37','static'),('dynamic_a37','dynamic')]:
            ref=np.asarray(Image.open(source_frame['channels'][key]).convert('RGB'))
            d=int(np.abs(images[name].astype(np.int16)-ref.astype(np.int16)).max())
            frame['native_png_replay_max_uint8_difference'][name]=d;max_replay=max(max_replay,d)
        rows.append(frame)
    weighted={}
    for actor in alias:
        count=sum(f['retained'][str(actor)]['pixels'] for f in rows)
        weighted[str(actor)]=dict(pixels=count,dynamic_alpha_mean=sum(f['retained'][str(actor)]['dynamic_alpha_sum'] for f in rows)/max(count,1),
            deletion_MAE=sum(f['retained'][str(actor)]['deletion_MAE_sum'] for f in rows)/max(count,1),
            delete_vs_observed_RGB_MAE=sum(f['retained'][str(actor)]['delete_vs_observed_RGB_MAE_sum'] for f in rows)/max(count,1))
    total_in=sum(f['target_pixels'] for f in rows);total_out=sum(f['outside_pixels'] for f in rows)
    result=dict(status='PASS_NATIVE_LAYER_RELOAD',iteration=opt.iteration,frames=rows,frame_count=len(rows),updates=0,
        dynamic_input_aliases=[37],retained_aliases=alias,static_rows=len(static.get_xyz),dynamic_rows=len(dynamic.get_xyz),
        identity_binding='single canonical a37 PLY/mask before initialization; no posthoc ownership partition',
        alpha_inside_mean=sum(f['alpha_inside_sum'] for f in rows)/total_in,
        alpha_outside_mean=sum(f['alpha_outside_sum'] for f in rows)/total_out,retained_weighted=weighted,
        native_png_replay_max_uint8_difference=max_replay,seconds=time.monotonic()-start,
        road_repair_updates=0,quality_accepted=False,human_verdict=None,evaluation_domain='49observedtrain BUILD; no independent test',
        native_engine=str(ENGINE),torch_version=torch.__version__,cuda_runtime=torch.version.cuda,
        caveat='Only a37 is a dynamic identity input; boundary/opacity leakage is measured, not assumed zero.')
    write(out/'summary.json',result)
    assert max_replay<=1,('native evaluation replay mismatch',max_replay)
    print(json.dumps({k:v for k,v in result.items() if k!='frames'},ensure_ascii=False),flush=True)

if __name__=='__main__':main()

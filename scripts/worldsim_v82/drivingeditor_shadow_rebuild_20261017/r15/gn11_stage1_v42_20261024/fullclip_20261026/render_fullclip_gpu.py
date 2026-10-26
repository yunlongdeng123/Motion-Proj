"""零更新GPU验收：固定V41-B；新道路只加入编辑合成，不改变主通路。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,gc,threading,subprocess,argparse
from pathlib import Path
import numpy as np
from PIL import Image
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v41_20261023')
B=C/'arms/B/run30k'
sys.path.insert(0,str(C/'engine'))

def run(out,gpu_dir='gpu'):
    dest=out/gpu_dir;dest.mkdir(exist_ok=False);stop=threading.Event();peak=[0];errors=[]
    def watch():
        with (dest/'nvml_samples.jsonl').open('w') as f:
            while not stop.is_set():
                try:
                    value=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=5)
                    used=int(value.strip().splitlines()[0]);peak[0]=max(peak[0],used);f.write(json.dumps(dict(time=time.time(),device_memory_used_mib=used))+'\n');f.flush()
                except Exception as e:errors.append(str(e))
                stop.wait(1)
    observer=threading.Thread(target=watch,daemon=True);observer.start();start=time.time()
    import torch
    from omegaconf import OmegaConf
    from scene import GaussianModel,Dynamic_GaussianModel,EnvLight
    from scene.cameras import Camera
    from utils.v4_road_render import render_composite,render_road
    torch.set_num_threads(1)
    config=json.loads((B/'effective_config.json').read_text());args=OmegaConf.create(config)
    files=[B/f'{prefix}30000.pth' for prefix in ['chkpnt','dynamic_chkpnt','env_light_chkpnt','training_state']]
    before={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in files}
    models=[GaussianModel(args),Dynamic_GaussianModel(args),EnvLight(resolution=args.env_map_res).cuda()]
    for p,model in zip(files,models):
        payload,step=torch.load(str(p),map_location='cpu',weights_only=False,mmap=True);assert step==30000;payload=list(payload)
        if 'env_light' in p.name:payload[0]=torch.nn.Parameter(payload[0].detach().cuda(),requires_grad=False)
        else:
            for i in range(1,11):payload[i]=payload[i].detach().cuda()
            model.time_duration=tuple(config['time_duration'])
        model.restore(tuple(payload),training_args=None)
        assert getattr(model,'optimizer',None) is None
        for value in vars(model).values():
            if torch.is_tensor(value):value.requires_grad_(False)
        del payload;gc.collect()
    bundle=json.loads(Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json').read_text())
    A=np.asarray(bundle['world_to_model']);scale=bundle['meter_to_model']
    class Road:
        def __init__(self,a):
            ids=np.flatnonzero(a['surface_supported']&a['eligible']);tensor=lambda x:torch.tensor(x,dtype=torch.float32,device='cuda')
            self.xyz=tensor(a['model_xyz'][ids]);self.scaling=tensor(a['gaussian_sigma_m'][ids]*scale)
            n=a['normal_world'][ids]@A[:3,:3].T/scale;n/=np.linalg.norm(n,axis=1)[:,None]
            q=np.c_[1+n[:,2],-n[:,1],n[:,0],np.zeros(len(n))];q/=np.linalg.norm(q,axis=1)[:,None];self.rotation=tensor(q)
            self.dc=tensor((a['rgb'][ids]-.5)/.28209479177387814)[:,None];self.opacity=tensor(np.zeros((len(ids),1)))
            self.active=torch.ones(len(ids),dtype=torch.bool,device='cuda')
        def output_mask(self):return self.active
        def colors(self):return .5+.28209479177387814*self.dc[:,0]
    roads={n:Road(dict(np.load(out/f'{n}_initialization.npz'))) for n in ['window49','full199']}
    cameras=json.loads((B/'cameras.json').read_text());stamps=json.loads((B/'actual_scene_binding.json').read_text())['timestamps']
    static,dynamic,environment=models;bg=torch.tensor([1.,1.,1.] if args.white_background else [0.,0.,0.],device='cuda')
    def cpu(t):return t.detach().cpu().numpy()
    def save(raw,name,t):Image.fromarray(np.uint8(np.round(np.clip(cpu(t).transpose(1,2,0),0,1)*255))).save(dest/f'{raw}_{name}.png')
    rows=[]
    with torch.inference_mode():
        for raw in range(89,138):
            record=next(x for x in cameras if x['img_name']==f'0008{raw:03d}');c2w=np.eye(4);c2w[:3,:3]=record['rotation'];c2w[:3,3]=record['position'];W=np.linalg.inv(c2w)
            c=Camera(raw,W[:3,:3].T.copy(),W[:3,3].copy(),cx=record['cx']/4,cy=record['cy']/4,fx=record['fx']/4,fy=record['fy']/4,
                image=torch.zeros((3,320,480)),image_name=f'0008{raw:03d}',resolution=(480,320),timestamp=stamps[raw-89])
            native=render_composite(c,static,dynamic,None,args,bg,environment);ordinary=render_composite(c,static,None,None,args,bg,environment)
            save(raw,'native_B',native['rgb']);save(raw,'static_B',ordinary['rgb']);row=dict(raw=raw,variants={})
            for name,road in roads.items():
                isolated=render_road(c,road,args);edited=render_composite(c,static,None,road,args,bg,environment)
                save(raw,name+'_road',isolated['rgb']);save(raw,name+'_edit',edited['rgb'])
                alpha=cpu(isolated['alpha'])[0];visible=cpu(edited['road_visible_alpha'])[0]
                Image.fromarray(np.uint8(np.clip(visible,0,1)*255)).save(dest/f'{raw}_{name}_contribution.png')
                row['variants'][name]=dict(points=len(road.xyz),critical={f'{x},{y}':dict(road_alpha=float(alpha[y,x]),actual_road_contribution=float(visible[y,x])) for x,y in [(255,240),(256,219)]} if raw==102 else {})
                if raw in [102,129,137]:np.savez_compressed(dest/f'{raw}_{name}_float.npz',road_rgb=cpu(isolated['rgb']),road_alpha=alpha,edited_rgb=cpu(edited['rgb']),actual_road_contribution=visible,static_rgb=cpu(ordinary['rgb']))
                del isolated,edited
            again=render_composite(c,static,dynamic,None,args,bg,environment)
            row['native_master_repeat_max_abs']=float((again['rgb']-native['rgb']).abs().max());assert row['native_master_repeat_max_abs']==0.
            rows.append(row);print(json.dumps(row),flush=True)
            del native,ordinary,again,c;gc.collect();torch.cuda.empty_cache()
    assert before=={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in files}
    stop.set();observer.join(timeout=7)
    result=dict(status='READ_ONLY_RENDER_COMPLETE',rows=rows,elapsed_seconds=time.time()-start,training_updates=0,optimizer_created=False,
        road_stage='untrained source-colour initialization, opacity0.5, sigma0.13/0.13/0.015m',native_B_files_unchanged=True,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),nvml_sample_peak_mib=peak[0],nvml_sampling_errors=errors,
        quality_accepted=False,human_verdict=None,limitation='no suppression of static ghost; no learned road training; original full reconstruction uses B without road')
    (dest/'receipt.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu-dir',default='gpu');a=p.parse_args();run(a.output,a.gpu_dir)

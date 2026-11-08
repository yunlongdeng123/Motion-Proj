from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'engine'),str(ROOT/'api')]

def load(model,iteration):
    sys.path[:0]=[str(ROOT/'engine'),str(ROOT/'api')]
    import torch
    from omegaconf import OmegaConf
    from scene import GaussianModel,Dynamic_GaussianModel,EnvLight
    from drivingeditor_dynamic_edit import install_identity_serialization
    install_identity_serialization();torch.set_num_threads(1)
    cfg=json.loads((model/'effective_config.json').read_text());args=OmegaConf.create(cfg)
    models=[GaussianModel(args),Dynamic_GaussianModel(args),EnvLight(resolution=args.env_map_res).cuda()]
    for obj,prefix in zip(models,['chkpnt','dynamic_chkpnt','env_light_chkpnt']):
        data,step=torch.load(str(model/f'{prefix}{iteration}.pth'),map_location='cpu',weights_only=False,mmap=True);assert step==iteration;data=list(data)
        if prefix=='env_light_chkpnt':data[0]=torch.nn.Parameter(data[0].detach().cuda(),requires_grad=False)
        else:
            for i in range(1,11):data[i]=data[i].detach().cuda()
            obj.time_duration=tuple(cfg['time_duration'])
        obj.restore(tuple(data),training_args=None)
        for value in vars(obj).values():
            if torch.is_tensor(value):value.requires_grad_(False)
        for value in obj.parameters() if isinstance(obj,torch.nn.Module) else []:value.requires_grad_(False)
    return args,models,cfg

def camera(model,cfg,raw):
    import torch
    from scene.cameras import Camera
    records=json.loads((model/'cameras.json').read_text());c=next(c for c in records if int(c['img_name'][-3:])==raw)
    pose=np.eye(4);pose[:3,:3]=c['rotation'];pose[:3,3]=c['position'];w=np.linalg.inv(pose)
    spec=cfg['followup_scene_spec'];times={int(n[-3:]):t for n,t in zip(spec['frame_ids'],spec['model_timestamps'])}
    return Camera(raw,w[:3,:3].T.copy(),w[:3,3].copy(),cx=c['cx']*480/c['width'],cy=c['cy']*320/c['height'],fx=c['fx']*480/c['width'],fy=c['fy']*320/c['height'],image=torch.zeros((3,320,480)),image_name=c['img_name'],resolution=(480,320),timestamp=times[raw])

def native_rows(cam,model,pipe,dynamic=False):
    import torch
    from gaussian_renderer import prepare_rasterization
    from utils.sh_utils import eval_sh
    with torch.no_grad():
        rows=prepare_rasterization(cam,model,pipe,dynamic=dynamic)
        if rows['colors_precomp'] is None:
            d=torch.nn.functional.normalize(rows['means3D']-cam.camera_center,dim=1)
            rgb=(eval_sh(model.active_sh_degree,rows['shs'].transpose(1,2),d)+.5).clamp_min(0)
        else:rgb=rows['colors_precomp']
        return dict(rows,colors_precomp=rgb,shs=None,means2D=torch.zeros_like(rows['means3D']))

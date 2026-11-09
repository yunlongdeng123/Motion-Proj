"""V1 无编辑独立入口：只恢复原生完整场景，直接原渲染器，不导入编辑包。

本脚本在 CPU 准备阶段只编译；真实恢复、渲染与数值对齐必须等 GPU smoke。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--engine',type=Path,required=True)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--iteration',type=int,default=10000)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--raws',type=int,nargs='+')
    ap.add_argument('--compare-saved-dir',type=Path)
    ap.add_argument('--camera-manifest',type=Path)
    a=ap.parse_args()
    cfg=json.loads((a.model/'effective_config.json').read_text())
    assert cfg['delete_actor_ids']==[] and a.iteration>0
    sys.path.insert(0,str(a.engine))
    import torch
    from omegaconf import OmegaConf
    from scene import GaussianModel,Dynamic_GaussianModel,EnvLight
    from scene.cameras import Camera
    import gaussian_renderer
    assert Path(gaussian_renderer.__file__).resolve()==(a.engine/'gaussian_renderer/__init__.py').resolve()
    assert not any(m.startswith('drivingeditor_dynamic_edit') for m in sys.modules)
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    args=OmegaConf.create(cfg)
    models=[GaussianModel(args),Dynamic_GaussianModel(args),EnvLight(resolution=args.env_map_res).cuda()]
    for model,prefix in zip(models,['chkpnt','dynamic_chkpnt','env_light_chkpnt']):
        data,step=torch.load(str(a.model/f'{prefix}{a.iteration}.pth'),map_location='cpu',weights_only=False,mmap=True)
        assert step==a.iteration
        data=list(data)
        if prefix=='env_light_chkpnt':
            data[0]=torch.nn.Parameter(data[0].detach().cuda(),requires_grad=False)
        else:
            assert len(data)==18
            for i in range(1,11):
                data[i]=data[i].detach().cuda()
            # 身份只在Adam附加元数据内；不解释或使用身份，不改任何渲染参数。
            data[14]=dict(data[14])
            data[14].pop('dynamic_actor_ownership_v1',None)
            model.time_duration=tuple(cfg['time_duration'])
        model.restore(tuple(data),training_args=None)
        assert getattr(model,'optimizer',None) is None
        for v in vars(model).values():
            if torch.is_tensor(v):
                v.requires_grad_(False)
    static,dynamic,env=models
    records=json.loads(a.camera_manifest.read_text())['cameras'] if a.camera_manifest else json.loads((a.model/'cameras.json').read_text())
    headers={int(c['img_name'][-3:]):c for c in records}
    spec=cfg['followup_scene_spec']
    times={int(n[-3:]):t for n,t in zip(spec['frame_ids'],spec['model_timestamps'])}
    times.update({int(c['img_name'][-3:]):c['model_time'] for c in records if 'model_time' in c})
    raws=a.raws or sorted(headers)
    assert set(raws)<=set(headers)
    a.output.mkdir(parents=True,exist_ok=False)
    bg=torch.zeros(3,device='cuda')
    rows=[]
    start=time.monotonic()
    with torch.no_grad():
        for raw in raws:
            c=headers[raw]
            pose=np.eye(4);pose[:3,:3]=c['rotation'];pose[:3,3]=c['position'];world=np.linalg.inv(pose)
            cam=Camera(raw,world[:3,:3].T.copy(),world[:3,3].copy(),
                cx=c['cx']*480/c['width'],cy=c['cy']*320/c['height'],
                fx=c['fx']*480/c['width'],fy=c['fy']*320/c['height'],
                image=torch.zeros((3,320,480)),image_name=c['img_name'],resolution=(480,320),timestamp=times[raw])
            pack=gaussian_renderer.render(cam,static,dynamic,args,bg,env_map=env,is_training=False)
            rgb=pack['render'].detach().cpu().numpy()
            assert np.isfinite(rgb).all()
            folder=a.output/f'{raw:03d}';folder.mkdir()
            Image.fromarray(np.uint8(rgb.transpose(1,2,0).clip(0,1)*255+.5)).save(folder/'keep.png')
            np.savez_compressed(folder/'keep.npz',rgb=rgb,alpha=pack['alpha'].cpu().numpy(),depth_sum=pack['depth'].cpu().numpy())
            error=None
            if a.compare_saved_dir:
                reference=np.load(a.compare_saved_dir/f'{raw:03d}/keep.npz')['rgb']
                error=float(abs(rgb-reference).max())
                assert error<2e-5,(raw,error)
            rows.append(dict(raw=raw,saved_native_maxabs=error))
            print(raw,error,flush=True)
    assert not any(m.startswith('drivingeditor_dynamic_edit') for m in sys.modules)
    (a.output/'summary.json').write_text(json.dumps(dict(status='PASS_V1_ORIGINAL_NO_EDIT_RENDER',
        model=str(a.model),iteration=a.iteration,renderer=str(gaussian_renderer.__file__),
        editing_package_imported=False,semantic_rebinding=False,asset_insert=False,
        image_composite=False,updates=0,seconds=time.monotonic()-start,rows=rows,human_verdict=None),indent=2))


if __name__=='__main__':
    main()

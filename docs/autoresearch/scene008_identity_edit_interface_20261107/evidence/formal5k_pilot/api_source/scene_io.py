"""薄 checkpoint/Camera 读取器：不创建 Scene，不读原 PLY、mask 或观测 RGB。"""
from pathlib import Path
import gc
import json
import re
import sys

import numpy as np
import torch

from .ownership import ActorEditQuery, OWNERSHIP_KEY, LINEAGE_KEY, _validate_table, install_identity_serialization

DEFAULT_ENGINE=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018/training_source/scene0008_089_137')


def checkpoint_actors(model, iteration):
    payload, step=torch.load(str(Path(model)/f'dynamic_chkpnt{iteration}.pth'),map_location='cpu',weights_only=False,mmap=True)
    if int(step)!=iteration or len(payload)!=18:
        raise ValueError('动态 checkpoint 格式/iteration 不匹配')
    state=payload[14]
    meta=state.get(OWNERSHIP_KEY) if isinstance(state,dict) else None
    lineage=state.get(LINEAGE_KEY) if isinstance(state,dict) else None
    if not isinstance(meta,dict) or meta.get('schema_version')!=1 or not isinstance(lineage,dict):
        raise ValueError('旧 checkpoint 缺少可靠 actor 身份；只能用原 renderer 做不可编辑 KEEP 参考')
    declared=_validate_table(meta['dynamic_actor_ids'],meta['source_actor_ids'],lineage['lineage_ids'],meta['actor_ids'])
    if len(meta['actor_ids'])!=len(payload[1]):raise ValueError('actor 身份与 checkpoint 高斯行数不匹配')
    info=dict(dynamic_actor_ids=list(declared),dynamic_rows=len(payload[1]),owner_schema_version=1)
    del payload;gc.collect()
    return info


def model_tensors(models):
    result={}
    for group,model in models.items():
        values={key:value for key,value in vars(model).items() if torch.is_tensor(value)}
        if callable(getattr(model,'named_parameters',None)):values.update(dict(model.named_parameters()))
        if callable(getattr(model,'named_buffers',None)):values.update(dict(model.named_buffers()))
        result.update({group+'.'+key:value for key,value in values.items()})
    if 'environment' in models and 'environment.base' not in result:
        raise ValueError('环境参数 base 未纳入只读验证')
    return result


def load_native_models(model, iteration, engine=DEFAULT_ENGINE):
    """复用原生 18tuple restore 约定；training_args=None，禁止创建优化器。"""
    engine=Path(engine).resolve();sys.path.insert(0,str(engine))
    from omegaconf import OmegaConf
    from scene import GaussianModel, Dynamic_GaussianModel, EnvLight
    import scene.gaussian_model as gaussian_model
    import gaussian_renderer
    if Path(gaussian_model.__file__).resolve()!=engine/'scene/gaussian_model.py':raise ValueError('加载了错误 native engine')
    install_identity_serialization(GaussianModel,Dynamic_GaussianModel)
    config=json.loads((Path(model)/'effective_config.json').read_text())
    args=OmegaConf.create(config)
    static,dynamic=GaussianModel(args),Dynamic_GaussianModel(args)
    env=EnvLight(resolution=args.env_map_res).cuda()
    for target,prefix in [(static,'chkpnt'),(dynamic,'dynamic_chkpnt'),(env,'env_light_chkpnt')]:
        data,step=torch.load(str(Path(model)/f'{prefix}{iteration}.pth'),map_location='cpu',weights_only=False,mmap=True)
        if int(step)!=iteration:raise ValueError('checkpoint iteration 不一致')
        values=list(data)
        if target is env:values[0]=torch.nn.Parameter(values[0].detach().cuda(),requires_grad=False)
        else:
            for index in range(1,11):values[index]=values[index].detach().cuda()
            target.time_duration=tuple(config['time_duration'])
        target.restore(tuple(values),training_args=None)
        if getattr(target,'optimizer',None) is not None:raise ValueError('查询不能创建优化器')
        del values,data;gc.collect()
    models=dict(static=static,dynamic=dynamic,environment=env)
    for value in model_tensors(models).values():value.requires_grad_(False)
    return args,models,gaussian_renderer


def camera_headers(model):
    """沿用已验证 Camera JSON 的 c2w -> w2c、内参与保存 timestamp。"""
    model=Path(model)
    rows=json.loads((model/'cameras.json').read_text())
    binding=json.loads((model/'actual_scene_binding.json').read_text())
    width,height=binding['render_wh']
    def raw(name):
        match=re.search(r'(\d{3})$',str(name))
        if match is None:raise ValueError('Camera frame ID 无法识别')
        return int(match.group(1))
    times={raw(name):float(t) for name,t in zip(binding['frame_ids'],binding['timestamps'])}
    if len(times)!=len(binding['frame_ids']) or len(times)!=len(binding['timestamps']):raise ValueError('Camera timestamp 集合无效')
    result={}
    for row in rows:
        index=raw(row['img_name'])
        if index in result or index not in times:raise ValueError('Camera 帧重复/缺 timestamp')
        c2w=np.eye(4);c2w[:3,:3]=row['rotation'];c2w[:3,3]=row['position']
        W=np.linalg.inv(c2w)
        result[index]=dict(raw=index,W=W,width=width,height=height,timestamp=times[index],
            fx=row['fx']*width/row['width'],fy=row['fy']*height/row['height'],
            cx=row['cx']*width/row['width'],cy=row['cy']*height/row['height'])
    if len(result)!=49 or set(result)!=set(range(89,138)):raise ValueError('此 Scene008 入口需要保存的原49帧 Camera')
    return result


def make_camera(header):
    from scene.cameras import Camera
    W=header['W'];raw=header['raw'];w,h=header['width'],header['height']
    camera=Camera(raw,W[:3,:3].T.copy(),W[:3,3].copy(),cx=header['cx'],cy=header['cy'],fx=header['fx'],fy=header['fy'],
        image=torch.zeros((3,h,w)),image_name=f'0008{raw:03d}',resolution=(w,h),timestamp=header['timestamp'])
    if not np.allclose(camera.world_view_transform.detach().cpu().numpy().T,W,atol=1e-5,rtol=0):raise ValueError('Native Camera 坐标系不匹配')
    return camera

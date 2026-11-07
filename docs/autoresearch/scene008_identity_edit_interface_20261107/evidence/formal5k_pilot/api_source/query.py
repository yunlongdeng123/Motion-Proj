"""同一已保存身份的 checkpoint，独立选择任意合法 delete_actor_ids。"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .ownership import ActorEditQuery
from .render import render_scene
from .scene_io import DEFAULT_ENGINE,checkpoint_actors,load_native_models,camera_headers,make_camera,model_tensors

PILOT=(105,130,137)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--iteration',type=int)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--delete-actor-ids',type=int,nargs='*',default=[])
    parser.add_argument('--scope',choices=['pilot','all49'],default='pilot')
    parser.add_argument('--engine',type=Path,default=DEFAULT_ENGINE)
    parser.add_argument('--describe',action='store_true',help='只在CPU读取 checkpoint 中的身份信息')
    preliminary,_=parser.parse_known_args(argv)
    if preliminary.config:
        config=json.loads(preliminary.config.read_text())
        allowed={'model','iteration','output','delete_actor_ids','scope','engine'}
        if set(config)-allowed:parser.error('配置含未知键；dynamic_actor_ids 只从 checkpoint 读取')
        parser.set_defaults(**config)
    opts=parser.parse_args(argv)
    if opts.model is None or opts.iteration is None or opts.iteration<=0:parser.error('需要 --model 与正整数 --iteration')
    opts.model=Path(opts.model).resolve();opts.engine=Path(opts.engine).resolve()
    info=checkpoint_actors(opts.model,opts.iteration)
    query=ActorEditQuery(info['dynamic_actor_ids'],opts.delete_actor_ids)
    if opts.describe:
        print(json.dumps(dict(**info,delete_actor_ids=list(query.delete_actor_ids),optimization_updates=0),ensure_ascii=False))
        return
    if opts.output is None:parser.error('渲染需要 --output')
    if opts.scope not in ['pilot','all49']:parser.error('scope 必须为 pilot 或 all49')
    output=Path(opts.output).resolve()
    if output.exists() or output.is_relative_to(opts.model):parser.error('输出必须是 checkpoint 目录之外的新目录')
    if not torch.cuda.is_available():raise RuntimeError('原生光栅化需要真实CUDA；不自动换后端')
    torch.set_num_threads(1)
    headers=camera_headers(opts.model)
    raws=list(PILOT) if opts.scope=='pilot' else sorted(headers)
    args,models,renderer=load_native_models(opts.model,opts.iteration,opts.engine)
    static,dynamic,env=[models[name] for name in ['static','dynamic','environment']]
    before={name:value.detach().cpu().clone() for name,value in model_tensors(models).items()}
    background=torch.tensor([1.,1.,1.] if args.white_background else [0.,0.,0.],device='cuda')
    output.mkdir(parents=True)
    for raw in raws:
        camera=make_camera(headers[raw])
        with torch.no_grad():
            package=render_scene(camera,static,dynamic,args,background,dynamic_actor_ids=query.dynamic_actor_ids,
                delete_actor_ids=query.delete_actor_ids,renderer=renderer,env_map=env,is_training=False)
        rgb=package['render'].detach().clamp(0,1).permute(1,2,0).cpu().numpy()
        if not np.isfinite(rgb).all():raise RuntimeError('原生渲染出现非finite RGB')
        Image.fromarray((rgb*255+.5).astype(np.uint8)).save(output/f'{raw:03d}.png')
        del package,camera
    current=model_tensors(models)
    if set(current)!=set(before) or any(not torch.equal(before[name],current[name].detach().cpu()) for name in before):
        raise RuntimeError('只读查询改变了模型参数/身份')
    summary=dict(status='PASS_READ_ONLY_NATIVE_QUERY',model=str(opts.model),iteration=opts.iteration,
        dynamic_actor_ids=list(query.dynamic_actor_ids),delete_actor_ids=list(query.delete_actor_ids),
        owner_schema_version=info['owner_schema_version'],scope=opts.scope,raw_ids=raws,frame_count=len(raws),
        parameters_unchanged=True,optimizer_created=False,optimization_updates=0,
        reads_original_PLY_or_masks=False,native_joint_depth_rerender=True,
        reconstruction_quality_verdict=None,editing_quality_verdict=None)
    (output/'query.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()

"""零训练：原完整checkpoint与推理包的原生KEEP/owner逐值比较。"""
from pathlib import Path
import argparse
import gc
import json

import numpy as np
import torch

from .ownership import actor_ids
from .render import render_scene
from .scene_io import DEFAULT_ENGINE,checkpoint_actors,load_native_models,camera_headers,make_camera

PILOT=(97,105,109)


def collect(model,iteration,engine):
    info=checkpoint_actors(model,iteration);args,models,renderer=load_native_models(model,iteration,engine)
    static,dynamic,env=[models[k] for k in ['static','dynamic','environment']]
    actors=tuple(info['dynamic_actor_ids']);owners=actor_ids(dynamic)
    result=dict(info=info,owners=owners.detach().cpu().clone(),source_actor_ids=dynamic._dynamic_actor_source_ids.detach().cpu().clone(),frames={})
    background=torch.tensor([1.,1.,1.] if args.white_background else [0.,0.,0.],device='cuda')
    static_tags=torch.zeros((len(static.get_xyz),len(actors)+1),device='cuda');static_tags[:,0]=1
    tags=torch.cat([torch.zeros((len(owners),1),device='cuda'),torch.stack([owners==a for a in actors],1).float()],dim=1)
    headers=camera_headers(model)
    for raw in PILOT:
        camera=make_camera(headers[raw])
        with torch.no_grad():package=render_scene(camera,static,dynamic,args,background,dynamic_actor_ids=actors,
            env_map=env,is_training=False,other=[static_tags],dynamic_other=[tags])
        result['frames'][raw]={key:package[key].detach().cpu().clone() for key in ['render','alpha','feature']}
        del package,camera
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--original',type=Path,required=True)
    parser.add_argument('--inference-package',type=Path,required=True);parser.add_argument('--iteration',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--engine',type=Path,default=DEFAULT_ENGINE)
    opts=parser.parse_args();torch.set_num_threads(1)
    original=collect(opts.original,opts.iteration,opts.engine);gc.collect()
    inference=collect(opts.inference_package,opts.iteration,opts.engine)
    if original['info']!=inference['info']:raise ValueError('Checkpoint actor集合/行数变化')
    for key in ['owners','source_actor_ids']:
        if not torch.equal(original[key],inference[key]):raise ValueError('导出包身份改变')
    rows=[]
    for raw in PILOT:
        delta={key:float((original['frames'][raw][key]-inference['frames'][raw][key]).abs().max()) for key in ['render','alpha','feature']}
        if any(value!=0 for value in delta.values()):raise ValueError(f'导出包原生float输出改变: {raw} {delta}')
        rows.append(dict(raw=raw,max_abs_difference=delta))
    result=dict(status='PASS_EXACT_NATIVE_KEEP_AND_OWNER_INFERENCE_PACKAGE',raw_ids=list(PILOT),frames=rows,
        original=str(opts.original),inference_package=str(opts.inference_package),iteration=opts.iteration,
        current_owner_ids_exact=True,source_actor_table_exact=True,optimizer_created=False,optimization_updates=0)
    opts.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()

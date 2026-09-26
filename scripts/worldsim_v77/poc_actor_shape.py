"""官方 Hunyuan3D-2.1 image-to-shape，两个目标各生成一个 GLB。"""
import json
import pathlib
import sys
import time

import numpy as np
import torch
import trimesh

BASE=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPLICIT-POC-20260926/r1')
REPO=pathlib.Path('/root/autodl-tmp/third_party/hunyuan3d-2.1-v61-me2')
MODEL='/root/autodl-tmp/models/worldsim_v77_poc_hunyuan3d21'
sys.path.insert(0,str(REPO/'hy3dshape'))
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

torch.set_num_threads(4)
pipeline=Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(MODEL,device='cuda',dtype=torch.float16,
  subfolder='hunyuan3d-dit-v2-1')
print('HUNYUAN_SHAPE_READY',flush=True)
for i,scene in enumerate(['scene_0230','scene_0255']):
    out=BASE/scene/'actor';selection=json.loads((out/'keyframes.json').read_text())
    target=out/'shape_untextured.glb'
    if target.exists():raise FileExistsError(target)
    seed=7701+i;start=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    generator=torch.Generator(device='cuda').manual_seed(seed)
    meshes=pipeline(image=selection['shape_input'],num_inference_steps=50,
       guidance_scale=7.5,octree_resolution=256,generator=generator,enable_pbar=False)
    assert len(meshes)==1
    mesh=meshes[0];mesh.export(target)
    loaded=trimesh.load(target,force='scene')
    geometries=list(loaded.geometry.values())
    vertices=sum(len(m.vertices) for m in geometries);faces=sum(len(m.faces) for m in geometries)
    if not vertices or not faces:raise RuntimeError('空 Hunyuan mesh')
    if not all(np.isfinite(m.vertices).all() for m in geometries):raise RuntimeError('非有限网格顶点')
    report={'scene':scene,'input':selection['shape_input'],'output':str(target),'seed':seed,
      'steps':50,'guidance_scale':7.5,'octree_resolution':256,'training_steps':0,
      'vertices':vertices,'faces':faces,'bounds':loaded.bounds.tolist(),
      'elapsed_s':time.monotonic()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
      'model_root':MODEL,'source_repo':str(REPO),'human_verdict':None}
    (out/'shape_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)
    del mesh,meshes,loaded,geometries;torch.cuda.empty_cache()

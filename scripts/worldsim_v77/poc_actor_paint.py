"""Run official Hunyuan3D-2.1 PBR texture stage, one scene per process."""
import argparse
import json
import pathlib
import sys
import time
import types

import torch
import trimesh

ROOT = pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPLICIT-POC-20260926/r1')
REPO = pathlib.Path('/root/autodl-tmp/third_party/hunyuan3d-2.1-v61-me2')
MODEL = pathlib.Path('/root/autodl-tmp/models/worldsim_v77_poc_hunyuan3d21')

parser = argparse.ArgumentParser()
parser.add_argument('scene', choices=['scene_0230', 'scene_0255'])
args = parser.parse_args()
torch.set_num_threads(4)
torch.manual_seed(7701 if args.scene == 'scene_0230' else 7702)

# The library imports bpy for its optional GLB exporter. Here the official
# PBR pipeline saves OBJ+maps; Blender CLI performs the actual GLB export.
sys.modules['bpy'] = types.ModuleType('bpy')
import torchvision.transforms.functional as tf
shim = types.ModuleType('torchvision.transforms.functional_tensor')
shim.rgb_to_grayscale = tf.rgb_to_grayscale
sys.modules['torchvision.transforms.functional_tensor'] = shim

import huggingface_hub
download = huggingface_hub.snapshot_download
def local_snapshot(repo_id, *a, **kw):
    if repo_id == 'tencent/Hunyuan3D-2.1':
        return str(MODEL)
    return download(repo_id, *a, **kw)
huggingface_hub.snapshot_download = local_snapshot

sys.path.insert(0, str(REPO / 'hy3dpaint'))
from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

actor = ROOT / args.scene / 'actor'
selection = json.loads((actor / 'keyframes.json').read_text())
mesh_in = actor / 'shape_untextured.glb'
mesh_out = actor / 'actor_pbr.obj'
if mesh_out.exists():
    raise FileExistsError(mesh_out)
cfg = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
cfg.multiview_cfg_path = str(REPO / 'hy3dpaint/cfgs/hunyuan-paint-pbr.yaml')
cfg.multiview_pretrained_path = 'tencent/Hunyuan3D-2.1'
cfg.dino_ckpt_path = '/root/autodl-tmp/models/worldsim_v77_poc_dinov2_giant'
cfg.realesrgan_ckpt_path = '/root/autodl-tmp/models/worldsim_v77_poc_esrgan/RealESRGAN_x4plus.pth'

start = time.monotonic()
print('PAINT_INIT', args.scene, flush=True)
pipeline = Hunyuan3DPaintPipeline(cfg)
print('PAINT_READY', args.scene, flush=True)
torch.cuda.reset_peak_memory_stats()
result = pipeline(mesh_path=str(mesh_in), image_path=selection['shape_input'],
                  output_mesh_path=str(mesh_out), use_remesh=False, save_glb=False)
loaded = trimesh.load(result, force='mesh')
report = {'scene': args.scene, 'input': selection['shape_input'],
          'source_shape': str(mesh_in), 'output_obj': str(mesh_out),
          'source': str(REPO), 'model_root': str(MODEL),
          'vertices': len(loaded.vertices), 'faces': len(loaded.faces),
          'elapsed_s': time.monotonic()-start,
          'peak_gpu_gib': torch.cuda.max_memory_allocated()/2**30,
          'human_verdict': None}
(actor / 'paint_summary.json').write_text(json.dumps(report, indent=2)+'\n')
print('PAINT_DONE', json.dumps(report), flush=True)

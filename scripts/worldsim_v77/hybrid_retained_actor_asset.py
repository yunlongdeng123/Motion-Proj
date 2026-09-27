"""r40：用现有Hunyuan2.1给真实保留SUV52建立固定资产，先做已知视角正控制。"""
import os,sys,time,datetime,argparse
from pathlib import Path
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('OMP_NUM_THREADS','4')
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from repair_common import dump,read
from PIL import Image
import numpy as np
BASE=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927');ROOT=BASE/'r40';REPO=Path('/root/autodl-tmp/third_party/hunyuan3d-2.1-v61-me2');MODEL=Path('/root/autodl-tmp/models/worldsim_v77_poc_hunyuan3d21')

def prepare():
 assert not ROOT.exists();ROOT.mkdir();spec=next(s for s in read(BASE/'r2/registration.json')['scenes'] if s['name']=='scene_0255');raw=Image.open(Path(spec['data'])/'images/130_3.jpg').convert('RGB');mask=Image.open(BASE/'r14/130_3/sam.png').convert('L').resize(raw.size,Image.Resampling.NEAREST);m=np.array(mask)>127;y,x=np.where(m);box=(max(0,int(x.min())-12),max(0,int(y.min())-12),min(raw.width,int(x.max())+13),min(raw.height,int(y.max())+13));rgba=np.dstack([np.array(raw),np.uint8(m)*255]);Image.fromarray(rgba).crop(box).save(ROOT/'source_rgba.png');raw.crop(box).save(ROOT/'source_context.png');Image.fromarray(np.uint8(m)*255).crop(box).save(ROOT/'source_mask.png')
 dump(ROOT/'registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r40',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene='scene_0255',delete_actor='25',preserve_actor='52',question='Can an explicit fixed asset preserve the real hidden neighbor better than asking video diffusion to re-create it each frame?',intervention='Reuse existing official Hunyuan3D-2.1 shape/PBR frozen modules, one real reference for actor52. This is extending background+explicit actors to a retained neighbor, not replacing DriveEditor or adding a model family.',source=dict(frame=130,camera=3,image=str(Path(spec['data'])/'images/130_3.jpg'),mask=str(BASE/'r14/130_3/sam.png'),crop_xyxy=box,native_crop_size=[box[2]-box[0],box[3]-box[1]]),roles='Original RGB+existing SAM2 only for generation; missing sides are inferred prior, never factual evidence. GT pose/size reserved for later placement. Human r18>r15 preserved; r18/r21 outputs never overwritten.',fixed=dict(model_root=str(MODEL),source_repo=str(REPO),shape_steps=50,seed=7740,guidance_scale=7.5,octree_resolution=256,texture_views=6,texture_resolution=512,cpu_threads=4),positive_control='Inspect mesh and render at observed source/other visible reference poses before hidden DELETE integration. Source f130 is generation input so its reconstruction is not heldout.',stop_rule='One asset only, no seed sweep. Reject wrong body/placement/appearance at observed controls; do not declare background success from valid GLB or use wrong placement to reject the mesh.',failure_ledger_refs=['V77-F02'],human_verdict=None,background_input_dir=None))
 print('R40_PREPARED',box,flush=True)

def shape():
 import torch,trimesh,signal
 assert (ROOT/'registration.json').exists() and not (ROOT/'shape_state.json').exists();state=dict(state='loading',pid=os.getpid(),human_verdict=None);dump(ROOT/'shape_state.json',state)
 try:
  torch.set_num_threads(4);sys.path.insert(0,str(REPO/'hy3dshape'));from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
  pipeline=Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(str(MODEL),device='cuda',dtype=torch.float16,subfolder='hunyuan3d-dit-v2-1');state['state']='running';dump(ROOT/'shape_state.json',state);torch.cuda.reset_peak_memory_stats();start=time.time();generator=torch.Generator(device='cuda').manual_seed(7740);meshes=pipeline(image=str(ROOT/'source_rgba.png'),num_inference_steps=50,guidance_scale=7.5,octree_resolution=256,generator=generator,enable_pbar=False);assert len(meshes)==1;meshes[0].export(ROOT/'shape_untextured.glb');scene=trimesh.load(ROOT/'shape_untextured.glb',force='scene');assert all(np.isfinite(m.vertices).all() for m in scene.geometry.values());state.update(state='complete_pending_visual_review',vertices=sum(len(m.vertices) for m in scene.geometry.values()),faces=sum(len(m.faces) for m in scene.geometry.values()),bounds=scene.bounds.tolist(),seconds=time.time()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30);dump(ROOT/'shape_state.json',state);print('R40_SHAPE_COMPLETE',state,flush=True)
 except Exception as exc:state.update(state='failed_engineering',error=repr(exc));dump(ROOT/'shape_state.json',state);raise

def paint(reference=None):
 import torch,trimesh,types
 assert (ROOT/'shape_untextured.glb').exists() and not (ROOT/'paint_state.json').exists();state=dict(state='loading',pid=os.getpid(),human_verdict=None);dump(ROOT/'paint_state.json',state)
 try:
  torch.set_num_threads(4);torch.manual_seed(7740);sys.modules['bpy']=types.ModuleType('bpy');import torchvision.transforms.functional as tf;shim=types.ModuleType('torchvision.transforms.functional_tensor');shim.rgb_to_grayscale=tf.rgb_to_grayscale;sys.modules['torchvision.transforms.functional_tensor']=shim
  import huggingface_hub
  download=huggingface_hub.snapshot_download
  def local(repo_id,*a,**kw):return str(MODEL) if repo_id=='tencent/Hunyuan3D-2.1' else download(repo_id,*a,**kw)
  huggingface_hub.snapshot_download=local;sys.path.insert(0,str(REPO/'hy3dpaint'));from textureGenPipeline import Hunyuan3DPaintConfig,Hunyuan3DPaintPipeline
  reference=Path(reference) if reference is not None else ROOT/'source_rgba.png'
  cfg=Hunyuan3DPaintConfig(max_num_view=6,resolution=512);cfg.multiview_cfg_path=str(REPO/'hy3dpaint/cfgs/hunyuan-paint-pbr.yaml');cfg.multiview_pretrained_path='tencent/Hunyuan3D-2.1';cfg.dino_ckpt_path='/root/autodl-tmp/models/worldsim_v77_poc_dinov2_giant';cfg.realesrgan_ckpt_path='/root/autodl-tmp/models/worldsim_v77_poc_esrgan/RealESRGAN_x4plus.pth';pipeline=Hunyuan3DPaintPipeline(cfg);state['state']='running';dump(ROOT/'paint_state.json',state);torch.cuda.reset_peak_memory_stats();start=time.time();result=pipeline(mesh_path=str(ROOT/'shape_untextured.glb'),image_path=str(reference),output_mesh_path=str(ROOT/'actor_pbr.obj'),use_remesh=False,save_glb=False);loaded=trimesh.load(result,force='mesh');state.update(state='complete_pending_visual_review',vertices=len(loaded.vertices),faces=len(loaded.faces),seconds=time.time()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,output_obj=str(result),texture_reference=str(reference));dump(ROOT/'paint_state.json',state);print('R40_PAINT_COMPLETE',state,flush=True)
 except Exception as exc:state.update(state='failed_engineering',error=repr(exc));dump(ROOT/'paint_state.json',state);raise
if __name__=='__main__':{'prepare':prepare,'shape':shape,'paint':paint}[sys.argv[-1]]()

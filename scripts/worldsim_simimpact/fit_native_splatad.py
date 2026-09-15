"""在已登记完整日志上拟合官方传感器场景；保留留出观测和原生优化预算。"""
import argparse, copy, json, os, random, subprocess, time
from pathlib import Path
import numpy as np
import torch

N=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-LIDAR-01/20260915-r1')
B=Path('/root/autodl-tmp/external/worldsim_simimpact')
ap=argparse.ArgumentParser()
ap.add_argument('--scene',choices=['scene-0004','scene-0061'],default='scene-0004')
ap.add_argument('--run',default='native-r1')
ap.add_argument('--resume',type=Path)
args=ap.parse_args()
from splatad_compat import apply
apply()
from nerfstudio.configs.method_configs import method_configs
from nerfstudio.data.dataparsers.nuscenes_dataparser import NuScenesDataParserConfig
config=copy.deepcopy(method_configs['splatad'])
target_steps=config.max_num_iterations
start_step=0
if args.resume:
    checkpoints=list(args.resume.glob('step-*.ckpt'))
    if not checkpoints:raise RuntimeError('No checkpoint in resume directory')
    start_step=max(int(p.stem.split('-')[-1]) for p in checkpoints)+1
    config.max_num_iterations=target_steps-start_step
    if config.max_num_iterations<=0:raise RuntimeError('Registered fit budget is already complete')
config.output_dir=N/'fits'
config.experiment_name=args.scene
config.timestamp=args.run
config.machine.seed=20260915
config.vis='tensorboard'
config.pipeline.datamanager.dataparser=NuScenesDataParserConfig(
    data=N/'data',sequence=args.scene,add_missing_points=True,
    train_split_fraction=0.5)
config.pipeline.datamanager.cache_images='cpu'
config.pipeline.datamanager.cache_lidars='cpu'
config.pipeline.datamanager.max_thread_workers=6
config.pipeline.calc_fid_steps=()
config.logging.local_writer.enable=False
config.load_dir=args.resume
out=config.get_base_dir()
if out.exists(): raise RuntimeError(f'Existing fit: {out}; resume into a new --run directory')
random.seed(config.machine.seed);np.random.seed(config.machine.seed);torch.manual_seed(config.machine.seed)
torch.set_num_threads(6)
torch.backends.cudnn.benchmark=True
config.save_config()
manifest={'task_id':'WS-SIM-NATIVE-LIDAR-01','run':args.run,'scene':args.scene,
    'start_unix':time.time(),'seed':config.machine.seed,'max_steps':target_steps,'start_step':start_step,
    'iterations_in_this_invocation':config.max_num_iterations,'resume_source':str(args.resume) if args.resume else None,
    'role':'RGB+LiDAR scene fitting, strong metric simulator reference; not a feed-forward method or equal information budget',
    'split':'Official LINSPACE 50% train / 50% held-out sensor observations; full log dynamics and calibration are supplied',
    'changes_from_official_splatad':'CPU image/LiDAR cache; six caching workers; local TensorBoard only; FID disabled because appearance ranking is not the objective; remove unsupported mode argument only for torch.compile backend=eager on Torch2.4; optimization/model defaults retained',
    'source_revisions':{r:subprocess.check_output(['git','-C',str(B/r),'rev-parse','HEAD'],text=True).strip() for r in ['neurad-studio','SplatAD_splat','viser-splatad']},
    'runtime':{'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name()},
    'completed':False,'human_verdict':None}
(out/'fit_manifest.json').write_text(json.dumps(manifest,indent=2))
print('FIT_BEGIN',str(out),flush=True)
trainer=config.setup(local_rank=0,world_size=1)
trainer.setup()
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackLocation
def progress(step):
    row={'step':step,'time_unix':time.time(),'gaussians':trainer.pipeline.model.num_points,
        'gpu_memory_allocated_mib':torch.cuda.memory_allocated()/1024**2}
    (out/'progress.json').write_text(json.dumps(row,indent=2))
    print('FIT_PROGRESS',json.dumps(row),flush=True)
trainer.callbacks.append(TrainingCallback([TrainingCallbackLocation.AFTER_TRAIN_ITERATION],progress,update_every_num_iters=100))
trainer.train()
manifest.update(completed=True,end_unix=time.time())
(out/'fit_manifest.json').write_text(json.dumps(manifest,indent=2))
print('FIT_COMPLETED',str(out),flush=True)

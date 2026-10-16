"""Finite input verification only. No Gaussian training or fallback auto-approval."""
from pathlib import Path
import os,json,subprocess,time,datetime
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');root=R/'ownership_controls_r12_sam3';py=str(root/'runtime_sam3_dinov3/bin/python');state=root/'input_pipeline_state.json';started=time.monotonic()
def write(status,**kw):
 p=state.with_suffix('.tmp');p.write_text(json.dumps(dict(status=status,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw),indent=2));p.replace(state)
def live(pid):
 p=Path('/proc')/str(pid)/'stat';return p.exists() and p.read_text().split()[2]!='Z'
def GPU_stage(script,label):
 while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
  write('WAITING_OTHER_GPU_TASK',next_phase=label);time.sleep(30)
 write('RUNNING',phase=label);env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 subprocess.run(['flock','/root/autodl-tmp/drivingeditor_repro/campaign/gpu_slot.lock',py,str(R/script)],cwd=R,env=env,check=True)
try:
 pid=json.loads((root/'runtime_launch.json').read_text())['pid']
 while live(pid):
  if time.monotonic()-started>2700:raise TimeoutError('Runtime setup exceeds finite45minute engineering limit; inspect download; no method verdict')
  write('WAITING_TASK_LOCAL_RUNTIME_NO_GPU_HELD',runtime_setup_pid=pid);time.sleep(30)
 # Optional video support is task-local. Never pip into existing SAM3 or training env.
 write('VERIFYING_TASK_LOCAL_IMPORTS')
 probe=subprocess.run([py,'-c','import torch,transformers,importlib.util;print(bool(importlib.util.find_spec("psutil")))'],capture_output=True,text=True);probe.check_returncode()
 if probe.stdout.strip().splitlines()[-1]=='False':
  subprocess.run([py,'-m','pip','install','--no-cache-dir','--disable-pip-version-check','psutil==7.0.0','--index-url','https://mirrors.aliyun.com/pypi/simple'],check=True)
 code="import sys;sys.path.insert(0,'/root/autodl-tmp/idsplat/asset_sources/sam3');import torch,torchvision,transformers;from sam3.model_builder import build_sam3_image_model;from sam3.model.sam3_video_predictor import Sam3VideoPredictor;print(torch.__version__,torchvision.__version__,transformers.__version__)"
 result=subprocess.run([py,'-c',code],capture_output=True,text=True);print(result.stdout,flush=True);print(result.stderr,flush=True);result.check_returncode()
 old=json.loads((root/'runtime_status.json').read_text());(root/'runtime_setup_initial_status.json').write_text(json.dumps(old,indent=2))
 (root/'runtime_status.json').write_text(json.dumps(dict(stage='CPU_IMPORTS_PASSED_GPU_NOT_TESTED',versions=result.stdout.strip(),python=py,shared_environment_modified=False,initial_setup_status=old['stage']),indent=2))
 subprocess.run([py,'-m','pip','freeze'],stdout=(root/'runtime_freeze.txt').open('w'),check=True)
 if (root/'asset_probe').exists() or (root/'shared_sam3_masks_r1').exists():raise FileExistsError('Existing GPU input result; never replay automatically')
 GPU_stage('asset_inference_probe_r12.py','actual_SAM3_and_DINOv3_GPU_inference')
 if json.loads((root/'asset_probe/status.json').read_text())['status']!='SAM3_AND_DINOV3_REAL_GPU_INFERENCE_PASSED':raise RuntimeError('Asset inference not passed')
 GPU_stage('shared_sam3_masks_r12.py','SAM3_train46_native_tracking_and_3_isolated_QUERY')
 if not (root/'shared_sam3_masks_r1/summary.json').exists():raise RuntimeError('Incomplete SAM3 bundle')
 write('WAITING_ROOT_SAM3_MASK_REVIEW_NO_GPU_HELD',actual_native_frames=49,Gaussian_training_started=False,continue_command='root inspects RGB/depth masks then writes shared_SAM3_root_review.json and builds same inputs for finite A/B/C controls')
except BaseException as exc:write('FAILED_ENGINEERING',error=repr(exc),no_method_verdict=True,Gaussian_training_started=False);raise

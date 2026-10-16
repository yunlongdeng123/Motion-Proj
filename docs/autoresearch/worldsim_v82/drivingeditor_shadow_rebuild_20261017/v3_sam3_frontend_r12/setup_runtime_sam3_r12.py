"""A task-local runtime; never install into the shared training environment."""
from pathlib import Path
import sys,subprocess,json,time,os
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
root=R/'ownership_controls_r12_sam3';root.mkdir(exist_ok=True)
runtime=root/'runtime_sam3_dinov3';status=root/'runtime_status.json'
def write(stage,**kw):status.write_text(json.dumps(dict(stage=stage,time=time.time(),shared_environment_modified=False,**kw),indent=2))
try:
 if runtime.exists():raise FileExistsError('Do not recreate runtime; inspect existing installation first')
 write('CREATING_TASK_LOCAL_VENV')
 subprocess.run(['/root/autodl-tmp/envs/worldsim-v77-sam3/bin/python','-m','venv',str(runtime)],check=True)
 py=str(runtime/'bin/python');write('INSTALLING_PINNED_TORCH')
 subprocess.run([py,'-m','pip','install','--disable-pip-version-check','--no-cache-dir','--timeout','60','--retries','2','torch==2.7.1','torchvision==0.22.1','--index-url','https://download.pytorch.org/whl/cu126'],check=True)
 write('INSTALLING_PINNED_PREPROCESSING_LIBRARIES')
 subprocess.run([py,'-m','pip','install','--disable-pip-version-check','--no-cache-dir','--timeout','60','--retries','2','numpy==1.26.4','scipy==1.15.3','opencv-python-headless==4.11.0.86','transformers==4.56.2','timm==1.0.30','ftfy==6.1.1','iopath==0.1.10','einops==0.8.1','pycocotools==2.0.11','av==15.1.0','decord==0.6.0','setuptools==75.8.0','--index-url','https://mirrors.aliyun.com/pypi/simple'],check=True)
 write('VERIFYING_CPU_IMPORTS')
 code="import torch,torchvision,transformers;from transformers import AutoModel;import sys;sys.path.insert(0,'/root/autodl-tmp/idsplat/asset_sources/sam3');from sam3.model_builder import build_sam3_image_model;print(torch.__version__,torchvision.__version__,transformers.__version__)"
 result=subprocess.run([py,'-c',code],capture_output=True,text=True);print(result.stdout);print(result.stderr);result.check_returncode()
 subprocess.run([py,'-m','pip','freeze'],stdout=(root/'runtime_freeze.txt').open('w'),check=True)
 write('CPU_IMPORTS_PASSED_GPU_NOT_TESTED',versions=result.stdout.strip(),python=py)
except BaseException as exc:write('FAILED_ENGINEERING',error=repr(exc));raise

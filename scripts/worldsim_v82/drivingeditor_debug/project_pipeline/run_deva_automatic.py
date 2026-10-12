#!/usr/bin/env python3
"""Run the author's actual DEVA+SAM automatic instance segmentation settings."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

BASE=Path('/root/autodl-tmp/drivingeditor_repro')
REPO=BASE/'DrivingEditor-run/Tracking-Anything-with-DEVA'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            block=stream.read(8*1024**2)
            if not block: break
            h.update(block)
    return h.hexdigest()

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--scene',type=Path,default=BASE/'data/waymo_scenes/0017085_demo_000_020')
parser.add_argument('--output',type=Path)
parser.add_argument('--check-only',action='store_true')
args=parser.parse_args()
scene=args.scene.resolve()
assert scene.is_relative_to(BASE/'data/waymo_scenes'), 'Scene must be scoped to this reproduction'
image_dir=scene/'image/image_0'
output=args.output or scene/'mask/mask_0'
output=output.resolve()
assert output.is_relative_to(BASE), 'Mask output must stay within this reproduction'
manifest=json.loads((scene/'preparation_manifest.json').read_text())
images=sorted(image_dir.glob('*.png'))
expected=[row['id'] for row in manifest['frames']]
assert [p.stem for p in images]==expected and len(images)>=2
assert len(list(image_dir.iterdir()))==len(images), 'Author reader cannot ignore non-image files'
weights={'deva':BASE/'weights/DEVA-propagation.pth','sam':BASE/'weights/sam_vit_h_4b8939.pth'}
for name,path in weights.items():
    assert path.is_file() and path.stat().st_size>100_000_000, f'{name} completed checkpoint absent'
assert weights['sam'].stat().st_size==2564550879, 'Incomplete SAM ViT-H checkpoint'
for name in ('resnet18-5c106cde.pth','resnet50-19c8e357.pth'):
    assert (BASE/'cache/torch/hub/checkpoints'/name).is_file(), f'Required existing backbone cache absent: {name}'
os.environ['TORCH_HOME']=str(BASE/'cache/torch')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
sys.path.insert(0,str(REPO))
import torch
from deva.ext.automatic_sam import get_sam_model
from deva.inference.result_utils import ResultSaver
from deva.inference import consensus_automatic
command=[sys.executable,str(REPO/'demo/demo_automatic.py'),
    '--model',str(weights['deva']),'--SAM_CHECKPOINT_PATH',str(weights['sam']),
    '--SAM_ENCODER_VERSION','vit_h','--sam_variant','original',
    '--chunk_size','4','--img_path',str(image_dir),'--amp',
    '--temporal_setting','semionline','--size','480','--output',str(output),
    '--use_short_id','--suppress_small_objects','--SAM_PRED_IOU_THRESHOLD','0.7']
record={'status':'preflight_passed','scene':str(scene),'frames':len(images),'frame_ids':expected,
    'input_manifest_sha256':digest(scene/'preparation_manifest.json'),'command':command,
    'author_settings_source':'Tracking-Anything-with-DEVA/process_mask.py',
    'author_settings_source_sha256':digest(REPO/'process_mask.py'),
    'execution_source_sha256':digest(REPO/'demo/demo_automatic.py'),
    'sam_generator_sha256':digest(REPO/'deva/ext/SAM/automatic_mask_generator.py'),
    'memory_compatibility':{'change':'Accumulate completed boolean SAM masks on CPU; GPU boxes, scores and NMS unchanged; restore only retained masks to GPU.',
                            'sampling':'Original 64x64 prompts, batch64, original resolution and thresholds unchanged.',
                            'patch':str(BASE/'logs/deva_sam_cpu_accumulation.patch')},
    'torch_version':torch.__version__,'torch_home':os.environ['TORCH_HOME'],
    'cuda_allocator_configuration':os.environ.get('PYTORCH_CUDA_ALLOC_CONF','default'),
    'consensus_default_gurobi_importable':bool(consensus_automatic.use_gurobi),
    'limitations':['Automatic instance IDs are not a declaration that each instance is dynamic.',
                   'Dynamic actor selection, selected RGB images, and dynamic actor PLYs are pending.',
                   'This run uses the prepared FRONT camera; no fabricated masks or actor files.']}
print(json.dumps(record,indent=2),flush=True)
if args.check_only: raise SystemExit(0)
assert torch.cuda.is_available(), 'CUDA unavailable'
output.mkdir(parents=True,exist_ok=False)
record['checkpoint_sha256']={key:digest(path) for key,path in weights.items()}
record['started_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
record['status']='running'
record_path=output/'inference_manifest.json'
record_path.write_text(json.dumps(record,indent=2)+'\n')
try:
    runtime_log=output/'inference_runtime.log'
    with runtime_log.open('x') as stream:
        subprocess.run(command,cwd=REPO,check=True,stdout=stream,stderr=subprocess.STDOUT)
    log_text=runtime_log.read_text(errors='replace')
    fallback_logged='GurobiError, falling back to pulp' in log_text
    record['consensus_solver']={
        'default_gurobi_importable':bool(consensus_automatic.use_gurobi),
        'pulp_fallback_logged':fallback_logged,
        'effective_solver':'PuLP/CBC' if fallback_logged or not consensus_automatic.use_gurobi else 'Gurobi',
        'evidence':'Official consensus branch plus completed execution and its runtime log',
        'runtime_log':str(runtime_log)}
    subprocess.run([sys.executable,str(BASE/'scripts/validate_deva_masks.py'),
        '--scene',str(scene),'--output',str(output)],check=True)
except Exception as error:
    record.update(status='failed',error=str(error))
    record_path.write_text(json.dumps(record,indent=2)+'\n')
    raise
record.update(status='validated',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
record_path.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'status':'validated','frames':len(images),'output':str(output)}),flush=True)

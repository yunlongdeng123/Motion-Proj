"""Resolve existing build tools before running the isolated DGGT environment.

Use the retained gsplat SM86 cache (compatible with the current SM89 device).
No installed package or official model source is edited.
"""
from pathlib import Path
import json,os,runpy,sys

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_vehicle_joint_dggt_20261123_r8')
os.environ['PATH']='/root/autodl-tmp/envs/motionproj/bin:/usr/local/cuda/bin:'+os.environ.get('PATH','')
os.environ.setdefault('CUDA_HOME','/usr/local/cuda')
os.environ.setdefault('MAX_JOBS','2')
os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.6')
if sys.argv[1:] == ['--probe']:
    import torch
    from gsplat.cuda._backend import _C
    row=dict(complete=True,torch=torch.__version__,torch_CUDA=torch.version.cuda,
             CUDA_HOME=os.environ['CUDA_HOME'],build_arch=os.environ['TORCH_CUDA_ARCH_LIST'],
             backend_file=_C.__file__,ninja_resolved=True,new_model_forward_calls=0,new_raster_calls=0)
    (ROOT/'runtime/backend_probe.json').write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps(row))
else:
    script=Path(sys.argv[1]).resolve()
    assert script.parent == (ROOT/'code').resolve(), 'Only this task code is allowed'
    sys.argv=sys.argv[1:]
    runpy.run_path(str(script),run_name='__main__')

"""Actual GPU model/Adam/RNG restore proof, no optimization updates."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'infra'))
from priority import require_gpu_slot
from runtime import validate_spec, load_models
from restore_contract import verify_runtime_restore
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint',default=str(BASE/'queue_gmc_r1/gpu_smoke/checkpoint_000300.pt'))
parser.add_argument('--output',default=str(BASE/'metadata/runtime_restore_300_preflight_r1'))
parser.add_argument('--expected-step',type=int,default=300)
parser.add_argument('--spec',default=str(BASE/'specs/scene0008_repair.json'))
args=parser.parse_args()
spec,_=validate_spec(args.spec)
output=Path(args.output)
slot,admission=require_gpu_slot('/root/autodl-tmp/drivingeditor_repro/campaign/gpu_slot.lock')
try:
    if output.exists(): raise FileExistsError('Preserve prior restore attempt')
    output.mkdir(parents=True)
    import numpy as np
    import torch
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    path=Path(args.checkpoint)
    restored=torch.load(path,map_location='cuda',weights_only=False)
    if restored.get('schema')!='drivingeditor_rigid_scene_v1' or restored.get('spec')!=spec or restored.get('step')!=args.expected_step:
        raise ValueError('Actual checkpoint schema/spec/step mismatch')
    golden=torch.load(path,map_location='cpu',weights_only=False)
    _,_,static,instances,optimizer,environment,_=load_models(spec,output,restore=restored)
    sampler=random.Random(0)
    sampler.setstate(restored['sampler'])
    random.setstate(restored['python_rng']); np.random.set_state(restored['numpy_rng'])
    torch.set_rng_state(restored['torch_rng'].cpu())
    torch.cuda.set_rng_state_all([state.cpu() for state in restored['cuda_rng']])
    result=verify_runtime_restore(golden,static,instances,optimizer,environment,sampler)
    result.update(checkpoint=str(path),step=args.expected_step,admission=admission,optimizer_updates=0,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    (output/'receipt.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))
except BaseException as error:
    if output.exists(): (output/'failure.json').write_text(json.dumps({'error':repr(error),'assets_preserved':True}),encoding='utf-8')
    raise
finally:
    slot.close()

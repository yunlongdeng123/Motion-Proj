"""Locked 800+800 runs, each independently initialized from unchanged S2."""
from pathlib import Path
import argparse
import gc
import json
import subprocess
import sys
import torch

C=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_direct_teacher_20261116_r1')
R=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
sys.path.insert(0,str(R/'scripts/worldsim_v82'))
from drivingeditor_dynamic_edit.road_direct_teacher_training import setup,reload_check,file_digest

protocol=C/'protocol_locked.json'
if json.loads(protocol.read_text()).get('formal_authorized') is not True:
    raise RuntimeError('formal protocol is not authorized')
baseline=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_asset_20261116_r1/phase2/formal_r1/asset')
values=dict(manifest=baseline/'persistent/scene.json',proof=baseline/'persistent/proof.json',asset=baseline,
    r8_asset=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_asset_20261116_r1/asset_r8_guarded'),
    supervision=C/'supervision_r3',protocol=protocol)
common=[sys.executable,'-m','drivingeditor_dynamic_edit.road_direct_teacher_training','--mode','formal','--schedule','200,300,300']
for key,value in values.items():common+=['--'+key.replace('_','-'),str(value)]
for role in ('direct_C','matched_F_control'):
    output=C/role/'formal_r1'
    output.parent.mkdir(parents=True,exist_ok=True)
    command=common+['--role',role,'--output',str(output)]
    print(json.dumps(dict(starting=role,output=str(output))),flush=True)
    with (output.parent/'formal_r1.log').open('x') as log:
        subprocess.run(command,cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
    # Check each real stage checkpoint against its separately saved public asset.
    # No optimization or checkpoint mutation; all five cameras are diagnostic here.
    args=argparse.Namespace(**values,role=role)
    state=setup(args)
    for step,stage in ((200,'color'),(500,'coverage'),(800,'joint')):
        checkpoint=torch.load(output/f'checkpoint{step:06d}.pt',map_location='cpu')
        for path,digest in checkpoint['contract']['input_binding'].items():
            if file_digest(path)!=digest:raise RuntimeError('frozen input changed:'+path)
        state['model'].load_state_dict(checkpoint['model'])
        reload_check(state,output/f'stage_{stage}/asset',output/f'stage_{stage}/reload.json')
    del state
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(dict(completed=role,formal_updates=800,three_stage_public_loader_validation=True)),flush=True)
print(json.dumps(dict(completed=True,formal_updates_total=1600,engineering_updates_in_this_script=0)),flush=True)

"""Export the selected material, verify its real render path, and package evidence."""
from pathlib import Path
import argparse,gzip,json,shutil,sys,tarfile,time
import numpy as np
import torch
from colour_fit import Experiment,Colour,ROOT,C,ASSETS,PILOTS,write

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--evaluation',required=True);p.add_argument('--smoke',required=True);a=p.parse_args()
    dest=ROOT/'delivery';dest.mkdir(exist_ok=False)
    for name in ('assets','evidence','frames','code'):(dest/name).mkdir()
    e=Experiment();e.load(Path(a.checkpoint));mutable=('static_dc','road_world','env.base')
    payload=dict(format='frozen_m2_shared_colour_v1',step=e.step,parameters={k:e.model.state_dict()[k].detach().cpu() for k in mutable},
                 contract=e.contract,parent=e.rt.meta,m2_asset=str(ASSETS/'m2/complete_a6.npz'))
    torch.save(payload,dest/'assets/fixed_material.pt')
    decoded=torch.load(dest/'assets/fixed_material.pt',map_location='cpu',weights_only=False)
    replay=Colour(e.rt)
    replay.load_state_dict(decoded['parameters'],strict=False)
    checks=[]
    for raw in PILOTS:
        for delete in (False,True):
            first=e.model.draw(raw,delete);second=replay.draw(raw,delete)
            values={k:float((first[k]-second[k]).abs().max()) for k in ('rgb','alpha','depth_sum','component_alpha_T')}
            assert max(values.values())==0,values
            checks.append(dict(raw=raw,delete=delete,maxabs=values))
    assert e.model.unchanged_geometry() and replay.unchanged_geometry()
    write(dest/'evidence/material_reload.json',dict(status='PASS_EXACT_EXPORTED_MATERIAL_RENDER',rows=checks))
    for f in Path(a.evaluation).glob('*.jpg'):shutil.copy2(f,dest/'frames'/f.name)
    shutil.copy2(Path(a.evaluation)/'metrics.json',dest/'evidence/final_metrics.json')
    shutil.copy2(Path(a.checkpoint).parent/'returned.json',dest/'evidence/formal_returned.json')
    shutil.copy2(Path(a.checkpoint).parent/'contract.json',dest/'evidence/contract.json')
    smoke=Path(a.smoke)
    shutil.copy2(smoke/'returned.json',dest/'evidence/smoke_returned.json')
    for folder in sorted(smoke.glob('eval*')):
        if (folder/'metrics.json').is_file():shutil.copy2(folder/'metrics.json',dest/'evidence'/f'smoke_{folder.name}_metrics.json')
    final_smoke=list(smoke.glob('final*/metrics.json'));assert len(final_smoke)==1
    shutil.copy2(final_smoke[0],dest/'evidence/smoke_full49_metrics.json')
    shutil.copy2(ROOT/'resume_check_r1/restore_audit.json',dest/'evidence/restore_audit.json')
    shutil.copy2(ROOT/'resume_check_r1/returned.json',dest/'evidence/resume_returned.json')
    for src in (ROOT/'audit',ROOT/'env_swap_probe'):
        shutil.copy2(src/'metrics.json',dest/'evidence'/f'{src.name}_metrics.json')
    shutil.copy2(ROOT/'audit/113_comparison.jpg',dest/'evidence/history_comparison.jpg')
    for f in (ROOT/'source').glob('*.py'):shutil.copy2(f,dest/'code'/f.name)
    versions=ROOT/'executed_versions'
    if versions.exists():shutil.copytree(versions,dest/'code/executed_versions')
    shutil.copy2(ASSETS/'m2/complete_a6.npz',dest/'assets/m2_complete_a6.npz')
    write(dest/'evidence/inventory.json',dict(remote_run=str(ROOT),selected_checkpoint=str(a.checkpoint),selected_evaluation=str(a.evaluation),
          required_frozen_parents=e.rt.meta,unchanged_M2_asset=str(ASSETS/'m2/complete_a6.npz'),
          formal_added_updates=e.step,smoke_separate=True,restoration_separate=True,power_action='none',subagents=0,
          deployment_scope='Shared 3-D/background material correction over the preserved M2 scene; no image-space colour filtering.'))
    # Full optimizer/RNG state remains independently recoverable; compression
    # changes storage only, never model values or training state.
    with open(a.checkpoint,'rb') as src,gzip.open(dest/'assets/full_training_state.pt.gz','wb',compresslevel=3) as dst:
        shutil.copyfileobj(src,dst,8*1024*1024)
    for name in ('fixed_material.pt',):
        with open(dest/'assets'/name,'rb') as src,gzip.open(dest/'assets'/(name+'.gz'),'wb',compresslevel=3) as dst:
            shutil.copyfileobj(src,dst,8*1024*1024)
    with tarfile.open(ROOT/'evidence_delivery.tar.gz','w:gz') as archive:
        for f in dest.rglob('*'):
            if f.is_file() and f.parts[-2]!='assets':archive.add(f,arcname='scene008_blue_fix/'+str(f.relative_to(dest)))
        archive.add(dest/'assets/m2_complete_a6.npz',arcname='scene008_blue_fix/assets/m2_complete_a6.npz')
    print(json.dumps(dict(status='PASS_EXPORT_AND_EXACT_RELOAD',files=sum(x.is_file() for x in dest.rglob('*')),
          material_bytes=(dest/'assets/fixed_material.pt.gz').stat().st_size,
          state_bytes=(dest/'assets/full_training_state.pt.gz').stat().st_size)),flush=True)
if __name__=='__main__':main()

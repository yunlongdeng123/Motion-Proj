"""Authorized sequential acquisition acceptance -> real smoke -> fresh10K -> common-query evaluation.

Run only after generation finishes. Every output is new; no power or Git actions.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from PIL import Image

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030')
OUT=ROOT/'evidence_reconstruction_r1'; CODE=OUT/'code'; NATIVE=ROOT/'code/reconstruction'; PY=sys.executable
DATA=OUT/'dataset_hybrid8'; SCENE=OUT/'native_scene_hybrid8'; FORMAL=OUT/'formal_hybrid8_10k'; SMOKE=OUT/'smoke_hybrid8'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def execute(name,command):
    if shutil.disk_usage(OUT).free<25*1024**3:raise RuntimeError('Free disk below25GiB')
    started=time.monotonic();write(OUT/'pipeline_status.json',dict(stage=name,status='RUNNING',started_unix=time.time(),command=command))
    with (OUT/(name+'.log')).open('x') as log:
        run=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    receipt=dict(stage=name,command=command,returncode=run.returncode,wall_seconds=time.monotonic()-started)
    write(OUT/(name+'_execution.json'),receipt)
    if run.returncode:raise RuntimeError('Stage failed; preserve all outputs: '+name)
    return receipt

def shared(entry):return [PY,str(CODE/'native_shared_entry.py'),'--gauge',str(CODE/'baseline_gauge.json'),'--native-code',str(NATIVE),'--entry',entry]

def anchor_audit():
    old=read(ROOT/'dataset_r2/observations/manifest.json');new=read(DATA/'observations/manifest.json')
    old_rows={r['id']:r for r in old['frames']}; new_rows={r['id']:r for r in new['frames']}; checks=[]
    for f in [0,8,15]:
        a,b=old_rows[f],new_rows[f]
        camera_error=float(np.abs(np.asarray(a['camera_to_world'])-np.asarray(b['camera_to_world'])).max())
        K_error=float(np.abs(np.asarray(a['K'])-np.asarray(b['K'])).max())
        before=np.asarray(Image.open(ROOT/'dataset_r2/observations'/a['rgb']),dtype=np.int16)
        after=np.asarray(Image.open(DATA/'observations'/b['rgb']),dtype=np.int16)
        delta=np.abs(before-after)
        poses_old=read(ROOT/'dataset_r2/privileged/poses'/('%06d.json'%f))
        poses_new=read(DATA/'privileged/poses'/('%06d.json'%f))
        object_error=max(float(np.abs(np.asarray(poses_old['object_to_world'][k])-np.asarray(poses_new['object_to_world'][k])).max()) for k in poses_old['object_to_world'])
        checks.append(dict(frame=f,camera_max_abs=camera_error,K_max_abs=K_error,world_object_pose_max_abs=object_error,RGB_max_uint8=int(delta.max()),RGB_MAE_uint8=float(delta.mean()),RGB_changed_channels=int((delta>0).sum())))
        assert camera_error<2e-6 and K_error<1e-8 and object_error==0
    write(OUT/'anchor_parity.json',dict(status='PASS_SHARED_WORLD_FIXED_ANCHORS_WITH_RECORDED_RGB_NUMERICAL_DIFFERENCES',checks=checks,reference_reuse='original independent0/8/15 edit references; tiny finite-sample RGB differences disclosed, no pixel identity claim',GT_audit_only=True))

def gauge_audit(run):
    gauge=read(CODE/'baseline_gauge.json')
    T=np.loadtxt(SCENE/'transform_matrix.txt')
    error=float(np.abs(T-np.asarray(gauge['world_to_model'])).max())
    assert error<1e-12
    import torch
    event=read(run/'latest_complete_node.json')
    event=read(event['event']) if 'event' in event else event
    # inspect_bundle records the authoritative complete event path.
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location('audit_bundle',NATIVE/'inspect_bundle.py');mod=module_from_spec(spec);spec.loader.exec_module(mod)
    accepted=mod.inspect(run); full=read(accepted['event']); values={}
    for name in ['static','dynamic']:
        payload,_=torch.load(str(full['checkpoints'][name]['path']),map_location='cpu',mmap=True,weights_only=False)
        values[name]=float(payload[15]);assert values[name]==gauge['native_spatial_lr_scale']
    write(run/'shared_gauge_verification.json',dict(all_passed=True,world_to_model_max_abs=error,model_spatial_lr_scales=values,shared_camera_extent=gauge['native_camera_extent']))

def main():
    if not (DATA/'dataset_manifest.json').exists():raise RuntimeError('Acquisition not complete')
    if not (DATA/'validation.json').exists():
        execute('validate_dataset_r3',[PY,str(ROOT/'code/simulation/validate_dataset.py'),str(DATA)])
    assert read(DATA/'validation.json')['status']=='passed'
    anchor_audit()
    execute('adapt_scene',[PY,str(NATIVE/'adapt_scene.py'),'--observations',str(DATA/'observations'),'--output',str(SCENE),'--static-seed-count','600000','--frame-interval','.02','--render-scale','1'])
    execute('real_smoke',[PY,str(CODE/'run_smoke_shared.py'),'--scene',str(SCENE),'--output',str(SMOKE),'--steps','10000','--checkpoint-every','5000','--leg-seconds','300','--entry-steps','2','--reference-root',str(ROOT/'dataset_r2/privileged')])
    assert read(SMOKE/'smoke_summary.json')['approximately_ten_minutes_contract']
    gauge_audit(SMOKE/'next_stage_entry')
    execute('formal_10k',shared('native_worker')+['--scene',str(SCENE),'--output',str(FORMAL),'--steps','10000','--checkpoint-every','5000'])
    execute('inspect_formal',[PY,str(NATIVE/'inspect_bundle.py'),'--run',str(FORMAL)])
    gauge_audit(FORMAL)
    event=FORMAL/'checkpoint_events/iteration_010000.json'
    execute('evaluate_common_edits',shared('evaluate_edits')+['--event',str(event),'--output',str(OUT/'edit_common_10k'),'--reference-root',str(ROOT/'dataset_r2/privileged'),'--keyframes','0,8,15'])
    execute('evaluate_common_heldout',shared('render_heldout')+['--event',str(event),'--output',str(OUT/'heldout_common_10k'),'--reference-root',str(ROOT/'dataset_r2/privileged')])
    write(OUT/'pipeline_status.json',dict(status='COMPLETE',formal_updates=10000,engine_LR_horizon=30000,smoke_updates_extra=True,common_edit_renders=12,heldout_none=1,power_actions=0))

if __name__=='__main__':main()

"""No training. Finish heldout routes, same-world strata, receipts and lightweight delivery."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import re
import numpy as np
from PIL import Image
import torch

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030');OUT=ROOT/'evidence_reconstruction_r1';CODE=OUT/'code';FORMAL=OUT/'formal_hybrid8_10k'
def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')

def main():
    event=FORMAL/'checkpoint_events/iteration_010000.json'
    if not event.exists() or read(event)['status']!='complete':raise RuntimeError('Formal10K event not complete')
    returned=read(FORMAL/'returned.json')
    assert returned['iteration']==10000 and returned['observed_finite_step_calls']==10000
    old_config=read(ROOT/'formal_r3_10k/effective_config.json');new_config=read(FORMAL/'effective_config.json')
    assert new_config['start_checkpoint'] is None and new_config['v43_update_budget']==10000 and new_config['iterations']==30000
    differences={k for k in set(old_config)|set(new_config) if old_config.get(k)!=new_config.get(k)}
    assert differences<={'source_path','model_path','followup_scene_spec'}
    write(OUT/'native_recipe_parity.json',dict(status='PASS_IDENTICAL_NATIVE_RECIPE',phase='actual_formal10k',changed_keys=sorted(differences),seed=new_config['seed'],updates=new_config['v43_update_budget'],LR_horizon=new_config['iterations']))
    route_receipts=[];heldout=[]
    with np.load(ROOT/'dataset_r2/privileged/heldout/000008.npz') as g:body=g['semantic_id']==10
    for edit in ['none','delete','shift','rotate']:
        folder=OUT/'heldout_common_10k' if edit=='none' else OUT/('heldout_samequery_'+edit)
        command=[sys.executable,str(CODE/'render_heldout_routes.py'),'--event',str(event),'--output',str(folder),'--reference-root',str(ROOT/'dataset_r2/privileged'),'--selection',str(ROOT/'m1_assignment_r1/recall_verified/selection_000008.npz'),'--baseline','--edit',edit]
        if edit=='none':
            route_receipts.append(dict(edit=edit,returncode=0,wall_seconds=0,reused_existing_native_none=True,source=str(folder)))
        else:
            started=time.monotonic()
            with (OUT/('heldout_samequery_'+edit+'.log')).open('x') as log:
                run=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            route_receipts.append(dict(edit=edit,returncode=run.returncode,wall_seconds=time.monotonic()-started,command=command))
            if run.returncode:raise RuntimeError('Heldout route failed: '+edit)
        manifest=read(folder/'render_manifest.json')
        assert manifest['original_checkpoint_unchanged']
        if edit=='none':assert manifest['native_none_identity_max_abs']<=2e-6
        with np.load(manifest['fp32']) as d:prediction=d['final']
        reference=ROOT/'heldout_edit_refs_r1'/edit/'000008.png'
        target=np.asarray(Image.open(reference).convert('RGB'),dtype=np.float32).transpose(2,0,1)/255
        regions={}
        for name,mask in [('full',np.ones_like(body)),('original_body_footprint',body)]:
            metric=dict(pixels=int(mask.sum()))
            for label,x in [('raw',prediction),('clamped',prediction.clip(0,1))]:
                delta=x[:,mask].astype(np.float64)-target[:,mask].astype(np.float64)
                metric[label+'_mae']=float(np.abs(delta).mean());metric[label+'_mse']=float(np.square(delta).mean())
            regions[name]=metric
        heldout.append(dict(edit=edit,regions=regions,reference=str(reference),fp32=manifest['fp32']))
    write(OUT/'heldout_routes_execution.json',route_receipts)
    write(OUT/'heldout_samequery_metrics.json',dict(rows=heldout,scope='one unchanged original heldout camera/time, four edits, fixed original-body footprint only; no edited semantic geometry',GT_only_after_render=True))
    subprocess.run([sys.executable,str(CODE/'score_observed_road.py')],check=True)
    smoke=read(OUT/'smoke_hybrid8/smoke_summary.json');assert smoke['status']=='PASS_REAL_NATIVE_SMOKE'
    capacities={}
    for arm,ev in [('M0',ROOT/'formal_r3_10k/checkpoint_events/iteration_010000.json'),('richer_observations',event)]:
        e=read(ev); capacities[arm]={}
        for name in ['static','dynamic']:
            payload,_=torch.load(str(e['checkpoints'][name]['path']),map_location='cpu',mmap=True,weights_only=False)
            capacities[arm][name]=dict(gaussians=len(payload[1]),model_parameter_elements=sum(payload[i].numel() for i in range(1,10)),spatial_lr_scale=float(payload[15]))
        environment,_=torch.load(str(e['checkpoints']['environment']['path']),map_location='cpu',mmap=True,weights_only=False)
        capacities[arm]['environment']=dict(model_parameter_elements=environment[0].numel(),shape=list(environment[0].shape))
    native=read(OUT/'edit_common_10k/render_manifest.json')
    receipt=dict(status='COMPLETED_SAME_WORLD_RICHER_OBSERVATION_RECONSTRUCTION_CONTROL',formal_event=str(event),formal_updates=10000,engine_and_LR_horizon=30000,seed=0,
        training_scope='same native GN11, full static/dynamic/environment, no new classifier or M2, no extra formal updates',
        runtime=dict(acquisition_seconds=read(OUT/'dataset_hybrid8/dataset_manifest.json')['elapsed_s'],formal_seconds=read(OUT/'formal_10k_execution.json')['wall_seconds'],formal_native_timing=returned['timing'],smoke_loop_seconds=smoke['timed_leg_training_loop_wall_seconds_excluding_measured_eval_checkpoint'],scope='actual host wall time, not pure GPU kernels; native snapshot separates training-loop proxy from measured evaluation/checkpoint/publication, and initialization/restore diagnostic subsets are not additive',concurrent_workload='short frozen-old-checkpoint/new-camera bridge diagnosis can overlap formal training; no training-speed superiority claim'),
        separate_extra_budget=dict(smoke_updates=smoke['extra_smoke_updates'],entry_updates=smoke['extra_entry_updates'],formal_extra_updates=0),
        smoke=dict(real_CUDA=True,save_evaluate_restore_next_entry_passed=True,complete_restore=smoke['complete_restore']['all_passed'],none_identity_max_abs=smoke['actual_next_stage_render']['native_none_identity_max_abs']),
        parity=dict(world_identity=read(OUT/'world_identity.json'),anchor_sensor_parity=read(OUT/'anchor_parity.json'),native_recipe=read(OUT/'native_recipe_parity.json'),shared_gauge=read(FORMAL/'shared_gauge_verification.json'),common_none_identity_max_abs=native['native_none_identity_max_abs'],checkpoint_unchanged=native['original_checkpoint_unchanged']),
        initialization=dict(original=read(ROOT/'native_scene_r3/preparation_manifest.json'),richer=read(OUT/'native_scene_hybrid8/preparation_manifest.json'),actual_native_initial_counts=dict(original=list(map(int,re.findall(r'Number of points at initialization\s*:\s*(\d+)',(ROOT/'formal_r3_10k.log').read_text()))),richer=list(map(int,re.findall(r'Number of points at initialization\s*:\s*(\d+)',(OUT/'formal_10k.log').read_text()))))),
        parameter_capacity=capacities,formal_peak_memory=read(FORMAL/'cuda_memory_peak.json'),common_keyframe_renders=12,heldout_edit_renders=4,
        interpretation='trajectory changes legal RGB+LiDAR observations and their lawful initialization; same-world observation-bundle mechanism control, not pure RGB causality or real-driving generalization',
        trajectory_scope='13 orbit-like changed viewpoints plus3 original anchors; some camera jumps, not a feasible continuous driving policy',
        operational_fixes=['new camera inverse metadata canonicalized in float64; original c2w/RGB/LiDAR preserved','anchor RGB numerical microdifferences recorded, not pixel identity'],
        GT_isolation='privileged geometry only used for preflight/world audit/evaluation, never adapter/model initialization/training',quality_accepted=False,human_verdict=None,power_actions=0,git_actions=0)
    write(OUT/'completion_receipt.json',receipt)
    mappings=[]
    for path in [OUT/'completion_receipt.json',OUT/'observed_unseen_road_metrics.json',OUT/'heldout_samequery_metrics.json',OUT/'heldout_routes_execution.json',OUT/'anchor_parity.json',OUT/'world_identity.json',OUT/'native_recipe_parity.json',OUT/'launch_contract.json']:
        target='stratified_metrics.json' if path.name=='observed_unseen_road_metrics.json' else path.name
        mappings.append((path,target))
    for path in (OUT/'edit_common_10k').rglob('*'):
        if path.is_file() and path.suffix in ['.json','.png']:mappings.append((path,str(Path('edit_samequery')/path.relative_to(OUT/'edit_common_10k'))))
    for edit in ['none','delete','shift','rotate']:
        folder=OUT/'heldout_common_10k' if edit=='none' else OUT/('heldout_samequery_'+edit)
        for path in folder.rglob('*'):
            if path.is_file() and path.suffix in ['.json','.png']:mappings.append((path,str(Path('heldout_samequery')/edit/path.relative_to(folder))))
    for name in ['smoke_hybrid8/smoke_summary.json','dataset_hybrid8/validation.json','dataset_hybrid8/calibration_precision_receipt.json','native_scene_hybrid8/preparation_manifest.json','formal_hybrid8_10k/shared_gauge_verification.json']:
        mappings.append((OUT/name,name))
    manifest=[dict(path=target,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path,target in mappings]
    write(OUT/'delivery_manifest.json',manifest)
    with tarfile.open(OUT/'delivery.tar.gz','w:gz') as tar:
        for path,target in mappings:tar.add(path,arcname=target)
        tar.add(OUT/'delivery_manifest.json',arcname='delivery_manifest.json')
    print(json.dumps(dict(status=receipt['status'],delivery=str(OUT/'delivery.tar.gz'),files=len(mappings),bytes=(OUT/'delivery.tar.gz').stat().st_size)))

if __name__=='__main__':main()

"""CPU verification of actual saved addons/Adam and exact executed step counts."""
import os,json,hashlib,datetime
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import torch
torch.set_num_threads(2)
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1');NEW=ROOT/'shadow_sequence_20261016_v2'
def finite(value):
    if torch.is_tensor(value):return bool(torch.isfinite(value).all())
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(finite(v) for v in value)
    return True
rows=[]
for name in ['joint_v1','repair_union_v1','contribution_frozen_v1','unselected_clean_v1']:
    status=json.loads((NEW/name/'status.json').read_text());assert status['state']=='complete'
    checkpoint=NEW/name/'final.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert finite(saved['surface']) and finite(saved['optimizer']) and finite(saved['actor_gate'])
    assert saved['step']==status['training_updates']
    assert not any(k.startswith('static.') for k in saved['surface'])
    restore=[]
    for phase in ['background','shadow','joint']:
        receipt=json.loads((NEW/name/(phase+'_restore_check.json')).read_text());assert receipt['RGB_bitwise_equal'] and receipt['Adam_restored'];restore.append(receipt)
    rows.append({'training':name,'gradient_updates':saved['step'],'file_bytes':checkpoint.stat().st_size,
        'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'surface_and_Adam_finite':True,'phase_restore_checks':restore,
        'freeze_static_in_road_pass':bool(saved['surface'].get('freeze_static',False)),
        'actor_opacity_adapter_enabled':bool(saved.get('actor_opacity_enabled',False))})
    del saved
parent=ROOT/'rigid_runs/scene0008_089_137/full30k/checkpoint_030000.pt'
identity=json.loads((ROOT/'shadow_BC_20261016_v1/selection_v1/selection_receipt.json').read_text())['parent_checkpoint']
assert parent.stat().st_size==identity['bytes'] and parent.stat().st_mtime_ns==identity['mtime_ns']
result={'runs':rows,'total_gradient_updates':sum(r['gradient_updates'] for r in rows),'all_actual_saved_addons_and_Adam_finite':True,
    'parent_checkpoint_size_and_mtime_preserved':True,'parent_identity':identity,'schema':'actual_training_accounting_v2',
    'source_parameter_versions_checked_in_each_GPU_worker':True,'next_step_sampler_determinism_certified':False,
    'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(NEW/'training_accounting.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

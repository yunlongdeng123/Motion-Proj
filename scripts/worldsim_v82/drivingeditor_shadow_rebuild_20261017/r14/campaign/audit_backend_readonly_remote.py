"""Read actual baseline and paused-R14 contracts on CPU, never start training."""
from pathlib import Path
import json,subprocess,torch,numpy as np
torch.set_num_threads(2)
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1')
R=C/'shadow_rebuild_20261017_v3';D=R/'ownership_rebuild_r14';B=C/'rigid_runs/scene0008_089_137/full30k'
def read(p):return json.loads(p.read_text()) if p.exists() else None
def shape(x):
    if torch.is_tensor(x):return dict(shape=list(x.shape),dtype=str(x.dtype))
    if isinstance(x,dict):return {k:shape(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [shape(v) for v in x]
    return x
value=dict(baseline_files=[p.name for p in B.iterdir()],pause=read(D/'user_pause_backend_audit/receipt.json'),GPU=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip())
for name in ['effective_config.json','training_policy.json','status.json','runtime_contract.json','resolved_spec.json','spec.json']:
    v=read(B/name)
    if v is not None:value['baseline_'+name]=v
s=torch.load(B/'checkpoint_030000.pt',map_location='cpu')
value['baseline_checkpoint_top_keys']=list(s)
value['baseline_spec']=s['spec'];value['baseline_training_policy']=s.get('training_policy')
value['baseline_background_shapes']=shape(s['background'])
value['baseline_environment_shapes']=shape(s['environment'])
value['baseline_instance_shapes']={k:shape(v) for k,v in s['instances'].items() if k not in ['optimizer','optimizer_state']}
for key in ['reference_config','environment_checkpoint','background_checkpoint']:
    p=Path(s['spec'][key]);value['baseline_'+key+'_path']=str(p)
    if p.suffix=='.json':value['baseline_'+key+'_content']=read(p)
value['baseline_runtime_dir_candidates']=[str(p) for p in C.iterdir() if 'rigid' in p.name]
del s
paused=torch.load(D/'user_pause_backend_audit/last_atomic_checkpoint.pt',map_location='cpu')
state=paused['state'];owner=state['field.owner'];road=state['field.road'];conf=state['field.reference_confidence']
value['R14_checkpoint']=dict(step=paused['global_step'],total=len(owner),static=int((owner<0).sum()),actor=int((owner>=0).sum()),road=int(road.sum()),road_valid_reference=int((road&(conf>0)).sum()),sky=shape(state['sky_logit']),config=paused['config'])
value['R14_reference_refresh']=read(D/'main/reference_refresh.json')
with np.load(D/'idsplat_frontend/data_IDSplat/seed.npz') as f:
    value['R14_seed_keys']=f.files
    if 'owner' in f:value['R14_initial_counts']=dict(total=len(f['owner']),static=int((f['owner']<0).sum()),actor=int((f['owner']>=0).sum()))
value['legacy18k_runtime']=read(C/'shadow_sequence_20261016_v2/unselected_clean_v1/runtime_contract.json')
print(json.dumps(value,indent=2,default=str))

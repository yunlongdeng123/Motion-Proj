"""Run after corrected raw rebuild; keep pending measurements separate from owners."""
from pathlib import Path
import os,sys,json,subprocess,shutil
import numpy as np
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');D=R/'ownership_rebuild_r14'
sys.path.insert(0,str(R/'implementation_r14_final/shadow_rebuild_v3'))
from shadow_rebuild.config import Config
from shadow_rebuild.data import Dataset
from shadow_rebuild.train import preflight,write
py='/root/autodl-tmp/envs/drivingeditor/bin/python'
assert (D/'initialization_summary.json').exists()
subprocess.run([py,str(D/'build_IDSplat_r14.py')],check=True,cwd=R)
parts=[]
for a in [5,6,12,18,25,26,37]:
    s=np.load(D/'idsplat_frontend/pointsets'/f'a{a:02d}_all_candidates.npz');take=~s['accepted'];p={k:s[k][take] for k in ['xyz','rgb','source_frame','source_raw_row','source_world','source_candidate_owner']}
    n=int(take.sum());p['identity_probability']=np.tile([.1,.1,.8],(n,1)).astype(np.float32);p['identity_evidence_status']=np.zeros(n,np.int64);parts.append(p)
shutil.move(D/'uncertain_candidate_pool.npz',D/'unused_B_semantic_pending_pool.npz')
np.savez_compressed(D/'uncertain_candidate_pool.npz',**{k:np.concatenate([p[k] for p in parts]) for k in parts[0]})
cfg=Config.load(D/'common_config.json');cfg.geometry_steps=13000;cfg.material_steps=7000;cfg.joint_steps=2400;cfg.continuous_ownership=True;cfg.train_opacity_stage2=True;cfg.material_opacity_lr_scale=.1;cfg.densify_until=12000;cfg.validate()
write(D/'config.json',__import__('dataclasses').asdict(cfg));dataset=Dataset(D/'idsplat_frontend/data_IDSplat','cpu',True);receipt=preflight(dataset,cfg)
receipt.update(mask_valid_pixelwise=True,IDSplat_pending_pool=sum(len(p['xyz']) for p in parts),old_18k_or_GN11_learned_state_used=False,initialization_proxy_is_diagnostic_not_stop=True)
write(D/'input_preflight.json',receipt);write(D/'pipeline_state.json',dict(status='INPUTS_READY_GPU_PROBE_REQUIRED',training_updates=0));print(json.dumps(receipt),flush=True)

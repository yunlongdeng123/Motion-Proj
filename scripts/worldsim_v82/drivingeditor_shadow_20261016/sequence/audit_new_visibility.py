import os,json
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1');NEW=ROOT/'shadow_sequence_20261016_v2'
frames=json.loads((ROOT/'rigid_post30k/scene0008_089_137/evaluation/scene_evaluation_manifest.json').read_text())['streams']['observed']['frames']
with np.load(NEW/'surface_guarded_data/surface.npz') as f:q=f['xyz'];conf=f['confidence']
with np.load(NEW/'guard_refinement_v2/overrides.npz') as f:free=f['free'];shadow=f['shadow']
rows=[]
for i,row in enumerate(frames):
    homogeneous=np.c_[q,np.ones(len(q))]@np.array(row['camera']['full_proj_transform']);front=homogeneous[:,3]>0
    v=free[i]&(conf>.9)
    rows.append({'frame':row['frame_id'],'current_valid_samples':int(v.sum()),'behind_camera_in_current_valid':int((v&~front).sum()),'behind_shadow_pseudo_samples':int((shadow[i]&~front).sum())})
result={'per_frame':rows,'total_behind_in_valid':sum(r['behind_camera_in_current_valid'] for r in rows),'total_behind_shadow':sum(r['behind_shadow_pseudo_samples'] for r in rows)}
(NEW/'new_visibility_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='per_frame'}))

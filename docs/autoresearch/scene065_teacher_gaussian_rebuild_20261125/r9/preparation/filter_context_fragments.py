"""Expand the persistent local takeover to the fixed real-visible BG domain."""
from pathlib import Path
import json,hashlib,sys,numpy as np
R=Path(__file__).resolve().parent
R7=R.parent/'road_teacher_region_rebuild_20261125_r7'
sys.path.insert(0,str(R7/'code'))
from teacher_gaussian_runtime import TeacherGaussianRuntime
from prepare_region_retirement_v5 import sample,in_box
def load(p):
    with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
rt=TeacherGaussianRuntime(R7/'config/runtime.json',R/'fragment_filter_runtime')
manifest=json.loads((R/'manifest.json').read_text());old=load(R7/'bounded_context_v2/retained_fragments.npz');n=len(old['parent_global_row_ids']);remove=np.zeros(n,bool);rows=[]
inv=np.linalg.inv(rt.base.transform)
for row in manifest['frames']:
    raw=int(row['raw']);support=load(row['fixed_masks']['path'])['bg_geometry_scope'];means=rt.context[raw]['means3D'][-n:].cpu().numpy().astype(np.float64)
    h=rt.base.camera_headers[raw];q=means@h['W'][:3,:3].T+h['W'][:3,3]
    with np.errstate(divide='ignore',invalid='ignore'):uv=q[:,:2]/q[:,2:3]*[h['fx'],h['fy']]+[h['cx'],h['cy']]
    hit=sample(support,uv,q[:,2]);world=means@inv[:3,:3].T+inv[:3,3];body=np.zeros(n,bool)
    obs=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs/observations')/f'{raw:03}.json'
    for box in json.loads(obs.read_text())['boxes']:
        if box['track']!='NTjq-OhZ_97OMJ4xwJAxXg':body|=in_box(world,box,.03)
    chosen=hit&~body;remove|=chosen;rows.append(dict(raw=raw,BG_visible_children=int(hit.sum()),protected_body_children=int((hit&body).sum()),retired_children=int(chosen.sum())))
keep=~remove;out=R/'initialization/retained_fragments_background_v1.npz';np.savez_compressed(out,**{k:v[keep]for k,v in old.items()})
receipt=dict(complete=True,source_rows=n,retained_rows=int(keep.sum()),retired_rows=int(remove.sum()),rows=rows,optimizer_updates=0,
    recipe=dict(path=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest()),body_core_preserved=True,render_time_image_gate=False)
(R/'fragment_filter_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True);rt.assert_fixed()

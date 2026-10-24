"""Export actual candidate identities for the CPU gate audit; no model updates."""
import argparse, json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);root=p.parse_args().audit
s=json.loads((root/'summary.json').read_text());bundle=Path(s['bundle']);m=json.loads(bundle.read_text())
u=np.load(bundle.parent/m['units']);a=np.load(root/'audit_arrays.npz')
actor_masks={str(actor):np.isin(np.arange(len(u['world'])),np.unique(u['query_unit_id'][u['query_actor']==actor])) for actor in m['actors']}
np.savez_compressed(root/'actor_units.npz',**actor_masks)
gates={k.removeprefix('gate_'):u[k] for k in u.files if k.startswith('gate_')}
old=u['base_weight']>0;cos=[];uv=[];zz=[]
for fid in m['frame_order']:
 f=m['frames'][fid];w=np.asarray(f['world_to_camera']);k=np.asarray(f['intrinsic'])
 xyz=u['world']@w[:3,:3].T+w[:3,3];q=xyz@k.T;uv.append(q[:,:2]/q[:,2:]);zz.append(xyz[:,2])
 r=np.linalg.inv(w)[:3,3]-u['world'];r/=np.maximum(np.linalg.norm(r,axis=1)[:,None],1e-12);cos.append(np.abs((r*u['normal_world']).sum(1)))
cos=np.asarray(cos);uv=np.asarray(uv);zz=np.asarray(zz)
choices=[('cosine_0.06','37'),('cosine_0.06','5'),('drop_boxes_clear','37'),('drop_boxes_clear','6'),('minimum_views_1','37'),('depth_0.5','37')]
out=[]
for name,actor in choices:
 pool=np.flatnonzero(a['eligible_'+name]&~a['baseline_active']&actor_masks[actor])
 if not len(pool):continue
 idx=int(pool[np.argsort(u['xy_local'][pool,0])[len(pool)//2]])
 if name=='cosine_0.06':accept=np.logical_and.reduce([cos[:,idx]>=.06 if k=='angle_ok' else v[:,idx] for k,v in gates.items()])
 elif name=='drop_boxes_clear':accept=np.logical_and.reduce([v[:,idx] for k,v in gates.items() if k!='boxes_clear'])
 elif name=='minimum_views_1':accept=old[:,idx]
 else:accept=np.logical_and.reduce([v[:,idx] for k,v in gates.items() if k!='measured_depth_match'])
 # depth example shows depth-gate upper-bound sources, not a replay of the .5 test.
 frames=np.flatnonzero(accept);witness=[]
 for fi in frames[[0,-1]] if len(frames)>1 else frames:
  witness.append(dict(frame=m['frame_order'][int(fi)],uv_legacy=uv[fi,idx].tolist(),camera_z=float(zz[fi,idx]),cosine=float(cos[fi,idx]),failed_baseline_gates=[k for k,v in gates.items() if not v[fi,idx]]))
 out.append(dict(variant=name,actor_query_union=actor,unit_id=int(u['unit_ids'][idx]),array_index=idx,local_xy=u['xy_local'][idx].tolist(),baseline_sources=int(old[:,idx].sum()),variant_sources=int(a['source_count_'+name][idx]),baseline_gate_pass_counts={k:int(v[:,idx].sum()) for k,v in gates.items()},witnesses=witness,witness_note='depth_0.5 example witnesses omit depth gate; .5 aggregate is in audit, no witness-level .5 acceptance claimed' if name=='depth_0.5' else 'accepted source frames under this variant; color agreement is not ground truth'))
(root/'point_examples.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))

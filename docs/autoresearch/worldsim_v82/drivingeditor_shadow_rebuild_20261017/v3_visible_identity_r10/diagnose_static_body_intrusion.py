"""Frozen per-Gaussian attribution of static/other-Actor intrusion into known bodies."""
from pathlib import Path
import json,numpy as np,torch
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
s=torch.load(r/'E5_ray_visible_from8k/latest.pt',map_location='cpu',weights_only=False)
if s['global_step']!=9000:raise ValueError('frozen9k source changed')
state=s['state'];e=np.load(r/'commonR9_E5_009000/identity_evidence.npz');mass=e['mass'];owner=state['field.owner'].numpy();road=state['field.road'].numpy()
scale=state['field.log_scales'].exp().numpy();anchor=state['field.anchor_log_scales'].exp().numpy();records=[]
for actor in [5,6,12,18,25,26,37]:
 contribution=mass[:,actor];static=owner<0;total=float(contribution.sum());intrusion=float(contribution[static].sum())
 selected=np.flatnonzero(static);top=selected[np.argsort(contribution[selected])[-20:][::-1]]
 rec=dict(actor=actor,total_visible_body_alpha=total,self_actor_mass=float(contribution[owner==actor].sum()),
  static_mass=intrusion,static_road_mass=float(contribution[static&road].sum()),other_actor_mass=float(contribution[(owner>=0)&(owner!=actor)].sum()),
  static_weighted_scale_max=float((contribution[static]*scale[static].max(-1)).sum()/max(intrusion,1e-9)),
  static_weighted_scale_inflation=float((contribution[static]*(scale[static]/anchor[static]).max(-1)).sum()/max(intrusion,1e-9)),
  top_static=[dict(row=int(j),body_contribution=float(contribution[j]),road=bool(road[j]),scale=scale[j].tolist(),anchor_scale=anchor[j].tolist(),
                   xyz=state['field.xyz'][j].tolist(),seed_xyz=state['field.seed_xyz'][j].tolist()) for j in top])
 records.append(rec);print(json.dumps({k:v for k,v in rec.items() if k!='top_static'}),flush=True)
(r/'static_body_intrusion_009000_r9.json').write_text(json.dumps(dict(global_step=9000,parameters_updated=False,train46=True,actors=records,
 limitation='positive known-body pixels; static contribution is not by itself a wrong ownership label, inspect actual depth/source and Gaussian footprint'),indent=2))

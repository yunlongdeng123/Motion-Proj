"""Bake existing gates x fitted residual and pin a portable persistent scene."""
import sys
from pathlib import Path
import numpy as np
import torch
import closedloop as t

def pin(p):return dict(path=str(Path(p).resolve()),sha256=t.sha(p))

def main():
 c=t.C;completed=t.read(c/'formal/completed.json');assert completed['steps']==1000
 source=t.active_manifest();m=t.read(source)
 fitted=torch.load(completed['checkpoint'],map_location='cpu',weights_only=False)
 assert fitted['binding']==t.binding_for(source,m['road'])
 with np.load(m['ownership']) as z:ns=len(z['static_actor_ids']);nd=len(z['dynamic_actor_ids'])
 oldids=np.load(m['static_gates']['indices'])['ids'];old=torch.load(m['static_gates']['checkpoint'],map_location='cpu',weights_only=False)
 newids=fitted['contributors'].numpy();assert np.array_equal(newids,np.load(c/'candidates.npz')['ids'])
 mult=torch.ones(ns);mult[oldids]=old['gate'].sigmoid();mult[newids]*=fitted['gate'].sigmoid()
 ids=np.union1d(oldids,newids);gate=torch.logit(mult[ids].clamp(1e-12,1-1e-7))
 d=c/'persistent';d.mkdir(exist_ok=True)
 np.savez_compressed(d/'static_indices.npz',ids=ids)
 m['static_gates']=dict(indices=str(d/'static_indices.npz'),checkpoint=str(d/'static_opacity.pt'))
 manifest=d/'scene.json';t.write(manifest,m)
 binding=dict(manifest_sha256=t.sha(manifest),road_sha256=t.sha(m['road']),static_rows=ns,dynamic_rows=nd,
  fitted_checkpoint_sha256=t.sha(completed['checkpoint']),old_checkpoint_sha256=t.sha(t.read(source)['static_gates']['checkpoint']))
 torch.save(dict(schema='persistent_static_opacity_handoff.v1',gate=gate,contributors=torch.from_numpy(ids),binding=binding),d/'static_opacity.pt')
 with np.load(m['road']) as z:total=len(z['active']);active=int(z['active'].sum());scale=float(z['meter_to_model'])
 sources=[dict(role=role,**pin(Path(m['model'])/f'{prefix}{m["iteration"]}.pth')) for role,prefix in [('static','chkpnt'),('dynamic','dynamic_chkpnt'),('environment','env_light_chkpnt')]]
 proof=dict(schema='persistent_road_handoff.v1',manifest=pin(manifest),road=dict(**pin(m['road']),total_rows=total,active_rows=active,meter_to_model=scale),
  native_sources=sources,ownership=pin(m['ownership']),topology=dict(static_rows=ns,dynamic_rows=nd),
  gates=dict(indices=pin(d/'static_indices.npz'),checkpoint=pin(d/'static_opacity.pt'),checkpoint_binding=binding),
  raster_units=dict(unit_scale=1/scale,legacy_native_near=.2))
 t.write(d/'proof.json',proof)
 print('PERSISTENT_READY',manifest,flush=True)
 if '--no-gpu-check' in sys.argv:return
 # Same generic public loader used on a future process; no experiment monkeypatch.
 sys.path.insert(0,str(t.R/'scripts/worldsim_v82'))
 from drivingeditor_dynamic_edit.persistent_road_handoff import load_persistent_handoff
 scene,manifest,receipt=load_persistent_handoff(d/'scene.json',d/'proof.json')
 checks=[]
 with torch.no_grad():
  for key in t.KEYS:
   out=scene.render(key,delete_actor_ids=(10,),focus_actor_ids=(10,),diagnostic_actor_ids=(10,8,21),diagnostic_branch_alpha=True)
   with np.load(c/'pilot_formal'/f'{key:03d}/arrays.npz') as z:expected=torch.as_tensor(z['rgb'],device='cuda')
   delta=float((out['rgb']-expected).abs().max())
   rec=dict(raw=key,max_RGB_abs_difference=delta,RGB8_max_difference=int((torch.round(out['rgb']*255)-torch.round(expected*255)).abs().max()),target_owner_alpha_max=float(out['owner_alpha_T'][0].max()))
   assert rec['target_owner_alpha_max']==0
   checks.append(rec)
 scene.assert_persistent_unchanged()
 t.write(d/'reload_validation.json',dict(completed=True,loader=receipt,rows=checks,
   max_RGB_abs_difference=max(r['max_RGB_abs_difference'] for r in checks),RGB8_max_difference=max(r['RGB8_max_difference'] for r in checks),
   original_checkpoints_content_pinned=True,source_models_never_overwritten=True,GPU='keep_on'))
 print('RELOAD',checks,flush=True)

if __name__=='__main__':main()

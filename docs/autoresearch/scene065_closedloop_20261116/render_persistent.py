"""Render the saved generic scene/proof; no adaptation or image-space copyback."""
import argparse,sys,time
import numpy as np
import torch
import coverage_handoff  # selects the final immutable run and exact binding
import closedloop as t

def main():
 p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args()
 sys.path.insert(0,str(t.R/'scripts/worldsim_v82'))
 from drivingeditor_dynamic_edit.persistent_road_handoff import load_persistent_handoff
 scene,manifest,receipt=load_persistent_handoff(t.C/'persistent/scene.json',t.C/'persistent/proof.json')
 root=t.C/('persistent_pilot' if a.pilot else 'rendered');root.mkdir(exist_ok=False)
 frames=t.KEYS if a.pilot else list(range(199));rows=[];begin=time.time()
 provider=next(p for p in scene._providers if p.kind=='road')
 with torch.no_grad():
  for key in frames:
   tick=time.time();d=root/f'{key:03d}';d.mkdir()
   pack,parts,error=t.contributions(scene,key)
   cam=scene.camera_provider(key);rv=provider.rows(key,cam,scene.pipe)
   road=scene._bindings().raster(cam,scene.pipe,0,rv,scene._background)
   for name,value in dict(new_delete=pack['rgb'],road_only=road['rgb'],static_contribution=parts['static'],road_contribution=parts['road'],dynamic_contribution=parts['dynamic']).items():t.save_rgb(d/(name+'.png'),value)
   np.savez_compressed(d/'arrays.npz',rgb=pack['rgb'].cpu().numpy(),road_alpha=pack['extra'][3:4].cpu().numpy(),road_only_alpha=road['alpha'].cpu().numpy(),static_rgb=parts['static'].cpu().numpy(),road_rgb=parts['road'].cpu().numpy(),alpha=pack['alpha'].cpu().numpy())
   record=dict(raw=key,seconds=time.time()-tick,conservation=error,finite=bool(torch.isfinite(pack['rgb']).all()),target_owner_alpha_max=float(pack['extra'][6].max()))
   assert record['finite'] and record['target_owner_alpha_max']==0
   rows.append(record)
   if key%20==0:print(record,flush=True);t.write(root/'progress.json',dict(completed_frames=len(rows),raw=key))
 scene.assert_persistent_unchanged()
 proof=t.read(t.C/'persistent/proof.json')
 for source in proof['native_sources']:assert t.sha(source['path'])==source['sha256']
 t.write(root/'completed.json',dict(completed=True,frames=len(frames),keys=frames,rows=rows,road=str(manifest.road),
  persistent_manifest=str(t.C/'persistent/scene.json'),persistent_proof=str(t.C/'persistent/proof.json'),loader=receipt,
  checkpoint=t.read(t.C/'formal/completed.json'),seconds=time.time()-begin,source_checkpoints_unchanged=True,
  image_space_final_compositing=False,per_frame_road_generation=False,extra_training_updates=0,GPU='keep_on'))
 print('RENDER_COMPLETE',len(frames),flush=True)

if __name__=='__main__':main()

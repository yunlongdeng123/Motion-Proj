"""Bounded conservative road-row fallback; same criterion on all 199 views."""
import argparse,gc,json
from pathlib import Path
import numpy as np
import closedloop as t
import road_visibility_veto as v

def main():
 import torch
 c=t.C;old=np.load(t.OLD/'materials/corrected/road.npz')
 source=c/'manifests/road_visible_guarded.json';rows=[]
 for iteration in range(2,6):
  output=c/f'road_visibility_veto_r{iteration}'
  args=argparse.Namespace(manifest=source,legacy_frames=t.OLD/'rendered/corrected_delete/frames',provenance=c/'field/road_provenance.npz',
      output_road=c/'field'/f'road_disabled_control_r{iteration}.npz',output_manifest=c/'manifests'/f'road_disabled_control_r{iteration}.json',output=output)
  v.execute(args)
  receipt=t.read(output/'receipt.json');ids=np.load(output/'vetoed_world_rows.npz')['world_row_ids']
  with np.load(t.read(source)['road']) as z:asset={k:z[k].copy() for k in z.files}
  new=ids[asset['minimum_model_depth'][ids]==0]
  row=dict(iteration=iteration,conflict_pixels=sum(r['retained']['pixels_over_1byte'] for r in receipt['rows']),responsible_world_rows=len(ids),new_fallback_rows=len(new))
  rows.append(row)
  if not len(new):break
  asset['reference_srgb'][new]=old['reference_srgb'][new];asset['minimum_model_depth'][new]=.2
  road=c/'field'/f'road_visible_guarded_r{iteration}.npz';np.savez_compressed(road,**asset)
  m=t.read(source);m['road']=str(road);source=c/'manifests'/f'road_visible_guarded_r{iteration}.json';t.write(source,m)
  gc.collect();torch.cuda.empty_cache()
 t.write(c/'active_scene.json',dict(manifest=str(source),selection='bounded fixed visibility criterion; no quality tuning',iterations=rows))
 t.write(c/'field/final_visibility_guard.json',dict(completed=True,manifest=str(source),rows=rows,
     source_fields_preserved=True,minimum_model_depth_policy='fallback rows retain original road RGB and legacy center near; others physical near .2m',
     extra_neural_forwards=0,optimizer_updates=0))
 t.render('pilot')

if __name__=='__main__':main()

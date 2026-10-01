from pathlib import Path
import sys,json
from collections import Counter
import numpy as np
from PIL import Image
P=Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected');sys.path.insert(0,str(P))
from geometry_factory import read,dump
from temporal_metrics import process,THRESHOLDS
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929');O=T/'r9'

def sources(folder,sid):
 for root in folder.parents:
  if (root/'source_manifest.json').exists():return next(c for c in read(root/'source_manifest.json')['clips'] if c['source_id']==sid)
 raise RuntimeError(('source not found',folder,sid))

def synthetic(root):
 rows=[]
 for c in read(root/'dataset_catalog.json')['cases']:
  folder=Path(c['folder']);pair=read(folder/'pair_manifest.json');s=sources(folder,pair['source_id'])
  for start in range(0,c['frame_count']-9,10):
   ids=list(range(start,start+10));fs=[s['frames'][i] for i in ids];poses=[pair['frames'][i]['actor'] for i in ids]
   load=lambda r,i:np.asarray(Image.open(folder/r/f'{i:03}.png'))>0
   h=[load('model_hole',i) for i in ids];protected={};ba={}
   if c['type']!='background':
    tokens={p.stem.split('_',1)[1] for p in (folder/'protected').glob('000_*.png')}
    for tok in tokens:
     mm=[np.asarray(Image.open(folder/'protected'/f'{i:03}_{tok}.png'))>0 for i in ids]
     if not any((m&hh).sum()>20 for m,hh in zip(mm,h)):continue
     protected[tok]=mm;ba[tok]=[next(a for a in f['actors'] if a['instance_token']==tok) for f in fs]
   rows.append({'case_id':c['case_id'],'scene':c['receiver_scene'],'split':c['split'],'type':c['type'],'window_start':start,**process(fs,poses,h,protected,ba)})
 return rows

def summary(rows):
 return {'windows':len(rows),'scene_count':len({r['scene'] for r in rows}),
  'static_A_moving_ego':sum(r['static_A_moving_ego'] for r in rows),
  'static_A_moving_ego_image_change':sum(r['static_A_moving_ego_image_change'] for r in rows),
  'sweep_over_any_B':sum(r['sweep_over_any_B'] for r in rows),
  'visibility_transition_any_B':sum(r['visibility_transition_any_B'] for r in rows),
  'protected_windows':sum(bool(r['protected']) for r in rows),
  'median_duration_s':float(np.median([r['duration_s'] for r in rows])),
  'median_max_B_centre_span':float(np.median([max((max(v['occlusion_centre_span']) for v in r['protected'].values()),default=0) for r in rows]))}

def main():
 out={'thresholds':THRESHOLDS,'units':'world meters / native1024x576 pixels / B-normalized bbox coordinates','human_verdict':None,'suites':{}}
 for name,root in [('r7',T/'r7/encoder_fixed_lowres'),('r8',T/'r8')]:
  rs=synthetic(root);out['suites'][name]={'rows':rs,'training':summary([r for r in rs if r['split']=='train']),'validation':summary([r for r in rs if r['split']=='validation'])};dump(O/'temporal_audit.json',out)
  print(name,out['suites'][name]['training'],flush=True)
 out['limits']='window-level counts include same-world variants; cuboid cells not exact material correspondence; no video quality or transfer causality claim'
 dump(O/'temporal_audit.json',out)
if __name__=='__main__':main()

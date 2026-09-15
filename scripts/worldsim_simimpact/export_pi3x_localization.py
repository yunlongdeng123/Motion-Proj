import json,shutil,tarfile
from pathlib import Path
import numpy as np
from PIL import Image
I=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-INTERACTION-01/20260915-r1');L=I/'lidar_policy';C=L/'pi3x_localization';D=C/'export';D.mkdir(exist_ok=True)
inp=json.loads((I/'inputs/scene-0061.json').read_text());gt=np.load(L/'scans/scene-0061/real.npz')
arrays={'gt':gt['points'],'origin':gt['origin'],'directions':gt['directions'],'ranges':gt['ranges']}
regions=np.load(C/'region_masks.npz');stats=[]
for variant in ['six','twelve']:
 s=np.load(L/'scans/scene-0061'/f'pi3x_{variant}_build_scale.npz');f=s['first_range'];finite=np.isfinite(f)
 pp=gt['points'].copy();pp[finite]=gt['origin']+gt['directions'][finite]*f[finite,None]
 arrays[variant]=pp;arrays[variant+'_range']=f
 m=regions[variant+'_forward_right'];delta=f-gt['ranges']
 stats.append({'variant':variant,'points':int(m.sum()),'finite':int((m&finite).sum()),
  'gt_xyz_quantiles':np.quantile(gt['points'][m],[0,.1,.5,.9,1],axis=0).tolist(),
  'range_delta_quantiles':np.quantile(delta[m&finite],[0,.1,.5,.9,1]).tolist(),
  'z_difference_quantiles':np.quantile((pp-gt['points'])[m&finite,2],[0,.1,.5,.9,1]).tolist()})
np.savez_compressed(D/'points.npz',**arrays)
shutil.copy2(C/'region_masks.npz',D/'region_masks.npz');shutil.copy2(C/'localization_outcomes.json',D/'localization_outcomes.json')
shutil.copy2(C/'registration.json',D/'registration.json');shutil.copy2(C/'pdm_simulation/scene-0061/states.npz',D/'states.npz')
meta=[]
for v in inp['views'][:6]:
 im=Image.open(v['image']).convert('RGB');im.thumbnail((1200,750));im.save(D/(v['camera']+'.jpg'),quality=93)
 meta.append({**v,'export_wh':list(im.size)})
(D/'views.json').write_text(json.dumps(meta,indent=2));(D/'right_region_stats.json').write_text(json.dumps(stats,indent=2))
with tarfile.open(C/'export.tar.gz','w:gz') as t:
 for p in D.iterdir():t.add(p,arcname=p.name)
print(json.dumps(stats,indent=2))

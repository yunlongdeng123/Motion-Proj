"""导出可复查的小型证据和科学绘图数据；大网格/权重留在运行根。"""
import json,os,shutil,tarfile
from pathlib import Path
import numpy as np
from PIL import Image
R=Path(os.environ.get('SIMIMPACT_RUN_ROOT','/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1'))
L=R/'lidar_policy';O=R/'lidar_evidence';O.mkdir(exist_ok=True)
names=sorted(p.stem for p in (R/'inputs').glob('*.json'))
for f in ['registration.json','selection_summary.json','renderer_depth_contract.json','alignment_pnp_r2.json']:
 if (R/f).exists():shutil.copy2(R/f,O/f)
for f in ['registration.json','raycast_summary.json','policy_probe_summary.json','pdm_simulation_summary.json','asset_downloads_ranged.json']:
 if (L/f).exists():shutil.copy2(L/f,O/('lidar_'+f if f=='registration.json' else f))
for name in names:
 D=O/name;D.mkdir(exist_ok=True)
 inp=json.loads((R/'inputs'/f'{name}.json').read_text());front=inp['views'][0]
 im=Image.open(front['image']);im.thumbnail((1200,800));im.save(D/'front.jpg',quality=92)
 (D/'input.json').write_text(json.dumps(inp,indent=2))
 shutil.copy2(L/f'{name}_log_reference.json',D/'log_reference.json')
 shutil.copy2(L/'pdm_simulation'/name/'states.npz',D/'states.npz')
 real=np.load(L/'scans'/name/'real.npz');np.savez_compressed(D/'real.npz',**{k:real[k] for k in real.files})
 for f in (L/'policy_outputs'/name).glob('*histogram.npy'):
  if f.name=='real_histogram.npy' or '_six_build_scale_full_' in f.name:shutil.copy2(f,D/f.name)
 for method in ['vggt','omega512','dvgt1','pi3x']:
  f=L/'scans'/name/f'{method}_six_build_scale.npz'
  s=np.load(f)
  np.savez_compressed(D/f.name,**{k:s[k] for k in ['points','first_range','gt_range','origin','directions']})
with tarfile.open(R/'lidar_evidence.tar.gz','w:gz') as t:
 for p in O.rglob('*'):
  if p.is_file():t.add(p,arcname=p.relative_to(O).as_posix())
print('EXPORTED',names,flush=True)

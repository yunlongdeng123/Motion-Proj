import pathlib,json,numpy as np
from geometry_contract import R
rows=[]
for f in sorted((R/'secondary/predictions/dggt').glob('*/*/result.json')):
 rec=json.loads(f.read_text());a=np.load(f.parent/'native_outputs.npz');b=np.load(R/'predictions/vggt'/rec['scene']/rec['variant']/'native_outputs.npz')
 rows.append({'scene':rec['scene'],'variant':rec['variant'],'depth_max_abs_difference':float(np.max(np.abs(a['depth']-b['depth']))),'pose_max_abs_difference':float(np.max(np.abs(a['pose_enc']-b['pose_enc'])))})
(R/'secondary/dggt_vggt_geometry_comparison.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows),flush=True)

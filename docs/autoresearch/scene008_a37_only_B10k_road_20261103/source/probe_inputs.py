from pathlib import Path
import numpy as np,json
OLD=Path('/root/autodl-tmp/drivingeditor_repro/campaign/shadow_a5_a37_199_20261103_r4/inputs')
for p in [OLD/'road_reference.npz',OLD/'views/105.npz']:
    d=np.load(p);print(str(p));print(json.dumps({k:dict(shape=d[k].shape,dtype=str(d[k].dtype),count=int(d[k].sum()) if d[k].dtype==bool else None) for k in d.files}))

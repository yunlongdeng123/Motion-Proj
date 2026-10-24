"""Stricter spatial withdrawal: guard against source-triangle geometry leakage.

Original atlas triangles span at most 1.6m. Remove a 1.7m halo around each
evaluation rectangle from all donor cells; no retained triangle can directly
reach the withheld rectangle (including 10cm cell rounding allowance).
"""
import json,argparse
from pathlib import Path
import numpy as np
from road_teacher_cpu import Teacher,metric
W=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=W/'road_teacher_cpu_r1');R=ap.parse_args().results
a=dict(np.load(R/'observed_atlas.npz'));old=json.loads((R/'validation.json').read_text());xy=a['xy']
rows=[];pool={}
for row in old['holes']:
    cx,cy=row['center'];withdraw=(abs(xy[:,0]-cx)<5.7)&(abs(xy[:,1]-cy)<2.7)
    core=(abs(xy[:,0]-cx)<3)&(abs(xy[:,1]-cy)<.6)&(a['source_frames']>=2)
    ids=np.flatnonzero(core);teacher=Teacher(a,~withdraw);rr=dict(center=[cx,cy],split=row['split'],core_count=len(ids),removed_cells=int(withdraw.sum()),methods={})
    for name in ['nearest','isotropic','road_gaussian']:
        p=teacher.predict(xy[ids],name);mm=metric(p['rgb'],a['rgb'][ids]);mm['height_mae']=float(abs(p['z']-a['z'][ids]).mean());rr['methods'][name]=mm
        pool.setdefault((row['split'],name),[]).append((p['rgb'],a['rgb'][ids],p['z'],a['z'][ids]))
    rows.append(rr)
aggregate={}
for (split,name),parts in pool.items():
    pred,ref,pz,rz=[np.concatenate([p[i] for p in parts]) for i in range(4)]
    mm=metric(pred,ref);mm['height_mae']=float(abs(pz-rz).mean());aggregate.setdefault(split,{})[name]=mm
out=dict(withdraw_halo_m=1.7,triangle_max_edge_m=1.6,rows=rows,aggregate=aggregate,
    limitation='Atlas uses shared camera calibration and global ground classification. This is teacher-level same-scene withdrawal, not an end-to-end unseen-scene reconstruction test.')
(R/'strict_withdrawal.json').write_text(json.dumps(out,indent=2));print(json.dumps(aggregate,indent=2))

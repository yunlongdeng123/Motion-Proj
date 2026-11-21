"""Describe the full frozen point proxy; never remove points or alter a fitted asset."""
import argparse
import json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('--audit-dir',type=Path,required=True)
a=p.parse_args()
b=np.load(a.audit_dir/'point_surface_residuals.npz',allow_pickle=False)
xyz=b['xyz_canonical_m'];d=b['point_to_surface_m'];src=b['source_raw']
def stat(mask):
    v=d[mask]
    return {'count':int(mask.sum()),'mean_m':float(v.mean()) if len(v) else None,
            'p90_m':float(np.quantile(v,.9)) if len(v) else None,
            'above_15cm_count':int((v>.15).sum()),'above_30cm_count':int((v>.30).sum())}
def bins(axis,edges):
    out=[]
    for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        mask=(xyz[:,axis]>=lo)&(xyz[:,axis]<(hi if j<len(edges)-2 else hi+1e-5))
        row={'interval_m':[lo,hi],**stat(mask)}
        row['by_source']={str(raw):stat(mask&(src==raw)) for raw in [86,110]}
        out.append(row)
    return out
result={'schema':'a20.hq.full_proxy_spatial_residuals.v1','point_count':len(d),'main_metric_unchanged':True,
        'point_ownership':'Actual measured points selected by OBB/a9 guard, not semantic instance ground truth.',
        'by_source':{str(raw):stat(src==raw) for raw in [86,110]},
        'z_height_bins':bins(2,[-1.04,-.74,-.24,.26,.56,.74,1.04]),
        'x_longitudinal_bins':bins(0,[-2.5,-1.8,-1.0,-.2,.6,1.4,2.5]),
        'z_upper_30cm':stat(xyz[:,2]>=.74),
        'z_upper_50cm':stat(xyz[:,2]>=.54),
        'heldout_used':False,'gpu_used':False,
        'note':'Spatial strata are descriptive; no subset replaces the full 1030-point gate. High residuals may reflect shape mismatch or point ownership uncertainty.'}
(a.audit_dir/'spatial_residual_breakdown.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
order=np.argsort(-d)[:100]
table=np.column_stack([src[order],xyz[order],d[order],b['closest_surface_xyz_m'][order]])
np.savetxt(a.audit_dir/'largest_100_residuals.csv',table,delimiter=',',header='source_raw,x_m,y_m,z_m,distance_m,closest_x_m,closest_y_m,closest_z_m',comments='')
print(json.dumps(result,indent=2))

import hashlib
import json
from pathlib import Path
import numpy as np

root=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030')
out=root/'m2_repair_r1'
runs={'strict_position_r2':'strict_r2','strict_position_scale':'strict_r2','balanced_position':'balanced','balanced_position_scale':'balanced','balanced_position_scale_split_r2':'balanced'}
result=[]
for run,baseline in runs.items():
    folder=out/run
    if not (folder/'reference_metrics.json').exists():
        continue
    mine=json.loads((folder/'reference_metrics.json').read_text())
    base=json.loads((root/'m1_assignment_r1'/baseline/'reference_metrics.json').read_text())
    lookup={(r['frame'],r['edit']):r for r in base['rows']}
    checks=[]
    for frame in ['000000','000008','000015']:
        with np.load(folder/('selection_'+frame+'.npz')) as a, np.load(root/'m1_assignment_r1'/baseline/('selection_'+frame+'.npz')) as b:
            checks.append(dict(frame=frame,indices_identical=bool(np.array_equal(a['selected'],b['selected'])),target_identical=bool(np.array_equal(a['vehicle_surface_target_model'],b['vehicle_surface_target_model']))))
    rows=[]
    for row in mine['rows']:
        prior=lookup[(row['frame'],row['edit'])]
        for region in ['full','body_reference','original_body_footprint','newly_exposed_road','road_reference']:
            metric=row['regions'][region]
            if 'raw_mae' in metric:
                old=prior['regions'][region]['raw_mae']; new=metric['raw_mae']
                rows.append(dict(frame=row['frame'],edit=row['edit'],region=region,m1_mae=old,m2_mae=new,delta_m2_minus_m1=new-old,relative_reduction_pct=100*(old-new)/old if old else None))
    result.append(dict(run=run,m1_baseline=baseline,selection_checks=checks,rows=rows,repairs=json.loads((folder/'responsibility_summary.json').read_text())['frames']))
(out/'comparison.json').write_text(json.dumps(result,indent=2))
print(json.dumps([dict(run=r['run'],selection_identical=all(c['indices_identical'] for c in r['selection_checks']),midframe=[x for x in r['rows'] if x['frame']=='000008' and x['region']=='full']) for r in result],indent=2))

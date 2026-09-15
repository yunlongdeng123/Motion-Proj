"""核对保存轨迹、报告分母和图册链接；不重跑任何推理。"""
import json,re
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parent;E=W/'evidence_fresh_baseline/evidence';O=W.parents[1]/'outputs/Simulation_Impact_Research'
total=0
for condition,name in [('baseline','fit_admission.json'),('baseline_route_r1','fit_admission_route.json')]:
    summary=json.loads((E/name).read_text());rows=summary['rows'][0]['individual_outcomes'];assert len(rows)==8
    for row in rows:
        a=np.load(E/condition/row['scene']/'states.npz');error=np.linalg.norm(a['simulated'][0,:,:2]-a['ground_truth_ego'][:,:2],axis=1)
        assert len(error)==41
        assert abs(float(error.mean())-row['ADE_vs_recorded_ego_m'])<1e-9
        assert abs(float(error[-1])-row['FDE_vs_recorded_ego_m'])<1e-9
        total+=1
    assert summary['selected_fit_scene'] is None
assert total==16
html=(O/'index.html').read_text(encoding='utf-8');links=re.findall(r'(?:href|src)="([^"]+)"',html)
local=[u for u in links if not u.startswith(('http:','https:','#'))]
assert all((O/u).is_file() for u in local),[u for u in local if not(O/u).exists()]
assert html.count('<img src="F21_')==1
result={'verified_saved_trajectories':total,'timesteps_each':41,'recomputed_ADE_FDE_match':True,'failed_admissions_preserved':2,'valid_local_gallery_links':len(local),'native_SparseDrive_forward_count':0,'new_scene_fits':0}
(O/'FRESH_BASELINE_VERIFICATION.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

"""复核已有真实输入基线与已冻结目标轨迹，不按重建误差选择新日志。"""
import json
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parent;D=W/'evidence_milestone3'
rows=json.loads((D/'pdm_simulation_summary.json').read_text())
selection=json.loads((D/'selection_summary.json').read_text());out=[]
for b in rows:
    if b['condition']!='real':continue
    scene=b['scene'];ref=json.loads((D/scene/'log_reference.json').read_text())
    target=selection[scene]['metadata_selection']['leads'][0];seq=[]
    for r in ref['records']:
        box=next((v for v in r['boxes'] if v['instance']==target['instance']),None)
        if box is None:continue
        t=np.linalg.inv(np.array(r['pose']));p=t[:3,:3]@np.array(box['box'][:3])+t[:3,3]
        seq.append({'time_s':r['time_s'],'forward_m':float(p[0]),'lateral_m':float(p[1])})
    out.append({'scene':scene,'real_ADE_m':b['ADE_vs_recorded_ego_m'],'real_FDE_m':b['FDE_vs_recorded_ego_m'],
                'real_annotated_overlap':b['actor_overlap_any'],'real_min_accel_mps2':b['minimum_longitudinal_acceleration_mps2'],
                'target_instance':target['instance'],'target_class':target['category'],'target_trajectory_in_ego':seq})
result={'role':'Review of already exposed six-log cohort and prior real-input PDM forecasts. No new inference, no independent confirmation.',
        'decision':'scene-0073 is a candidate for crossing-actor visibility after scene-0004 full-fit quality/cost review. Do not automatically fit it or reopen scene-0061 road residuals. scene-0028 already has baseline overlap.',
        'rows':out}
(W/'native_cohort_review.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
for r in out:print(r['scene'],r['target_class'],round(r['real_ADE_m'],3),r['real_annotated_overlap'])

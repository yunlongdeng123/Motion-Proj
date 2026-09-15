"""汇总全分母、基线失败及误差分解；不按有害结果筛掉样本。"""
import json
from pathlib import Path
import numpy as np
R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-INTERACTION-01/20260915-r1');L=R/'lidar_policy'
a=json.loads((L/'pdm_simulation_summary.json').read_text());p=json.loads((L/'policy_probe_summary.json').read_text());q=json.loads((L/'raycast_summary.json').read_text())
base={r['scene']:r for r in a if r['condition']=='real'}
summary={'task':'WS-SIM-INTERACTION-01','run':'20260915-r1','policy_predictions':len(p),'vehicle_executions':len(a),
 'geometry_forwards':len(list((R/'predictions').glob('*/*/*/result.json'))),'raycast_conditions':len(q),
 'scope':'Official policy and nonreactive 4s vehicle tracking with nuScenes adapter, not full PDM score or pose-responsive sensor loop',
 'registered_scenes':list(base),'real_lidar_baselines':list(base.values()),'groups':{},'new_overlap_cases':[],'largest_controlled_shifts':[]}
for protocol in ['native_scale','build_scale']:
 for condition in ['full','early_only','late_only','missing_only']:
  rows=[r for r in a if r.get('protocol')==protocol and r['condition']==condition]
  diff=np.array([r['ADE_vs_recorded_ego_m']-base[r['scene']]['ADE_vs_recorded_ego_m'] for r in rows]);shift=np.array([r['final_position_change_vs_real_m'] for r in rows])
  summary['groups'][protocol+'/'+condition]={'n':len(rows),'actor_overlap':sum(r['actor_overlap_any'] for r in rows),
   'new_overlap_vs_real_lidar':sum(r['actor_overlap_any'] and not base[r['scene']]['actor_overlap_any'] for r in rows),
   'overlap_removed_vs_real_lidar':sum(not r['actor_overlap_any'] and base[r['scene']]['actor_overlap_any'] for r in rows),
   'ADE_difference_min_m':float(diff.min()),'ADE_difference_median_m':float(np.median(diff)),'ADE_difference_max_m':float(diff.max()),
   'endpoint_shift_median_m':float(np.median(shift)),'endpoint_shift_max_m':float(shift.max())}
  for r in rows:
   if r['actor_overlap_any'] and not base[r['scene']]['actor_overlap_any']:summary['new_overlap_cases'].append(r)
rows=[r for r in a if r.get('protocol')=='build_scale' and r['condition']=='full']
summary['largest_controlled_shifts']=sorted(rows,key=lambda r:r['final_position_change_vs_real_m'],reverse=True)[:8]
summary['recorded_ego_overlap_scenes']=[n for n,b in base.items() if b['recorded_ego_actor_overlap_under_same_vehicle']]
(L/'outcome_audit.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:summary[k] for k in ['policy_predictions','vehicle_executions','geometry_forwards','groups','recorded_ego_overlap_scenes']},indent=2),flush=True)

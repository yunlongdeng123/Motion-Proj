"""按预先写定的真实输入质量条件决定是否值得开展昂贵场景拟合。"""
import json,time,shutil,argparse
from pathlib import Path
import numpy as np
R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-FRESH-NATIVE-01/20260915-r1')
ap=argparse.ArgumentParser();ap.add_argument('--route-control',action='store_true');args=ap.parse_args()
B=R/('baseline_route_r1' if args.route_control else 'baseline');L=B/'lidar_policy';admission=R/('fit_admission_route.json' if args.route_control else 'fit_admission.json')
if admission.exists():raise RuntimeError('Preserve admission result')
reg=json.loads((B/'registration.json').read_text());policy=json.loads((L/'policy_probe_summary.json').read_text());pdm=json.loads((L/'pdm_simulation_summary.json').read_text())
assert len(policy)==len(pdm)==reg['expected_policy_forwards']
assert all(r['condition']=='real' for r in policy+pdm)
rows=[]
for scene in json.loads((R/'registration.json').read_text())['scenes']:
    rr=[r for r in pdm if r['scene'].startswith(scene+'_start')];assert len(rr)==8
    ade=float(np.mean([r['ADE_vs_recorded_ego_m'] for r in rr]));fde=float(np.mean([r['FDE_vs_recorded_ego_m'] for r in rr]));overlap=sum(r['actor_overlap_any'] for r in rr);gt_overlap=sum(r['recorded_ego_actor_overlap_under_same_vehicle'] for r in rr)
    rows.append({'scene':scene,'real_starts':len(rr),'mean_ADE_m':ade,'mean_FDE_m':fde,'executed_overlap_starts':overlap,'recorded_overlap_starts':gt_overlap,'qualifies':overlap==0 and gt_overlap==0 and ade<=1 and fde<=2,'individual_outcomes':rr})
selected=next((r['scene'] for r in rows if r['qualifies']),None)
result={'task_id':'WS-SIM-FRESH-NATIVE-01','stage':'real_baseline_complete','created_unix':time.time(),'rows':rows,'selected_fit_scene':selected,
    'new_policy_forwards':len(policy),'new_vehicle_forecasts':len(pdm),'cumulative_policy_vehicle_forecasts':802+len(policy)+(8 if args.route_control else 0),
    'information':'Current real RGB+LiDAR, backward-derived velocity/acceleration/yaw rate. PDM initial steering uses causal bicycle relation. '+('One GT-derived high-level route oracle is extra information; full future numeric trajectory/actors remain evaluator-only.' if args.route_control else 'Inherited fixed client command; future ground-truth trajectory/actors are evaluator-only.'),
    'decision':'Admit exactly one official full-budget scene fit; no automatic second scene.' if selected else 'No source passes frozen real-baseline requirements. Do not fit a scene or call this a reconstruction failure. Keep all eight starts and inspect the upstream baseline/interface limitation before changing research direction.',
    'failure_ledger_refs':['V74-H2-F18','V74-H2-F21'],'failure_ledger_delta':'pending','human_verdict':None,'goal_status':'active'}
admission.write_text(json.dumps(result,indent=2));shutil.copy2(__file__,B/'audit_source_snapshot.py')
reg.update(completed=True,completed_unix=time.time(),policy_forwards=len(policy),vehicle_forecasts=len(pdm));(B/'registration.json').write_text(json.dumps(reg,indent=2))
for row in rows:print({k:v for k,v in row.items() if k!='individual_outcomes'},flush=True)
print('ADMITTED',selected,flush=True)

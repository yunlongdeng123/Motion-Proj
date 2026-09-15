"""记录有限矩阵的终止判定，不将低杠杆 IoU 候选继续扩成几何扫参。"""
import json,shutil,time,tarfile
from pathlib import Path
O=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-FF-COHORT-PERCEPTION-01/20260915-r1');E=O/'evidence'
for p in [O/'registration.json',E/'decision.json']:
    backup=p.with_name(p.stem+'_before_closeout.json')
    if backup.exists():raise RuntimeError(f'Preserve {backup}')
    shutil.copy2(p,backup)
reg=json.loads((O/'registration.json').read_text());reg.update(completed=True,failure_ledger_delta='V74-H2-F21',end_unix=time.time())
for p in [O/'registration.json',E/'registration.json']:p.write_text(json.dumps(reg,indent=2))
decision=json.loads((E/'decision.json').read_text());decision.update(failure_ledger_delta='V74-H2-F21',
    outcome='Matrix completed. No real-qualified target has persistent center-match loss at both fixed scores in both view budgets after original missing-return restoration. All48 metadata lead conditions keep center matches;47 keep IoU matches.',
    new_candidates='Two scene0028 roadside pedestrians, about15m lateral, remain confidently detected. Center error about0.34..0.48m and small-box IoU loss. Geometry association only; no local causal intervention or demonstrated planning harm.',
    stop_decision='Close this finite cohort screen without ROI/threshold/model sweeps or training on the two new pedestrians. Retain all positive and negative results; no universal phantom, serious safety or independent confirmation claim.',
    new_geometry_forwards=0,new_policy_forwards=0,new_closed_loops=0,
    overall_goal_status='active',human_verdict=None)
(E/'decision.json').write_text(json.dumps(decision,indent=2));shutil.copy2(__file__,E/'closeout_source_snapshot.py')
shutil.copytree(O/'candidate_support_r1',E/'candidate_support')
with tarfile.open(O/'evidence_final.tar.gz','w:gz') as t:
    for p in E.rglob('*'):
        if p.is_file():t.add(p,arcname=p.relative_to(E))
print('CLOSED_FINITE_COHORT',decision['cumulative_detector_forwards'],flush=True)

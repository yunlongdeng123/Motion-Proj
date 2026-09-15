import json,shutil
from pathlib import Path
R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1')
out=R/'evidence_milestone1';out.mkdir(exist_ok=True)
for name in ['registration.json','native_collision_replay.json','ltf_input_dependency.json','native_collision_audit.json','paired_collision_geometry.json','paired_control_audit.json','geometry_swap_protocol.json','rollout_reproducibility.json','runtime_packages_final.json','asset_downloads_ranged.json']:
    if (R/name).exists():shutil.copy2(R/name,out/name)
rollouts=[]
for p in sorted((R/'rollouts').glob('*/*/summary.json')):
    row=json.loads(p.read_text());row['tag']=p.parent.name;row['path']=str(p.parent)
    row['runtime_qa']=json.loads((p.parent/'runtime_qa.json').read_text())
    if (p.parent/'official_post_eval.json').exists():
        ev=json.loads((p.parent/'official_post_eval.json').read_text());row['official_post_eval']={k:v for k,v in ev.items() if k!='details'}
    rollouts.append(row)
(out/'rollouts.json').write_text(json.dumps(rollouts,indent=2))
pred=[json.loads(p.read_text()) for p in sorted((R/'predictions').glob('*/*/*/result.json'))]
(out/'official_predictions.json').write_text(json.dumps(pred,indent=2))
coverage={'rollouts':len(rollouts),'executed_steps':sum(x['steps'] for x in rollouts),'official_forwards':len(pred),'geometry_swaps':len(json.loads((R/'geometry_swap_protocol.json').read_text())),
    'status':'RESEARCH_ACTIVE','not_completed':['natural feed-forward geometry -> meaningful simulation harm','reliable physical collision ground truth','second full simulator'],
    'failure_ledger_delta':'V74-H2-F14','human_verdict':None}
(out/'milestone.json').write_text(json.dumps(coverage,indent=2));print(json.dumps(coverage))

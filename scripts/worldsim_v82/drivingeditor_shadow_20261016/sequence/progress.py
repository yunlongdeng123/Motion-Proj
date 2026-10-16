import json,os,datetime
from pathlib import Path
root=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_sequence_20261016_v2')
jobs=[]
for path in root.glob('*_launch.json'):
    r=json.loads(path.read_text());pid=r['pid'];cmd=Path('/proc',str(pid),'cmdline')
    alive=cmd.exists() and bool(cmd.read_bytes())
    jobs.append({'name':path.stem,'pid':pid,'alive':alive})
current=next((name for name in ['unselected_clean_v1','contribution_frozen_v1','repair_union_v1','joint_v1'] if (root/name/'status.json').exists()),None)
status=json.loads((root/current/'status.json').read_text()) if current else {}
if status.get('state')=='running':
    phase=status.get('phase');step=status.get('stage_step',0)
    bg,sh,joint=(1500,2000,4500) if current=='joint_v1' else (500,250,1250) if current=='unselected_clean_v1' else (1000,500,2500)
    rem={'background':(bg-step)*.04+sh*.20+joint*.20,'shadow':(sh-step)*.20+joint*.20,'joint':(joint-step)*.20}.get(phase)
else:rem=None
follow=root/('final49_unselected_v1_follow.json' if current=='unselected_clean_v1' else 'final49_contribution_v1_follow.json' if current=='contribution_frozen_v1' else 'repair_follow_status.json' if current=='repair_union_v1' else 'finish_pipeline_status.json')
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':jobs,'current_training':current,'training_state':status.get('state'),'phase':status.get('phase'),
    'global_step':status.get('global_step'),'phase_step':status.get('stage_step'),'approx_remaining_training_minutes':rem/60 if rem is not None else None,
    'evaluation':json.loads(follow.read_text()) if follow.exists() else None}
(root/'latest_progress.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

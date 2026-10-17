from pathlib import Path
import json, time
D=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3/backend_restore_r15')
policy=json.loads((D/'execution_policy.json').read_text())
step=json.loads((D/'main/progress.json').read_text())['global_step']
if policy.get('defer_intermediate_previews_and_statistics') and step<45000:
    O=D/'previews_deferred';O.mkdir(exist_ok=True)
    receipt=dict(step=step,status='DEFERRED_BY_LATEST_USER_POLICY',at=time.time(),GPU_rendered=False,statistics_generated=False,quality_accepted=False)
    (O/f'{step:06d}.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)
else:
    source=D/'preview_before_hourly_policy.py'
    exec(compile(source.read_bytes(),str(source),'exec'))

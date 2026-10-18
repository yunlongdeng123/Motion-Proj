"""Stop only the two verified waiting controllers superseded by user scope."""
from pathlib import Path
import json,os,signal,time,datetime
E=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/baseline_engineering_20261018')
records=[]
for pid,script,status,expected in [(18165,'run_engineering_review.py','review_queue_status.json','WAITING_FOR_ACTIVE_MATCHED_GN11_NO_GPU_ACQUIRED'),(27418,'run_baseline_attribution.py','attribution_queue_status.json','WAITING_FOR_REGISTERED_REVIEW_NO_GPU_ACQUIRED')]:
    path=E/status; previous=json.loads(path.read_text())
    if previous['state']=='CANCELLED_BY_USER_DIRECT_30K_30K_NO_INTERSTAGE_ABLATIONS':
        records.append(previous);continue
    assert previous['state']==expected,previous
    cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert str(E/script) in cmd,(pid,cmd)
    children=Path(f'/proc/{pid}/task/{pid}/children').read_text().strip()
    assert not children,(pid,children)
    backup=path.with_suffix('.before_direct_plan.json')
    if not backup.exists():backup.write_bytes(path.read_bytes())
    os.kill(pid,signal.SIGTERM)
    for _ in range(30):
        if not Path(f'/proc/{pid}').exists():break
        time.sleep(.1)
    assert not Path(f'/proc/{pid}').exists(),'controller still alive'
    rec=dict(state='CANCELLED_BY_USER_DIRECT_30K_30K_NO_INTERSTAGE_ABLATIONS',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_state=expected,pid=pid,training_updates_consumed=0,assets_preserved=True,reason='User explicitly requested direct 30K+30K, no interstage ablations; existing GN11 worker unchanged')
    path.write_text(json.dumps(rec,indent=2));records.append(rec)
(E/'direct_plan_cancellation_receipt.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records))

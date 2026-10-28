"""Apply the user's explicit new detection authorization, without restarting training."""
import json, shutil, datetime
from pathlib import Path
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028')
p=ROOT/'authorization.json'
a=json.loads(p.read_text())
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
history=ROOT/'authorization_history';history.mkdir(exist_ok=True)
shutil.copy2(p,history/('before_detection_'+stamp+'.json'))
a.update(allow_v43_training=False,allow_target_detection=True,gpu_execution_hold=False,user_research_hold=False,
         latest_user_scope='User approved target-specific unknown detection, immediate underbody/shadow inclusion and separate effect color; adaptive experiments; no resumption of old repair/training queue',
         hold_reason=None,gpu_hold_reason=None,detection_authorized_utc=stamp,
         detection_contexts=['14705303724557273004_3105_000_3125_000'],old_queue_resumed=False,old_automation_resumed=False)
p.write_text(json.dumps(a,indent=2))
print(json.dumps({k:a[k] for k in ['allow_v43_training','allow_target_detection','gpu_execution_hold','user_research_hold','detection_contexts']}))

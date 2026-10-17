"""User-requested immediate R14 pause. Preserve exact last atomic checkpoint."""
from pathlib import Path
import os, signal, json, time, shutil, hashlib, datetime
D=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3/ownership_rebuild_r14')
def cmd(pid):
    try:return (Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    except (OSError,UnicodeError):return ''
owned=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name)==os.getpid():continue
    c=cmd(int(p.name))
    if str(D) in c and ('python' in c or c.startswith('flock ')):
        owned.append(dict(pid=int(p.name),cmd=c))
owned.sort(key=lambda v:0 if 'full_controller' in v['cmd'] else 1)
for v in owned:
    if cmd(v['pid'])==v['cmd']:
        try:os.kill(v['pid'],signal.SIGSTOP)
        except ProcessLookupError:pass
run=D/'main';pause=D/'user_pause_backend_audit';pause.mkdir(exist_ok=False)
for name in ['pipeline_state.json','full_controller_recovery_launch.json']:
    if (D/name).exists():shutil.copy2(D/name,pause/name)
for name in ['progress.json','reference_refresh.json','metrics.jsonl']:
    if (run/name).exists():shutil.copy2(run/name,pause/name)
shutil.copy2(run/'latest.pt',pause/'last_atomic_checkpoint.pt')
sha=hashlib.sha256((pause/'last_atomic_checkpoint.pt').read_bytes()).hexdigest()
for v in owned:
    if cmd(v['pid'])==v['cmd']:
        try:
            os.kill(v['pid'],signal.SIGTERM)
            os.kill(v['pid'],signal.SIGCONT)
        except ProcessLookupError:pass
time.sleep(2)
for v in owned:
    if cmd(v['pid'])==v['cmd']:
        try:os.kill(v['pid'],signal.SIGKILL)
        except ProcessLookupError:pass
import torch
saved=torch.load(pause/'last_atomic_checkpoint.pt',map_location='cpu')
progress=json.loads((pause/'progress.json').read_text())
value=dict(status='PAUSED_BY_USER_BACKEND_BASELINE_AUDIT',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           user_request='Stop current R14; audit simplifications versus old30k and30k+18k before any further training.',
           checkpoint_step=saved['global_step'],checkpoint_sha256=sha,
           last_logged_step=progress['global_step'],logged_updates_not_in_checkpoint=progress['global_step']-saved['global_step'],
           stopped_processes=owned,remaining_live=[v for v in owned if cmd(v['pid'])==v['cmd']],
           checkpoint_optimizer_rng_preserved=True,GPU_shutdown=False,automatic_resume_authorized=False)
(pause/'receipt.json').write_text(json.dumps(value,indent=2))
(D/'pipeline_state.json').write_text(json.dumps(value,indent=2))
print(json.dumps(value,indent=2))

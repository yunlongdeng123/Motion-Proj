from pathlib import Path
import json,time,subprocess
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');D=R/'backend_restore_r15'
def read(p):
    p=D/p;return json.loads(p.read_text()) if p.exists() else None
owned=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:c=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    except (OSError,UnicodeError):continue
    if str(D) in c and ('python' in c or c.startswith('flock ')):owned.append(dict(pid=int(p.name),cmd=c[:1200]))
v=dict(checked_epoch=time.time(),pipeline=read('pipeline_state.json'),controller=read('full_controller_recovery_launch.json') or read('full_controller_launch.json'),progress=read('main/progress.json'),GPU=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip(),owned_live_processes=owned,selfcheck=read('GPU_selfcheck.json'),bootstrap=read('bootstrap_verified.json'),schedule_recovery=read('schedule_recovery_verified.json'),topology=read('main/topology_latest.json'),SGCM=read('main/sgcm_latest.json'),reference=read('main/reference_refresh.json'),log_tail='\n'.join((D/'controller.log').read_text(errors='replace').splitlines()[-2:])[-4000:])
print(json.dumps(v,indent=2))

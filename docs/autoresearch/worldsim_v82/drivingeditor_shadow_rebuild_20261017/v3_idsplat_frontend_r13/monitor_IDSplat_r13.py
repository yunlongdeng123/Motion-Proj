from pathlib import Path
import json,subprocess,time
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');D=R/'idsplat_frontend_r13'
v={'epoch':time.time(),'files':{},'processes':[]}
for name in ['status.json','launch.json','dinov3_launch.json','dinov3_validation/status.json','pointset_QUERY_gate.json']:
    p=D/name
    if p.exists():
        value=json.loads(p.read_text());v['files'][name]={k:x for k,x in value.items() if k not in ['models','screen']}
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode();state=(p/'stat').read_text().split()[2]
    except (OSError,UnicodeError):continue
    if state!='Z' and str(R) in cmd:v['processes'].append(dict(pid=int(p.name),state=state,cmd=cmd))
for name in ['frontend.log','dinov3.log']:
    p=D/name
    if p.exists():v[name+'_tail']=p.read_text(errors='replace').splitlines()[-3:]
v['GPU']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
(D/'last_monitor.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))

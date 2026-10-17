from pathlib import Path
import json,subprocess,time
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');D=R/'ownership_rebuild_r14'
def read(p):
    p=D/p
    return json.loads(p.read_text()) if p.exists() else None
owned=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:c=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    except (OSError,UnicodeError):continue
    if str(D) in c and 'python' in c:owned.append(dict(pid=int(p.name),cmd=c[:1000]))
progress=read('main/progress.json');controller=read('full_controller_recovery_launch.json') or read('full_controller_launch.json')
log=D/controller.get('log','controller.log')
gates={}
for key in ['003000','009000']:
    parts={}
    for phase in ['before','after_split']:
        status=read(f'gates/{key}/{phase}/progress.json')
        summary=read(f'gates/{key}/{phase}/summary.json')
        if status:parts[phase]=dict(progress=status,complete=summary is not None)
    for phase in ['split','owner']:
        status=read(f'gates/{key}/{phase}/{phase}.json')
        if status:parts[phase]={k:v for k,v in status.items() if 'provenance' not in k}
    if parts:gates[key]=parts
value=dict(checked_epoch=time.time(),pipeline=read('pipeline_state.json'),controller=controller,progress=progress,owned_live_processes=owned,GPU=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip(),gates=gates,active_log=str(log.relative_to(D)),log_tail='\n'.join(log.read_text(errors='replace').splitlines()[-4:])[-3500:] if log.exists() else None,mask_visualization_frames=len(list((D/'mask_visualization_r2/reviews').glob('*.png'))),mask_visualization_complete=(D/'mask_visualization_r2/verification.json').exists(),ETA_scope='initial size measured G/M/J throughput; densification/audits/checkpoints/final49 rendering not included')
if progress:
    step=progress['global_step'];value['training_only_remaining_hours_initial_probe']=max(13000-step,0)*.354/3600+max(20000-max(step,13000),0)*1.52073/3600+max(22400-max(step,20000),0)*2.13942/3600
print(json.dumps(value,indent=2))

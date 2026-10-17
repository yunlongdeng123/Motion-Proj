from pathlib import Path
import os,signal,json,time,shutil,torch
D=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3/backend_restore_r15')
def cmd(pid):
    try:return (Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    except OSError:return ''
owned=[]
for p in Path('/proc').iterdir():
    if p.name.isdigit():
        c=cmd(int(p.name))
        if str(D) in c and ('python' in c or c.startswith('flock ')):owned.append(dict(pid=int(p.name),cmd=c))
owned.sort(key=lambda x:0 if 'controller.py' in x['cmd'] else 1)
for v in owned:
    try:os.kill(v['pid'],signal.SIGSTOP)
    except ProcessLookupError:pass
for v in owned:
    try:os.kill(v['pid'],signal.SIGTERM);os.kill(v['pid'],signal.SIGCONT)
    except ProcessLookupError:pass
time.sleep(1)
for v in owned:
    if cmd(v['pid'])==v['cmd']:
        try:os.kill(v['pid'],signal.SIGKILL)
        except ProcessLookupError:pass
progress=json.loads((D/'main/progress.json').read_text());saved=torch.load(D/'main/latest.pt',map_location='cpu');assert saved['global_step']==1,'Do not adopt a later checkpoint with early topology'
failed=D/'engineering_early_topology';failed.mkdir(exist_ok=False)
assert (D/'main').resolve().parent==D.resolve()
(D/'main').rename(failed/'main')
for name in ['controller.log','full_controller_launch.json','pipeline_state.json']:
    if (D/name).exists():shutil.copy2(D/name,failed/name)
(D/'main').mkdir();shutil.copy2(failed/'main/latest.pt',D/'main/latest.pt')
lines=(failed/'main/metrics.jsonl').read_text().splitlines();first=[json.loads(x) for x in lines if json.loads(x)['global_step']==1];assert len(first)==1
(D/'main/metrics.jsonl').write_text(json.dumps(first[0])+'\n');(D/'main/progress.json').write_text(json.dumps(first[0],indent=2))
receipt=dict(status='EARLY_GENERIC_DENSIFY_SCHEDULE_REPAIR',last_logged=progress['global_step'],executed_failed_range=[progress['global_step'],progress['global_step']+49],adopted_checkpoint=1,adopted_bootstrap_update_already_counted=True,replayed_tail_updates_will_be_charged_separately=True,model_Adam_RNG_unchanged=True,owned_remaining=[v for v in owned if cmd(v['pid'])==v['cmd']],GPU_shutdown=False)
(failed/'receipt.json').write_text(json.dumps(receipt,indent=2));(D/'pipeline_state.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))

"""Only final failed-front-end closeout may request AutoDL power-off."""
from pathlib import Path
import os,json,sys,subprocess,fcntl,datetime,shutil,errno
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');D=R/'idsplat_frontend_r13'
def main(poweroff=False):
    review=json.loads((D/'root_pointset_review.json').read_text());pub=json.loads((D/'github_publish.json').read_text())
    assert review['status']=='ROOT_POINTSET_REVIEW_NOT_ACCEPTED_FOR_TRAINING' and review['scene_training_updates']==0
    assert pub['status']=='PUSHED_AND_BRANCH_REF_SOURCE_VERIFIED' and pub['commit']=='3986935b7f7111122be0c4bbcca653ed3bf076fd'
    owned=[];other=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:
            stat=(p/'stat').read_text().split();cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode();cwd=str((p/'cwd').resolve())
        except (OSError,UnicodeError):continue
        if stat[2]=='Z':continue
        if str(R) in cmd or (cwd.startswith(str(R)) and any(s in cmd for s in ['python','flock','bash'])):owned.append(dict(pid=int(p.name),cmd=cmd))
        elif any(s in cmd for s in ['python','flock']) and ('/root/autodl-tmp' in cmd or cwd.startswith('/root/autodl-tmp')) and not any(s in cmd for s in ['ipykernel_launcher','jupyter','vscode-server','http.server']):other.append(dict(pid=int(p.name),cmd=cmd))
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
    cron=subprocess.run(['crontab','-l'],text=True,capture_output=True).stdout if shutil.which('crontab') else ''
    for path in [Path('/etc/crontab'),*Path('/etc/cron.d').glob('*')]:
        if path.is_file():cron+='\n'+path.read_text(errors='replace')
    pending_cron=[line for line in cron.splitlines() if line.strip() and not line.lstrip().startswith('#') and any(t in line for t in ['autodl-tmp','train','render','eval','controller','idsplat'])]
    atq=subprocess.run(['atq'],capture_output=True,text=True).stdout.strip() if shutil.which('atq') else ''
    checkpoint={}
    for a in ['A_box','B_semantic_motion']:
        for f in ['latest.pt','gate_001500.pt']:
            p=R/'ownership_controls_r12_sam3'/a/f;assert p.is_file() and p.stat().st_size>100000000;checkpoint[a+'/'+f]=p.stat().st_size
    value=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='IDLE_PREFLIGHT_PASSED',owned_live=owned,other_potential_work=other,GPU=gpu,pending_cron_jobs=len(pending_cron),pending_at_jobs=bool(atq),checkpoints_preserved=checkpoint,commit=pub['commit'],human_shutdown_authorized=True,automation_paused=True)
    assert not owned and not other and not gpu and not pending_cron and not atq,value
    with open('/root/autodl-tmp/drivingeditor_repro/campaign/gpu_slot.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);value['GPU_slot_exclusive_free']=True
        (D/'power_preflight.json').write_text(json.dumps(value,indent=2));print(json.dumps(value),flush=True)
        if poweroff:
            # Recheck after acquiring slot. Never shut down during unrelated work.
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            value.update(status='AUTODL_SHUTDOWN_REQUESTED',command='/usr/bin/shutdown')
            (D/'shutdown_receipt.json').write_text(json.dumps(value,indent=2));os.sync()
            print('AUTODL_SAFE_SHUTDOWN_REQUEST',flush=True)
            try:result=subprocess.run(['/usr/bin/shutdown'],capture_output=True,text=True,timeout=45)
            except OSError as exc:
                if exc.errno!=errno.ENOEXEC:raise
                # AutoDL's executable text command has no shebang on this host.
                value['effective_command']=['/bin/bash','/usr/bin/shutdown']
                result=subprocess.run(value['effective_command'],capture_output=True,text=True,timeout=45)
            value.update(command_returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
            (D/'shutdown_receipt.json').write_text(json.dumps(value,indent=2));os.sync();print(json.dumps(value),flush=True)
            assert result.returncode==0
if __name__=='__main__':main('--shutdown' in sys.argv)

"""Freshly audit and request the already-authorized AutoDL power backend.

Default is read-only. --request requires a completed, pushed closeout receipt.
No Trash deletion, wrapper execution, broad process killing, or GPU imports.
"""
import argparse,datetime,fcntl,json,os,signal,subprocess,sys
from pathlib import Path
F=Path('/root/autodl-tmp/drivingeditor_repro/campaign/followup_scene_prep_20261013')
P=F/'metadata/final_closeout_20261013'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(name,d):
    P.mkdir(parents=True,exist_ok=True)
    target=P/name;temp=target.with_suffix(target.suffix+'.tmp')
    with temp.open('w') as f:
        json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(str(temp),str(target))
def argv(pid):
    try:return [x.decode() for x in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if x]
    except FileNotFoundError:return []
def parents():
    seen=set();pid=os.getpid()
    while pid>1 and pid not in seen:
        seen.add(pid)
        try:pid=int((Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[1])
        except FileNotFoundError:break
    return seen
args=argparse.ArgumentParser();args.add_argument('--request',action='store_true');opts=args.parse_args()
audit=subprocess.run([sys.executable,str(F/'tools/audit_pre_power_20261013.py')],capture_output=True,text=True,timeout=20,close_fds=False)
assert audit.returncode==0,audit.stderr
d=json.loads(audit.stdout);save('fresh_pre_power_audit.json',d)
assert not d['any_owned_process_running']
assert d['GPU_compute_processes']=='' and d['shared_GPU_slot_free'] is True
supervisors=d['supervisord_candidates'];assert len(supervisors)==1
sup=supervisors[0]
expected=['/bin/supervisord','-c','/init/supervisor/supervisor.ini']
assert sup['pid']==857 and sup['ppid']==1 and sup['command']==expected
vendor=d['shutdown_vendor_script']
lines=[s.strip() for s in vendor.splitlines() if s.strip() and not s.strip().startswith('#')]
assert len(lines)==3 and lines[0]=='rm -rf /root/.local/share/Trash'
assert lines[1].startswith('echo "[$(date ') and lines[1].endswith('> /proc/1/fd/1')
assert lines[2]=="ps -ef | grep supervisord | grep -v grep | awk '{print $2}' | xargs kill"
allowed_ancestors=parents();unexpected=[]
for r in d['all_processes']:
    pid=r['pid'];cmd=r['command']
    if pid in allowed_ancestors or pid==d['audit_pid'] or pid==857:continue
    if cmd[:2]==['bash','/init/boot/boot.sh']:continue
    if cmd[:2]==['/root/miniconda3/bin/python','/root/miniconda3/bin/jupyter-lab']:continue
    if cmd[:2]==['/root/miniconda3/bin/python','/root/miniconda3/bin/tensorboard']:continue
    if cmd==['autopanel','serve','--work-dir=/root/autodl-tmp','--cache-dir=/root/autodl-tmp']:continue
    if cmd in [['proxy','-c','/init/proxy/proxy-hot-backup.ini'],['proxy','-c','/init/proxy/proxy.ini']]:continue
    if cmd[0]=='/root/miniconda3/lib/python3.10/site-packages/tensorboard_data_server/bin/server' and '--logdir=/root/tf-logs' in cmd:continue
    if cmd[0].startswith('sshd:') and 'listener' in ' '.join(cmd):continue
    if r['state'] in ('Z','X'):continue
    unexpected.append(r)
assert not unexpected,('Unrelated or unrecognized processes: ',unexpected)
checks={'observed_utc':now(),'status':'PRECONDITIONS_PASSED_SIGNAL_NOT_SENT','host':'wm-vgpu-1011','vendor_Trash_deletion_skipped':True,
        'supervisord_pid':857,'supervisord_cmdline':expected,'no_owned_jobs':True,'no_unrelated_work_detected':True,'GPU_compute_empty':True,
        'signal_sent':False,'deletion_performed':False,'independent_platform_power_verified':False}
if not opts.request:
    print(json.dumps(checks,ensure_ascii=False),flush=True);sys.exit(0)
receipt=json.loads((P/'saved_and_pushed_prerequisites.json').read_text())
assert receipt['status']=='RESULTS_SAVED_AND_PUSHED_READY_FOR_AUTHORIZED_SHUTDOWN'
assert receipt['total_training_steps']==180000 and receipt['checkpoint_bundles_checked']==18
assert receipt['red_visual_review_complete'] and receipt['local_offline_package_verified'] and receipt['repository_push_verified']
assert receipt['shutdown_user_already_authorized'] and receipt['human_GN11_verdict'] is None
for rel in ['reports/final_closeout_20261013/DrivingEditor_final_report.html','reports/final_closeout_20261013/DrivingEditor_final_summary.json',
            'reports/red_gn11_30k_saved_cpu_20261013/engineering-visual-review.json','reports/red_gn11_30k_saved_cpu_20261013/results-report.md']:
    assert (F/rel).is_file(),rel
if (P/'shutdown_request.json').exists():
    prior=json.loads((P/'shutdown_request.json').read_text())
    assert prior['status']=='PRE_SIGNAL_FAILED' and prior['signal_sent'] is False,'Do not repeat a sent or uncertain power request'
with (F.parent/'gpu_slot.lock').open('a') as slot:
    fcntl.flock(slot,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert argv(857)==expected,'supervisord identity changed'
    assert Path('/usr/bin/shutdown').read_text()==vendor,'vendor behavior changed'
    with open('/proc/1/fd/1','w') as log:
        log.write('[%s] user-authorized DrivingEditor completed; request shutdown backend; Trash deletion skipped!\n'%now());log.flush()
    checks.update(status='POWER_DISPATCH_PREPARED',signal_sent=None)
    save('shutdown_request.json',checks)
    try:os.kill(857,signal.SIGTERM)
    except OSError as error:
        checks.update(status='PRE_SIGNAL_FAILED',signal_sent=False,error=str(error));save('shutdown_request.json',checks);raise
    checks.update(status='AUTHORIZED_SHUTDOWN_REQUEST_SENT',signal_sent=True,signal='SIGTERM',signal_sent_utc=now(),
                  independent_platform_power_verified=False,platform_status='unavailable',wrapper_executed=False)
    # Flush the actual successful kill() acknowledgement before backend teardown.
    print(json.dumps(checks,ensure_ascii=False),flush=True)
    save('shutdown_request.json',checks)

"""结果取回并推送确认后才关闭本次带卡实例。"""
from pathlib import Path
import datetime,json,subprocess,os,signal,sys

BASE=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/editable_ownership_20261030')
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
DOC=REPO/'docs/autoresearch/v43_ownership_20261028/editable_ownership_20261030'

def run(argv): return subprocess.check_output(argv,text=True).strip()

def supervisor():
    found=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
            if cmd=='/bin/supervisord -c /init/supervisor/supervisor.ini':
                found.append(dict(pid=int(p.name),start_ticks=(p/'stat').read_text().split(') ',1)[1].split()[19],command=cmd))
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    assert len(found)==1
    return found[0]

def active_jobs():
    jobs=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:
            argv=(p/'cmdline').read_bytes().split(b'\0')
            exe=Path(argv[0].decode()).name if argv else ''
            cmd=b' '.join(argv).decode()
            if ('python' in exe and not any(x in cmd for x in ('tensorboard','jupyter-lab'))) or exe in ('torchrun','accelerate','xz','tar','scp','sftp-server'):
                jobs.append(dict(pid=int(p.name),command=cmd))
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return jobs

assert not active_jobs(),active_jobs()
assert run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'])==''
gpu=run(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader'])
assert 'RTX 4080 SUPER' in gpu
if '--poweroff' in sys.argv:
    receipt=json.loads((BASE/'closeout.json').read_text())
    assert receipt['delivery_complete'] and receipt['git_push_verified']
    current=supervisor(); assert current==receipt['supervisor']
    receipt['shutdown_signal_sent_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    receipt['shutdown_method']='AutoDL main supervisor SIGTERM'
    (BASE/'closeout.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    os.sync()
    print(json.dumps(dict(signal_sent=True,supervisor=current,utc=receipt['shutdown_signal_sent_utc'])),flush=True)
    os.kill(current['pid'],signal.SIGTERM)
else:
    for name in ['probe_anchors_r1','support_geometry_r1','refit_raw113_r2']:
        data=json.loads((BASE/name/'summary.json').read_text())
        assert data.get('status')=='complete', (name,data.get('status'))
    assert json.loads((DOC/'report_validation.json').read_text())['passed']
    transfer=json.loads((DOC/'raw_archive_receipt.json').read_text())
    assert transfer['raw_transfer_verified']
    assert json.loads((BASE/'source_deployment.json').read_text())['passed']
    head=run(['git','-C',str(REPO),'rev-parse','HEAD'])
    branch=run(['git','-C',str(REPO),'branch','--show-current'])
    pushed=run(['git','-C',str(REPO),'-c','http.proxy=socks5h://127.0.0.1:42130','ls-remote','origin','refs/heads/'+branch]).split()[0]
    assert head==pushed
    dirty=run(['git','-C',str(REPO),'status','--short'])
    assert dirty=='M scripts/run_worldsim_v72_e3_render_bridge.py',dirty
    auth_path=BASE/'authorization.json'
    auth=json.loads(auth_path.read_text())
    if not (BASE/'authorization_at_execution.json').exists():
        (BASE/'authorization_at_execution.json').write_bytes(auth_path.read_bytes())
    auth.update(allow_gpu_probe=False,allow_child_refit=False,allow_classifier_training=False,
                resume_old_queue=False,gpu_execution_hold=True,
                phase='diagnostics_delivered_representation_repair_unproven')
    auth_path.write_text(json.dumps(auth,ensure_ascii=False,indent=2))
    parent_auth=json.loads((BASE.parent/'authorization.json').read_text())
    assert parent_auth.get('gpu_execution_hold') and not parent_auth.get('allow_v43_training')
    receipt=dict(task='WS-V43-EDITABLE-OWNERSHIP-20261030',delivery_complete=True,
        git_head=head,git_branch=branch,git_push_verified=True,unrelated_dirty_preserved=dirty,
        raw_transfer=transfer,child_updates_total=200,classifier_updates=0,original_model_updates=0,
        active_jobs=[],gpu_compute_processes=[],gpu=gpu,supervisor=supervisor(),
        process_snapshot=run(['ps','-eo','pid,ppid,etime,args','--sort=pid']),
        shutdown_authorized=True,prepared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (BASE/'closeout.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k!='process_snapshot'}))

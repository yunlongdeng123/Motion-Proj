"""所有C2依赖收尾后才调用AutoDL官方关机机制，省略无关Trash删除。"""
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess

RUN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a37_C2_20261103_r1')
ready=json.loads((RUN/'closeout_ready.json').read_text())
for key in ('all_required_work_complete','local_archive_verified','git_remote_head_verified',
            'no_remote_file_or_environment_dependencies','no_other_active_or_queued_tasks','gpu_mode_confirmed'):
    assert ready.get(key) is True,key
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
assert not gpu,('GPU compute still active',gpu)
supervisor=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    except (OSError,PermissionError):continue
    if cmd.startswith('/bin/supervisord ') or cmd.startswith('/usr/bin/supervisord '):supervisor.append(int(p.name))
assert len(supervisor)==1,supervisor
receipt=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),supervisor_pid=supervisor[0],
    method='AutoDL /usr/bin/shutdown underlying supervisord SIGTERM; unrelated Trash cleanup omitted',
    ready_checks=ready,supervisor_shutdown_signal_sent=False)
path=RUN/'shutdown_execution_receipt.json'
path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
os.kill(supervisor[0],signal.SIGTERM)
receipt['supervisor_shutdown_signal_sent']=True
path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False),flush=True)

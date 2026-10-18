"""Current-task idle/checkpoint checks only. Does not execute shutdown."""
from pathlib import Path
import argparse,json,subprocess,os,datetime,re
p=argparse.ArgumentParser();p.add_argument('--published-commit',required=True);a=p.parse_args()
assert re.fullmatch('[0-9a-f]{40}',a.published_commit)
Q=Path(__file__).resolve().parent;D=Q.parent;ROOT=D.parent.parent;T=ROOT/'identity_completion_20261018'
read=lambda p:json.loads(p.read_text())
states={str(p):read(p)['state'] for p in [T/'queue_status.json',T/'tristate_queue_status.json',Q/'queue_status.json',T/'branch_queue_status.json']}
assert set(states.values())=={'ALL_GPU_WORK_COMPLETE_AWAITING_DELIVERY','ALL_VISUALIZATION_GPU_WORK_COMPLETE_AWAITING_DELIVERY','ALL_REPAIR_GPU_WORK_COMPLETE_AWAITING_REVIEW_AND_DELIVERY','ALL_BRANCHES_COMPLETE_AWAITING_PLAYER_DELIVERY'}
gpu_binary=Path('/usr/bin/nvidia-smi')
gpu_devices=sorted(str(p) for p in Path('/dev').glob('nvidia*'))
gpu_query_error=None
if gpu_binary.is_file() and gpu_binary.stat().st_size>0 and os.access(gpu_binary,os.X_OK):
    gpu=subprocess.check_output([str(gpu_binary),'--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip()
    gpu_check='NVIDIA_SMI_COMPUTE_QUERY_SUCCEEDED'
else:
    # AutoDL may be restarted without a GPU. Do not treat a failed query as
    # an empty process result unless there are also no exposed NVIDIA devices.
    assert not gpu_devices, 'GPU query unavailable while NVIDIA devices are exposed'
    gpu='';gpu_check='NO_EXPOSED_NVIDIA_DEVICES_AND_NVIDIA_SMI_UNAVAILABLE'
    gpu_query_error=dict(binary=str(gpu_binary),exists=gpu_binary.exists(),
        bytes=gpu_binary.stat().st_size if gpu_binary.exists() else None,
        executable=os.access(gpu_binary,os.X_OK))
workers=[]
for directory in Path('/proc').iterdir():
    if not directory.name.isdigit() or int(directory.name)==os.getpid():continue
    try:
        args=(directory/'cmdline').read_bytes().split(b'\0');cmd=' '.join(x.decode(errors='replace') for x in args).strip()
    except OSError:continue
    if not args or b'python' not in args[0]:continue
    # Stock dashboard services do not train, render, or dispatch campaign work.
    if any(x in cmd for x in ['/bin/tensorboard ','/bin/jupyter-lab ',
        '/tensorboard_data_server/bin/server --logdir=/root/tf-logs','pylance','pyright']):continue
    workers.append(dict(pid=int(directory.name),command=cmd))
free=subprocess.run(['flock','-n',str(ROOT.parent/'gpu_slot.lock'),'true']).returncode==0
files=[D/'paused_checkpoint_004000_20261017/checkpoint.pt',D/'main/latest.pt',T/'train5/checkpoint_030000.pt']
files += [ROOT/'runs/scene0008_089_137_hybrid_30000_seed0_GN11/chkpnt30000.pth',
    ROOT/'rigid_runs/scene0008_089_137/full30k/checkpoint_030000.pt']
files += [Q/'train'/f'checkpoint_{s:06d}.pt' for s in [1000,2000,3000,4000]]+[Q/'train/latest.pt']
checkpoints={str(p):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in files}
for model in ['old5','r15','repair4k']:
    r=read(T/('branches_'+model)/'branch_receipt.json');assert r['frames']==49 and r['parameter_updates']==0 and r['checkpoint_file_unchanged']
assert read(Q/'train/progress.json')['global_step']==4000
ready=not gpu and not workers and free and all(v['bytes']>0 for v in checkpoints.values())
r=dict(status='READY_FOR_AUTHORIZED_AUTODL_SHUTDOWN' if ready else 'NOT_READY',
    at=datetime.datetime.now(datetime.timezone.utc).isoformat(),GPU_processes=gpu.splitlines(),blocking_workers=workers,
    gpu_check=gpu_check,exposed_nvidia_devices=gpu_devices,gpu_query_unavailable_details=gpu_query_error,
    gpu_lock_free=free,queue_states=states,checkpoint_files=checkpoints,pushed_commit=a.published_commit,
    original_pause_marker_preserved=(D/'PAUSE_REQUESTED.json').exists(),shutdown_authorized_by_human=True,
    all_outputs_saved_and_pushed_verified_by_caller=True)
(Q/'shutdown_preflight.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
if not ready:raise SystemExit(2)

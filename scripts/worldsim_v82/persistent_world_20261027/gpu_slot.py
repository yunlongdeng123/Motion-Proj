"""单卡工作进程互斥；发现其他任务的CUDA进程则等待，不杀其他作业。"""
from contextlib import contextmanager
from pathlib import Path
import fcntl,json,os,subprocess,time
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')


@contextmanager
def exclusive():
    with (ROOT/'gpu_slot.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        while True:
            query=subprocess.run(['nvidia-smi','--id=0','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True)
            if query.returncode:raise RuntimeError('无法核对GPU活动进程：'+query.stderr[-300:])
            others=[int(x.strip()) for x in query.stdout.splitlines() if x.strip().isdigit() and int(x.strip())!=os.getpid()]
            if not others:break
            (ROOT/f'gpu_slot_wait_{os.getpid()}.json').write_text(json.dumps(dict(time=time.time(),waiting_pid=os.getpid(),other_gpu_pids=others)))
            time.sleep(15)
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)

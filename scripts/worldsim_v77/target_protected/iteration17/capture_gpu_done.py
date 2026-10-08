"""GPU阶段完成后保存进程与资源证据；不关机，不启动后续任务。"""
from common import *
import datetime, shutil, socket, subprocess

s=read(O/'drive_state.json');controller=read(O/'controller_state.json')
assert s['stage']=='DELETE_complete_pending_structure_review' and len(s['completed'])==40
assert controller['GPU_jobs']==0
compute=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],
                       check=True,capture_output=True,text=True).stdout.strip()
assert not compute,compute
process=subprocess.run(['ps','-p',str(s['pid']),'-o','stat=,args='],capture_output=True,text=True).stdout.strip()
assert not process or process.startswith('Z'),process
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],
                   check=True,capture_output=True,text=True).stdout.strip()
quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip();disk=shutil.disk_usage(O)
record={'host_alias':'wm-vgpu-1008','hostname':socket.gethostname(),
        'finished_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'GPU_processes_idle':True,'nvidia_compute_processes':compute,'DELETE_process':process,
        'nvidia_GPU_name_total_used_utilization':gpu,'cpu_cgroup_quota':quota,
        'disk_total_gib':disk.total/2**30,'disk_free_gib':disk.free/2**30,
        'CUDA_jobs_after_batch':0,'training_steps':0,'power_action':None}
dump(O/'gpu_runtime.json',record)
print('GPU_IDLE_CONFIRMED',gpu,record['disk_free_gib'],flush=True)

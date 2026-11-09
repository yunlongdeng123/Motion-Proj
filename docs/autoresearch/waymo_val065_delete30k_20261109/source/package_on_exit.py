"""等待既有控制器退出再打包；不轮询，不训练，不操作电源。"""
import os
import select
import subprocess
import pathlib
import sys

pid=int(sys.argv[1]);root=pathlib.Path('/root/autodl-tmp/drivingeditor_repro/campaign/waymo_val065_delete30k_20261109_r1')
try:
    fd=os.pidfd_open(pid);poll=select.poll();poll.register(fd,select.POLLIN);poll.poll();os.close(fd)
except ProcessLookupError:pass
if not (root/'completed.json').is_file():raise RuntimeError('Existing deletion run did not complete; retained logs and partial outputs')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
source=pathlib.Path(__file__).with_name('package_delete065_30k.py')
subprocess.run(['/root/autodl-tmp/envs/drivingeditor/bin/python','-B',str(source)],env=env,check=True)

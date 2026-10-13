"""本地透传stdin到指定SSH主机；只把远端JSON回执保存到work。"""
from pathlib import Path
import argparse
import json
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--step', type=int, default=10000, choices=(10000, 20000, 30000))
parser.add_argument('--expected-bytes', type=int)
args = parser.parse_args()
directory = Path(__file__).resolve().parent
source = directory / 'checkpoint_cpu_structure_audit.py'
receipt_path = directory / ('checkpoint_%06d_CPU_structure_receipt.json' % args.step)
if receipt_path.exists():
    raise FileExistsError('保留已有CPU回执，不覆盖: ' + str(receipt_path))
command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'wm-vgpu-1011',
           'env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 nice -n 10 /root/autodl-tmp/envs/drivingeditor/bin/python -B - --step ' + str(args.step)
           + (' --expected-bytes ' + str(args.expected_bytes) if args.expected_bytes is not None else '')]
result = subprocess.run(command, input=source.read_bytes(), capture_output=True, timeout=180)
stdout = result.stdout.decode('utf-8', errors='strict')
if result.stderr:
    (directory / ('checkpoint_%06d_CPU_structure_stderr.txt' % args.step)).write_bytes(result.stderr)
receipt = json.loads(stdout)
receipt['inspection_transport'] = {'ssh_alias': 'wm-vgpu-1011', 'stdin_script': True,
                                   'remote_script_written': False, 'ssh_returncode': result.returncode}
with receipt_path.open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, ensure_ascii=False, indent=2)
print(json.dumps({'status': receipt['status'], 'receipt_path': str(receipt_path), 'remote_returncode': result.returncode,
                  'actual_step': receipt.get('actual_step'), 'wall_seconds': receipt.get('wall_seconds'),
                  'cuda_initialized': receipt.get('cuda_initialized'), 'error': receipt.get('error')}, ensure_ascii=False), flush=True)
raise SystemExit(result.returncode)

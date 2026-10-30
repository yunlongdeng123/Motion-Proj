"""本任务SSH传输/执行辅助；参数不经本地shell插值，无电源操作。"""
import argparse
import base64
from pathlib import Path
import shlex
import subprocess

HOST = 'wm-vgpu-1011'
ROOT = '/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030'

def remote_python(source, python='python3'):
    encoded = base64.b64encode(source.encode('utf8')).decode('ascii')
    code = 'import base64;exec(compile(base64.b64decode(' + repr(encoded) + '),"remote_task","exec"))'
    return subprocess.run(['ssh', '-o', 'ConnectTimeout=15', HOST, shlex.quote(python) + ' -u -c ' + shlex.quote(code)], check=True)

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--file',type=Path);p.add_argument('--code');p.add_argument('--python',default='python3');a=p.parse_args()
    remote_python(a.file.read_text(encoding='utf-8-sig') if a.file else a.code,a.python)

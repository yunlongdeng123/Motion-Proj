"""镜像已完成的本地照片审计，只写本轮 campaign 的 photo_audit 子目录。"""
from pathlib import Path
import base64
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
LOCAL = HERE.parent / 'photo_audit'
ROOT = '/root/autodl-tmp/drivingeditor_repro/campaign/road_source_fidelity_20261122_r2'
REMOTE = ROOT + '/photo_audit'
PYTHON = '/root/autodl-tmp/envs/motionproj/bin/python'


def remote_python(code):
    encoded = base64.b64encode(code.encode()).decode()
    return subprocess.run(['ssh', 'wm-vgpu-1011', PYTHON + " -c \"import base64;exec(base64.b64decode('" + encoded + "'))\""],
                          check=True, capture_output=True, text=True).stdout


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    files = {p.relative_to(LOCAL).as_posix(): dict(bytes=p.stat().st_size, sha256=digest(p))
             for p in LOCAL.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    assert len(files) >= 70
    # Existing root/masks and root/probe_plan.json are intentionally outside target.
    inventory_code = """from pathlib import Path
import hashlib,json
p=Path(%r)
print(json.dumps({f.relative_to(p).as_posix(): {'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in p.rglob('*') if f.is_file() and '__pycache__' not in f.parts}))
""" % REMOTE
    prior = json.loads(remote_python(inventory_code))
    unequal = [p for p in set(prior) & set(files) if prior[p] != files[p]]
    if unequal:
        raise RuntimeError('Preserve unequal remote files before a new mirror: ' + repr(unequal))
    unknown = set(prior) - set(files)
    if unknown:
        raise RuntimeError('Remote audit has additional user files: ' + repr(sorted(unknown)))
    remote_python("from pathlib import Path; p=Path(%r); assert str(p.resolve())==%r; p.mkdir(parents=True,exist_ok=True)" % (REMOTE, REMOTE))
    if prior != files:
        subprocess.run(['scp', '-q', '-r', str(LOCAL), 'wm-vgpu-1011:' + ROOT + '/'], check=True)
    after = json.loads(remote_python(inventory_code))
    if after != files:
        raise RuntimeError('Mirror verification failed')
    receipt = dict(complete=True, local_root=str(LOCAL), remote_root=REMOTE,
                   files=len(files), total_bytes=sum(x['bytes'] for x in files.values()),
                   exact_file_set=True, exact_file_content=True,
                   protected_remote_paths=[ROOT+'/masks', ROOT+'/probe_plan.json'],
                   remote_writes_outside_new_campaign=False, GPU_calls=0, power_operations=0)
    (HERE / 'photo_audit_mirror_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()

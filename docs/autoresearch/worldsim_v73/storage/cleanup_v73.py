"""清理已退役依赖，保留环境恢复信息和逐项字节记录。"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

root = Path('/root/autodl-tmp')
out = root / 'cleanup_manifests/worldsim-v73-20260907'
retired = ['worldsim-v32-asset-harvester', 'worldsim-v33-sam2',
           'worldsim-v61-gaussianworld', 'worldsim-v61-hy3d-omni', 'worldsim-v61-irwm']
targets = [root / 'envs' / name for name in retired] + [
    root / 'third_party/worldsim_v32/asset-harvester/checkpoints',
    root / 'hf_cache/worldsim_v32_asset_harvester',
    root / 'pip_cache/http-v2',
    root / 'conda_pkgs/cache',
]
targets += sorted((root / 'conda_pkgs').glob('*.conda'))
targets += sorted((root / 'conda_pkgs').glob('*.tar.bz2'))
parser = argparse.ArgumentParser()
parser.add_argument('--apply', action='store_true')
args = parser.parse_args()
out.mkdir(parents=True, exist_ok=True)

def save(name, value):
    (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

rows = []
for path in targets:
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or resolved == root or path.is_symlink():
        raise RuntimeError(f'清理目标不是数据盘中的独立路径：{path}')
    if path.exists():
        size = int(subprocess.check_output(['du', '-sx', '-B1', str(path)], text=True).split()[0])
        rows.append({'path': str(path), 'allocated_bytes': size})

for name in retired:
    env = root / 'envs' / name
    packages = []
    for meta in sorted((env / 'conda-meta').glob('*.json')):
        data = json.loads(meta.read_text())
        packages.append({k: data.get(k) for k in ('name', 'version', 'build', 'channel', 'subdir')})
    sites = list((env / 'lib').glob('python*/site-packages'))
    pip = sorted({f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions(path=sites)})
    save(name + '-packages.json', {'prefix': str(env), 'conda': packages, 'python_distributions': pip})

save('plan.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
     'authorization': '用户要求清理后续研究暂时不用的环境、权重、产物；仅清理明确退役依赖与下载缓存',
     'retained': ['all data', 'all runs', 'all current eas_vggt weights', 'motionproj',
                  'adgs', 'drivestudio', 'worldsim-v72-pointr', 'worldsim-v72-lidar4d'],
     'targets': rows})
print(json.dumps({'mode': 'apply' if args.apply else 'plan', 'targets': rows,
                  'allocated_gib': sum(r['allocated_bytes'] for r in rows) / 2**30}, ensure_ascii=False), flush=True)
if args.apply:
    # 检查保留环境没有通过符号链接依赖待删前缀；硬链接删除单一目录项不影响保留文件。
    for env in (root / 'envs').iterdir():
        if env.name in retired:
            continue
        for parent, dirs, files in os.walk(env, followlinks=False):
            for name in dirs + files:
                p = Path(parent) / name
                if p.is_symlink() and any(p.resolve().is_relative_to(t) for t in targets):
                    raise RuntimeError(f'保留环境依赖待清理路径：{p}')
    before = shutil.disk_usage(root)
    for row in rows:
        path = Path(row['path'])
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        row['deleted'] = True
        print('removed ' + str(path), flush=True)
    os.sync()
    after = shutil.disk_usage(root)
    result = {'targets': rows, 'before_free_bytes': before.free, 'after_free_bytes': after.free,
              'reclaimed_bytes': after.free - before.free, 'all_data_and_runs_retained': True,
              'environment_versions_saved': True}
    save('result.json', result)
    print(json.dumps(result), flush=True)

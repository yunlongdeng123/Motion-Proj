"""只读取回12张既有原照与资格输入，复用已验证本地缓存。"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import importlib.util
import json
import shutil
import time

OUT = Path(__file__).resolve().parent
R2 = OUT.parents[1] / 'source_fidelity_20261122_r2/photo_audit'
RAWS = [86, 88, 90, 92, 94, 96, 99, 100, 104, 110, 115, 117]


def main():
    start = time.monotonic()
    dep = OUT / 'code_dependencies'
    dep.mkdir(parents=True, exist_ok=True)
    for name in ('retrieve_inputs.py', 'audit_native_photos.py'):
        shutil.copy2(R2 / name, dep / name)
    for raw in RAWS:
        cached = R2 / '_inputs' / f'{raw:03d}'
        dest = OUT / '_inputs' / f'{raw:03d}'
        if cached.exists():
            dest.mkdir(parents=True, exist_ok=True)
            for p in cached.iterdir():
                if p.is_file() and not (dest / p.name).exists():
                    shutil.copy2(p, dest / p.name)
    (OUT / '_inputs').mkdir(parents=True, exist_ok=True)
    shutil.copy2(R2 / '_inputs/patch_02.npz', OUT / '_inputs/patch_02.npz')
    spec = importlib.util.spec_from_file_location('prior_retrieve', dep / 'retrieve_inputs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = OUT
    with ThreadPoolExecutor(max_workers=3) as pool:
        files = sum(list(pool.map(module.retrieve, RAWS)), [])
    value = dict(complete=True, files=files, raw_frames=RAWS,
                 reuse='r2 verified cache copied byte-for-byte where present',
                 seconds=time.monotonic()-start, GPU_calls=0, optimizer_updates=0)
    (OUT / 'retrieval_receipt.json').write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(complete=True, files=len(files), seconds=value['seconds'])))


if __name__ == '__main__':
    main()

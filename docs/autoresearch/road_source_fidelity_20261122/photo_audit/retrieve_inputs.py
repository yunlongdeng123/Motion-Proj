"""Read-only SCP retrieval for the independent native photo audit."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import subprocess
import time

OUT = Path(__file__).resolve().parent
B = '/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1'
C = '/root/autodl-tmp/drivingeditor_repro/campaign'
RAWS = [86, 100, 110, 115, 116, 117, 134, 137]


def retrieve(raw):
    dest = OUT / '_inputs' / f'{raw:03d}'
    dest.mkdir(parents=True, exist_ok=True)
    sources = [
        f'{B}/inputs/image/image_0/0065{raw:03d}.png',
        f'{B}/semantics/{raw:03d}.png',
        f'{B}/repaired_masks_all_r4/{raw:03d}/dynamic_union.png',
        f'{B}/repaired_masks_all_r4/{raw:03d}/labels.png',
        f'{C}/waymo_val065_road_completion_20261111_r1/raw_road_annotations/{raw:03d}.json',
    ]
    expected = [dest / Path(s).name for s in sources]
    missing = [(s, p) for s, p in zip(sources, expected) if not p.exists()]
    if missing:
        subprocess.run(['scp', '-q', *['wm-vgpu-1011:' + s for s, _ in missing], str(dest)], check=True)
    return [dict(raw=raw, remote=s, local=str(p), bytes=p.stat().st_size,
                 sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            for s, p in zip(sources, expected)]


if __name__ == '__main__':
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=3) as pool:
        files = sum(list(pool.map(retrieve, RAWS)), [])
    centers = OUT / '_inputs'
    for cid in ('patch_00', 'patch_02'):
        p = centers / f'{cid}.npz'
        if not p.exists():
            subprocess.run(['scp', '-q', f'wm-vgpu-1011:{C}/road_gaussian_transfer_20261122_r1/source_selection/candidates/{cid}.npz', str(p)], check=True)
    receipt = dict(complete=True, mode='read-only SCP; local CPU audit only', files=files,
                   seconds=time.monotonic() - start, GPU_calls=0, optimizer_updates=0)
    (OUT / 'retrieval_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(dict(complete=True, files=len(files), seconds=receipt['seconds'])))

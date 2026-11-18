"""Read-only export of an independent permission ledger around the frozen T2 patch."""
from pathlib import Path
import argparse, os, json, hashlib
import numpy as np


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU-only')
    p = argparse.ArgumentParser()
    p.add_argument('--ledger', type=Path, required=True)
    p.add_argument('--fixed-source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    with np.load(a.fixed_source) as z:
        wanted = z['source_world_rows'].ravel()
    with np.load(a.ledger) as z:
        grid = z['world_row_grid']
        y, x = np.where(np.isin(grid, wanted))
        if len(y) != len(wanted):
            raise ValueError('patch/source world-row identities differ')
        halo = 18
        y0, x0 = max(int(y.min())-halo, 0), max(int(x.min())-halo, 0)
        y1, x1 = min(int(y.max())+halo+1, grid.shape[0]), min(int(x.max())+halo+1, grid.shape[1])
        keys = [key for key in z.files if z[key].shape[:2] == grid.shape and (z[key].ndim == 2 or key == 'world_xyz')]
        data = {key: z[key][y0:y1, x0:x1] for key in keys}
        base = z['common_base_srgb']
        base_hash = hashlib.sha256(np.ascontiguousarray(base).tobytes()).hexdigest()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, **data, crop_origin_yx=np.asarray([y0, x0]), global_chart_shape=np.asarray(grid.shape),
        source_ledger_path=np.asarray(str(a.ledger)), source_ledger_sha256=np.asarray(hashlib.sha256(a.ledger.read_bytes()).hexdigest()),
        common_base_buffer_sha256=np.asarray(base_hash))
    print(json.dumps(dict(path=str(a.output), shape=data['world_row_grid'].shape, fields=list(data), GPU_operations=0)))


if __name__ == '__main__':
    main()

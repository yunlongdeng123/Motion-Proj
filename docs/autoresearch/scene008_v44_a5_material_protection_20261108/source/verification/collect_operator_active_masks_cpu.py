"""CPU extract exact A>0 masks; preserve every sparse coefficient and source."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from scipy import sparse


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    mpath = args.operator_root / 'operator_manifest.json'
    manifest = json.loads(mpath.read_text())
    assert manifest['status'] == 'PASS_M4_NATIVE_PIXEL_PARENT_OPERATOR_NO_FIT'
    masks, rows = {}, []
    for rec in manifest['records']:
        matrix_path = args.operator_root / rec['operator_file']
        domains_path = args.operator_root / rec['pixel_domains_file']
        A = sparse.load_npz(matrix_path).tocsr()
        assert np.isfinite(A.data).all() and (A.data >= 0).all()
        with np.load(domains_path, allow_pickle=False) as z:
            pixels = z['pixel_index']
            union = z['protected_nonroad'] | z['originalroad']
        assert A.shape == (len(pixels), manifest['variable_parent_count'])
        assert np.array_equal(pixels, np.unique(pixels))
        positive = np.asarray(A.sum(axis=1)).ravel() > 0
        active = np.zeros((320, 480), bool)
        active.ravel()[pixels[positive]] = True
        active &= union
        label = f"raw{rec['raw']:03d}_{rec['state']}"
        masks[label] = active
        rows.append(dict(raw=rec['raw'], state=rec['state'], positive_protection_pixels=int(active.sum()),
                         matrix_file=str(matrix_path), pixel_domains_file=str(domains_path),
                         sparse_rows=A.shape[0], sparse_columns=A.shape[1], nonzeros=A.nnz,
                         operation='Exact nonnegative CSR row sum >0 intersect fixed protection union; no coefficient threshold/drop'))
    assert len(masks) == 8
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / 'exact_A_positive_protection_masks.npz', **masks)
    receipt = dict(schema='INDEPENDENT_EXACT_A_POSITIVE_MASKS_CPU_V1',
                   status='PASS_CPU_EXACT_ACTIVE_MASK_EXTRACTION',
                   operator_manifest=str(mpath), operator_manifest_sha256=sha(mpath),
                   variable_parent_count=manifest['variable_parent_count'], records=rows,
                   GPU_forwards=0, backwards=0, optimizer_updates=0, model_imported=False,
                   sparse_coefficients_changed=False, source_files_modified=False)
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output), records=[{k:r[k] for k in ['raw','state','positive_protection_pixels']} for r in rows])))


if __name__ == '__main__':
    main()

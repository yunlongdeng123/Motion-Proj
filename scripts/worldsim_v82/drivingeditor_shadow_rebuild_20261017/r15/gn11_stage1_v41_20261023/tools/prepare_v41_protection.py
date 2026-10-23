"""最终合成的未知像素用冻结 V1 参考保护；不扩大道路真实 RGB 域。"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np


def main():
    p = argparse.ArgumentParser(); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    if args.output.exists(): raise FileExistsError('保留旧证据')
    meta = json.loads((args.source/'manifest.json').read_text())
    if meta['version'] != 'V41_FIXED_DENSE_EVIDENCE_R3': raise ValueError('需要已验证的车身保护数据')
    shutil.copytree(args.source, args.output)
    for row in meta['frames']:
        path = args.output/'frames'/f"{row['frame']}.npz"
        with np.load(path, allow_pickle=False) as z: data = {k:z[k].copy() for k in z.files}
        data['evidence_protect'] = data['protect'].copy()
        data['unknown_reference_only'] = ~data['visible'] & ~data['protect']
        data['protect'] = ~data['visible']
        row.update(protect_pixels=int(data['protect'].sum()), reference_only_unknown_pixels=int(data['unknown_reference_only'].sum()))
        np.savez_compressed(path, **data)
    meta.update(version='V41_FIXED_DENSE_EVIDENCE_R4', derived_from=str(args.source.resolve()))
    meta['policy']['outside_reliable'] = 'full composite matches frozen current V1; no road GT RGB; vehicle categories normalized separately'
    meta['input_roles']['protection'] = 'all pixels outside reliable road use frozen native full-scene RGB, never independent road GT'
    meta['overlay_legend'] = 'R3 evidence classification retained: yellow unknown pixels now use frozen native composite reference only'
    for key in ('protect_pixels', 'reference_only_unknown_pixels'):
        meta['totals'][key] = sum(row[key] for row in meta['frames'])
    (args.output/'manifest.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(meta['totals'], indent=2))


if __name__ == '__main__': main()

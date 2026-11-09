# Historical Scene008 replay only. Use v44_cli / SceneManifest for new scenes.
"""用固定语义身份包查询 v44 KEEP/DELETE；默认四帧，不训练或求解。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--semantic-sidecar', type=Path, required=True)
    ap.add_argument('--model', type=Path)
    ap.add_argument('--v44-contract', type=Path)
    ap.add_argument('--transform', type=Path)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--scope', choices=['pilot', 'all49'], default='pilot')
    ap.add_argument('--raw-ids', type=int, nargs='+')
    ap.add_argument('--state', choices=['keep', 'delete', 'both'], default='both')
    ap.add_argument('--save-float', action='store_true')
    ap.add_argument('--describe', action='store_true')
    args = ap.parse_args()
    sidecar = args.semantic_sidecar.resolve()
    if args.describe:
        manifest = json.loads(sidecar.read_text(encoding='utf-8'))
        print(json.dumps({key: manifest.get(key) for key in
                         ['schema', 'iteration', 'raw_ids', 'threshold', 'branch_rows',
                          'selection_file', 'selection_sha256', 'gaussian_parameter_updates']},
                         ensure_ascii=False, indent=2))
        return
    if args.output is None:
        ap.error('渲染需要 --output 指向新的目录')
    if args.raw_ids and args.scope == 'all49':
        ap.error('--raw-ids 与 --scope all49 不能同时使用')
    raws = args.raw_ids or (list(range(89, 138)) if args.scope == 'all49' else [94, 105, 130, 137])
    if len(set(raws)) != len(raws) or not set(raws).issubset(range(89, 138)):
        ap.error('raw IDs 必须是无重复的89..137整数')
    states = ['keep', 'delete'] if args.state == 'both' else [args.state]
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        os.environ[key] = '1'
    import numpy as np
    import torch
    from PIL import Image
    from .v44_integration import V44_ROOT, FULL5_MODEL, SemanticActor5V44Integration
    contract = (args.v44_contract or (V44_ROOT / 'contract/input_contract.json')).resolve()
    sys.path.insert(0, str(contract.parent.parent / 'source'))
    from v44_runtime import V44Runtime
    if not torch.cuda.is_available():
        raise RuntimeError('此查询需要原生 CUDA 光栅化环境')
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.70)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    launch = dict(schema='scene008.v44.semantic_query.v1', semantic_sidecar=str(sidecar),
                  model=str((args.model or FULL5_MODEL).resolve()), v44_contract=str(contract),
                  raw_ids=raws, states=states, optimizer_updates=0, calibration_solves=0,
                  geometry_or_colour_updates=0, human_verdict=None, quality_accepted=False)
    write(output / 'launch.json', launch)
    runtime = SemanticActor5V44Integration(V44Runtime(contract), semantic_sidecar_path=sidecar,
        full5_model=args.model or FULL5_MODEL, full5_transform_path=args.transform)
    cpu = lambda value: value.detach().cpu().numpy()
    records = []
    with torch.no_grad():
        for raw in raws:
            folder = output / f'{raw:03d}'
            folder.mkdir()
            for state in states:
                result = runtime.render_scene(raw, delete=state == 'delete')
                rgb = cpu(result['rgb'])
                png = folder / f'{state}.png'
                Image.fromarray(np.uint8(np.clip(rgb.transpose(1, 2, 0), 0, 1) * 255 + .5)).save(png)
                if args.save_float:
                    alpha = cpu(result['alpha'])[0]
                    np.savez_compressed(folder / f'{state}_float.npz', rgb=rgb, alpha=alpha,
                        depth_m=cpu(result['depth_sum'])[0] / np.maximum(alpha, 1e-8) / runtime.scale,
                        group_alpha_T=cpu(result['group_alpha_T']))
                records.append(dict(raw=raw, state=state, delete_actor_ids=[37] if state == 'delete' else [],
                    PNG=str(png.relative_to(output)), PNG_sha256=hashlib.sha256(png.read_bytes()).hexdigest(),
                    meta=result['meta'], group_conservation_maxabs=result['group_conservation_maxabs']))
                print(json.dumps(dict(raw=raw, state=state, complete=len(records))), flush=True)
                del result
            runtime._data.clear()
            torch.cuda.empty_cache()
    write(output / 'query.json', dict(launch, status='RENDERED_FIXED_SEMANTIC_ROWS',
        frames=records, runtime=runtime.receipt(), native_forwards=runtime.render_calls))


if __name__ == '__main__':
    main()

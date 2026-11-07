"""真实四帧集成渲染；评分独立在保存的 float 上完成。"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

CAM = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ROOT = CAM / 'scene008_v44_full5_integration_20261107_r1'
V44 = CAM / 'scene008_v44_rgb_asset_20261107_r1'
MODEL = CAM / 'scene008_full_dynamic_edit_20261107_r1/inference10k'
REPO = Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
OUTPUT = ROOT / 'results/pilot_root_r1'
RAWS = (105, 113, 130, 137)

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def main():
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.70)
    sys.path[:0] = [str(REPO), str(V44 / 'source')]
    from v44_runtime import V44Runtime
    from scripts.worldsim_v82.drivingeditor_dynamic_edit.v44_integration import Full5V44Integration
    code = REPO / 'scripts/worldsim_v82/drivingeditor_dynamic_edit/v44_integration.py'
    assert hashlib.sha256(code.read_bytes()).hexdigest() == '88998b82ad472797b88f4a3fcacf22b9f87af078f5c20eb75774f8eb50b6666c'
    config = json.loads((MODEL / 'effective_config.json').read_text())
    transform_source = Path(config['source_path']) / 'transform_matrix.txt'
    transform = ROOT / 'contract/full5_transform_matrix.txt'
    if transform.exists():
        assert transform.read_bytes() == transform_source.read_bytes()
    else:
        transform.write_bytes(transform_source.read_bytes())
    OUTPUT.mkdir(parents=True, exist_ok=False)
    started = time.time()
    base = V44Runtime(V44 / 'contract/input_contract.json')
    rt = Full5V44Integration(base, MODEL, full5_transform_path=transform)
    frames = []
    cpu = lambda x: x.detach().cpu().numpy()
    with torch.no_grad():
        for raw in RAWS:
            folder = OUTPUT / f'{raw:03d}'
            folder.mkdir()
            record = dict(raw=raw, states={})
            for state in ('keep', 'delete'):
                output = rt.render_scene(raw, delete=state == 'delete')
                rgb, alpha = cpu(output['rgb']), cpu(output['alpha'])[0]
                depth = cpu(output['depth_sum'])[0] / np.maximum(alpha, 1e-8) / rt.scale
                weights = cpu(output['group_alpha_T'])
                assert rgb.shape == (3, 320, 480) and weights.shape == (7, 320, 480)
                if state == 'delete':
                    assert np.max(np.abs(weights[1])) == 0
                np.savez_compressed(folder / f'{state}_float.npz', rgb=rgb, alpha=alpha,
                    depth_m=depth, group_alpha_T=weights,
                    added_owner_rgb_T=cpu(output['added_owner_rgb_T']))
                Image.fromarray(np.uint8(np.clip(rgb.transpose(1, 2, 0), 0, 1) * 255 + .5)).save(folder / f'{state}.png')
                record['states'][state] = dict(meta=output['meta'],
                    group_conservation_maxabs=output['group_conservation_maxabs'],
                    delete_a37_max_alpha=float(np.max(np.abs(weights[1]))) if state == 'delete' else None)
                del output
            write(folder / 'meta.json', record)
            frames.append(record)
            write(OUTPUT / 'progress.json', dict(completed_raws=[item['raw'] for item in frames], native_forwards=rt.render_calls))
            print(json.dumps(dict(raw=raw, native_forwards=rt.render_calls)), flush=True)
            rt._data.clear()
            torch.cuda.empty_cache()
    write(OUTPUT / 'adapter_receipt.json', rt.receipt())
    write(OUTPUT / 'render_receipt.json', dict(status='RENDERED_AWAITING_CPU_METRICS_AND_ROOT_VISUAL_REVIEW',
        raws=list(RAWS), native_forwards=rt.render_calls, frames=frames,
        seconds=time.time()-started, optimizer_updates=0, geometry_updates=0,
        colour_fit_updates=0, calibration_solves=0, human_verdict=None, quality_accepted=False))

if __name__ == '__main__':
    try:
        main()
    except Exception:
        if OUTPUT.is_dir():
            write(OUTPUT / 'engineering_failure.json', dict(status='FAILED', traceback=traceback.format_exc()))
        raise

"""Paired fixed-ray metrics and pilot grids for early native checkpoints."""
from pathlib import Path
import json

import numpy as np
from PIL import Image
import torch

from evaluate_a6_probe import front_opacity

PILOTS = (105, 113, 130)
METRICS = ('a6_iso_rgb', 'a37_iso_rgb', 'a6_delete_rgb', 'a6_keep_rgb', 'a37_keep_rgb',
           'a6_delete_w', 'a6_wrong_w37', 'a37_wrong_w6', 'strict_a37_w',
           'a6_depth_m', 'a6_static_front', 'a6_reveal_static_front',
           'a6_reveal_w', 'road_keep_rgb', 'protected_keep_rgb', 'full_keep_rgb')


@torch.no_grad()
def readout(objective, scene, args, background, env, output, iteration, frames=None):
    from gaussian_renderer import prepare_rasterization
    from attribute_diag import load_raster
    raster, _ = load_raster()
    frames = frames or (tuple(range(89, 138)) if iteration == 10000 else PILOTS)
    folder = Path(output) / f'probe_{iteration:05d}'
    folder.mkdir()
    rows = []
    static, dynamic = scene.gaussians, scene.dynamic_gaussians
    cameras = {int(cam.image_name[-3:]): cam for cam in scene.getTrainCameras()}
    for raw in frames:
        camera = cameras[raw]
        evidence, _ = objective.evidence(camera)
        with np.load(Path(objective.observations) / f'{raw:03d}.npz', allow_pickle=False) as source:
            target = torch.as_tensor(source['rgb'], device='cuda').permute(2, 0, 1).float()
        if target.max() > 1:
            target /= 255
        full = objective.draw(camera, args, static, dynamic, background, env, [], [])
        deleted = objective.draw(camera, args, static, dynamic, background, env, [], [], hide=37)
        if (deleted['owner'][1].abs().max() != 0 or
                (full['owner'].sum(0) - full['alpha'][0]).max() > 1e-5):
            raise RuntimeError(f'raw{raw} joint attribution violates alpha/deletion contract')
        visible = evidence['visible']
        obs_depth = evidence['depth_m'][0]
        probe = objective.reference['depth_m'][raw-89]
        # Exact historical static transmittance, including separate road/PVD
        # marker domains for calibration. Only measured a6 layers enter metrics.
        front, calibration = front_opacity(camera, args, static, raster, prepare_rasterization,
                                           probe.max(0), objective.scale, .30, raw == 113, torch)
        front = torch.as_tensor(front, device='cuda')
        known, reveal = [torch.as_tensor(a > 0, device='cuda') for a in probe[:2]]
        rgb_err = lambda value: (value - target).abs().mean(0)
        iso = [rgb / alpha.clamp_min(.01) for rgb, _, alpha in full['isolated']]
        depth = deleted['depth'][0] / deleted['alpha'][0].clamp_min(1e-5) / objective.scale
        values = (
            (rgb_err(iso[0]), visible[0]), (rgb_err(iso[1]), visible[1]),
            (rgb_err(deleted['render']), visible[0]), (rgb_err(full['render']), visible[0]),
            (rgb_err(full['render']), visible[1]), (deleted['owner'][0], visible[0]),
            (full['owner'][1], visible[0]), (full['owner'][0], visible[1]),
            (full['owner'][1], evidence['occluded'][0]),
            ((depth - obs_depth).abs(), evidence['depth_valid'][0]),
            (front, known), (front, reveal), (deleted['owner'][0], reveal),
            (rgb_err(full['render']), evidence['road']),
            (rgb_err(full['render']), evidence['protect']),
            (rgb_err(full['render']), torch.ones_like(known)),
        )
        row = {'raw': raw, 'calibration': calibration, 'metrics': {}}
        for key, (value, mask) in zip(METRICS, values, strict=True):
            selected = value[mask]
            if not torch.isfinite(selected).all():
                raise RuntimeError(f'Nonfinite {key} raw{raw}')
            row['metrics'][key] = dict(count=selected.numel(), sum=float(selected.double().sum()))
        rows.append(row)
        if raw in PILOTS:
            owner = full['owner']
            attribution = torch.stack((owner[1], owner[0], (full['alpha'][0]-owner.sum(0)).clamp_min(0)))
            panels = (target, full['render'], deleted['render'],
                      full['isolated'][0][0], full['isolated'][1][0], attribution)
            img = torch.cat((torch.cat(panels[:3], 2), torch.cat(panels[3:], 2)), 1)
            arr = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + .5).astype(np.uint8)
            Image.fromarray(arr).save(folder / f'raw{raw}.png')
    pooled = {}
    for key in METRICS:
        count = sum(row['metrics'][key]['count'] for row in rows)
        total = sum(row['metrics'][key]['sum'] for row in rows)
        pooled[key] = dict(count=count, mean=total/count if count else None)
    result = dict(iteration=iteration, frames=frames, pooled=pooled, rows=rows)
    (folder / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('EARLY_PROBE', iteration, {key: value['mean'] for key, value in pooled.items()}, flush=True)
    return result

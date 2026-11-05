"""Check which native layer fills newly visible domains after a37 is hidden."""
import json

import numpy as np
from PIL import Image
import torch

import compare_control

PILOTS = (105, 113, 130)


@torch.no_grad()
def inspect(objective, scene, args, background, env, output, iteration, frames=None):
    from gaussian_renderer import render

    result = []
    names = ('measured_visible', 'measured_reprojected', 'fixed_pvd_proxy', 'road_depth_proxy')
    m, d = scene.gaussians, scene.dynamic_gaussians
    for camera in scene.getTrainCameras():
        raw = int(camera.image_name[-3:])
        with np.load(objective.observations/f'{raw:03d}.npz', allow_pickle=False) as source:
            road = torch.as_tensor(source['road_edit'], device='cuda')
            road_rgb = torch.as_tensor(source['road_edit_rgb'], device='cuda').permute(2, 0, 1).float()
        if road_rgb.max() > 1:
            road_rgb /= 255
        full = objective.draw(camera, args, m, d, background, env, [], [], hide=37)
        tagged = objective.draw(camera, args, m, d, background, env, [], [])
        native = render(camera, m, d, args, background, env_map=env)
        parity = float((tagged['render']-native['render']).abs().max())
        if parity > 1e-6:
            raise RuntimeError(f'Native and tagged joint RGB differ at raw{raw}: {parity}')
        owner = full['owner'][0]
        alpha = full['alpha'][0]
        static = (alpha - owner).clamp_min(0)
        environment = 1 - alpha
        tracks = torch.stack((owner, static, environment))
        row = dict(raw=raw, native_rgb_max_abs=parity, domains={})
        depths = objective.reference['depth_m'][raw-89]
        for name, mask in zip(names, depths > 0):
            mask = torch.as_tensor(mask, device='cuda')
            vals = tracks[:, mask]
            row['domains'][name] = dict(count=vals.shape[1], sums=vals.double().sum(1).tolist())
            if name == 'road_depth_proxy':
                verified = mask & road
                errors = (full['render'] - road_rgb).abs().mean(0)[verified]
                row['road_reobserved_rgb'] = dict(count=errors.numel(), sum=float(errors.double().sum()))
        result.append(row)
        if raw in PILOTS:
            # Red = static, green = a6, blue = environment.
            attribution = torch.stack((static, owner, environment))
            panels = torch.cat((full['render'], attribution), dim=2)
            array = (panels.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + .5).astype(np.uint8)
            Image.fromarray(array).save(output / f'edit_raw{raw}.png')

    pooled = {}
    for name in names:
        total = sum(row['domains'][name]['count'] for row in result)
        sums = np.asarray([row['domains'][name]['sums'] for row in result]).sum(0)
        pooled[name] = dict(count=total, mean=(sums/total).tolist() if total else None)
    road_total = sum(row['road_reobserved_rgb']['count'] for row in result)
    road_sum = sum(row['road_reobserved_rgb']['sum'] for row in result)
    data = dict(iteration=iteration, component_order=['a6', 'static', 'environment'], pooled=pooled, rows=result,
                road_reobserved_rgb=dict(count=road_total, mean=road_sum/road_total if road_total else None))
    (output / 'delete_domains.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    print(pooled, data['road_reobserved_rgb'])


if __name__ == '__main__':
    compare_control.readout = inspect
    compare_control.main()

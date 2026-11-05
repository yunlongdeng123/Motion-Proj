"""Separate static photometric errors from dynamic leakage on fixed pilot road pixels."""
import json

import numpy as np
import torch

import compare_control


@torch.no_grad()
def inspect(objective, scene, args, background, env, output, iteration, frames=None):
    from gaussian_renderer import render
    result = []
    for c in scene.getTrainCameras():
        raw = int(c.image_name[-3:])
        data, _ = objective.evidence(c)
        with np.load(objective.observations/f'{raw:03d}.npz', allow_pickle=False) as source:
            gt = torch.as_tensor(source['rgb'], device='cuda').permute(2, 0, 1).float()
        if gt.max() > 1:
            gt /= 255
        m, d = scene.gaussians, scene.dynamic_gaussians
        full = objective.draw(c, args, m, d, background, env, [], [])
        deleted = objective.draw(c, args, m, d, background, env, [], [], hide=37)
        native = render(c, m, d, args, background, env_map=env)
        names = {'combined':full['render'], 'static_only':full['render_s'], 'delete':deleted['render'],
                 'native':native['render']}
        row = {'raw': raw, 'tagged_native_maxabs':(full['render']-native['render']).abs().max().item()}
        for key in ('road','protect'):
            mask = data[key]
            error = {k: (v-gt).abs()[:,mask].mean().item() for k,v in names.items()}
            weights = full['owner'][:,mask].mean(1).tolist()
            row[key] = {'count':int(mask.sum()), 'rgb':error, 'a6_a37':weights,
                        'static_alpha':full['static_alpha'][0,mask].mean().item()}
        result.append(row)
    (output / f'road_only_{iteration}.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for key in ('road','protect'):
        n = sum(row[key]['count'] for row in result)
        terms = {term:sum(row[key]['rgb'][term]*row[key]['count'] for row in result)/n
                 for term in result[0][key]['rgb']}
        print(iteration, key, terms)
    print('tagged/native maxabs', max(r['tagged_native_maxabs'] for r in result))


if __name__ == '__main__':
    compare_control.readout = inspect
    compare_control.main()

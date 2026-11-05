"""PVD-only black-background RGB and its premultiplied contribution in the full scene."""
import argparse
import json

import numpy as np
from PIL import Image
import torch

from render_pvd import (ROOT, ROAD, OBS, REF, PVD, Experiment, generated_layer, draw, raster, save)

PILOTS = (105, 113, 130)


def pool(values):
    alone = sum(item['isolated_alpha_sum'] for item in values)
    full = sum(item['joint_weight_sum'] for item in values)
    return dict(pixels=sum(item['pixels'] for item in values), isolated_alpha_sum=alone,
                joint_weight_sum=full, retained_fraction=full/alone if alone else None)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hide-native-a6', action='store_true')
    diagnostic = parser.parse_args().hide_native_a6
    torch.set_num_threads(1)
    exp = Experiment()
    road_path = ROAD/'formal/checkpoint001000.pt'
    saved = torch.load(road_path, map_location='cpu', weights_only=False)
    assert saved['step'] == 1000 and saved['parent_identity'] == exp.n.before
    exp.repair.load_state_dict(saved['repair'], strict=True)
    exp.repair.requires_grad_(False)
    patch = generated_layer()
    assert len(patch.anchor) == 1477
    dest = ROOT/('black_diagnostic' if diagnostic else 'black_column')
    dest.mkdir(exist_ok=False)
    with np.load(REF, allow_pickle=False) as source:
        fixed = source['depth_m'].copy()
    previous = ROOT/'render'
    rows = []
    replay_max = 0
    alpha_overshoot = 0.
    states = ('no_native_a6',) if diagnostic else ('keep','delete')
    for raw in range(89, 138):
        camera = exp.camera(raw)
        with np.load(OBS/f'{raw:03d}.npz', allow_pickle=False) as data:
            assert abs(float(data['model_time']) - camera.timestamp) < 1e-6
            assert np.allclose(data['model_to_camera'].T,
                               camera.world_view_transform.cpu().numpy(), atol=1e-6)
            pose = torch.as_tensor(data['model_from_local'][0], dtype=torch.float32, device='cuda')
            known = torch.as_tensor(data['visible'][0], device='cuda')
        black = exp.n.bg.new_zeros(3)
        pvd_rows = patch.raster_rows(pose, camera, exp.n.args, 1, precomputed=True)
        alone = raster(camera, exp.n.args, 0, pvd_rows, black)
        isolated_alpha = alone['alpha'][0]
        if not diagnostic:
            save(dest/f'{raw:03d}_isolated.png', alone['rgb'])
        if raw in PILOTS and not diagnostic:
            save(dest/f'{raw:03d}_isolated_alpha.png', alone['alpha'].expand(3,-1,-1))
        domains = dict(full=torch.ones_like(known), observed_a6=known,
                       fixed_hidden_proxy=torch.as_tensor(fixed[raw-89,2]>0, device='cuda'))
        record = dict(raw=raw)
        for state in states:
            full = draw(exp, camera, patch, pose, delete=state!='keep', patch_rgb=True,
                        hide_native_a6=diagnostic)
            weight = full['extra'][3]
            premultiplied = full['extra'][4:7]
            assert torch.isfinite(premultiplied).all() and (premultiplied>=0).all()
            assert float((premultiplied-weight).max()) < 2e-6
            alpha_overshoot = max(alpha_overshoot, float((weight-isolated_alpha).max()))
            if not diagnostic:
                with Image.open(previous/f'{raw:03d}_{state}.png') as image:
                    old = np.asarray(image).astype(np.int16)
                check = np.uint8(np.round(full['rgb'].clamp(0,1).permute(1,2,0).cpu().numpy()*255))
                replay_max = max(replay_max, int(np.abs(check.astype(np.int16)-old).max()))
                assert replay_max <= 1
            save(dest/f'{raw:03d}_{state}_visible.png', premultiplied)
            if raw in PILOTS:
                save(dest/f'{raw:03d}_{state}_weight.png', weight[None].expand(3,-1,-1))
            record[state] = {name:dict(pixels=int(mask.sum()),
                                       isolated_alpha_sum=float(isolated_alpha[mask].double().sum()),
                                       joint_weight_sum=float(weight[mask].double().sum()))
                             for name,mask in domains.items()}
        rows.append(record)
    assert exp.unchanged() and alpha_overshoot < 3e-6
    pooled = {state:{domain:pool([row[state][domain] for row in rows]) for domain in rows[0][state]}
              for state in states}
    result = dict(status='PASS', rows=rows, pooled=pooled, pvd_rows=len(patch.anchor), training_updates=0,
                  checkpoints=[str(road_path), str(PVD)], joint_rgb_replay_max_uint8=None if diagnostic else replay_max,
                  diagnostic_hides_native_a6=diagnostic,
                  pvd_alpha_overshoot=alpha_overshoot,
                  isolated_definition='Sum alpha*T*PVD_RGB with only PVD rows. Black RGB background.',
                  joint_definition='Sum alpha*T*PVD_RGB with T from the entire scene. Black RGB background.')
    (dest/'black_column.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(pooled,indent=2))


if __name__ == '__main__':
    main()

"""Bind R3 to the frozen owner-tagged native 10K checkpoint."""
from pathlib import Path
import sys

import numpy as np
import torch

import editor_common as common

ROOT = Path(__file__).resolve().parent.parent
PARENT = ROOT.parent / 'scene008_native_ownership_20261105'
OLD_NATIVE = ROOT.parent / 'scene008_a37_only_B10k_20261103_r1'
OLD_ROAD = ROOT.parent / 'scene008_a37_only_B10k_road_20261103_r3'
OBS = ROOT.parent / 'scene008_instances_20261105_r1/observations/frames'
REF = ROOT.parent / 'scene008_a6_a37_dynamic_20261104_r1/inputs/a6_probe_reference.npz'
sys.path.insert(0, str(PARENT / 'source'))
from dual_actor_runtime import install_identity_serialization, actor_ids

install_identity_serialization()
common.B = PARENT / 'run_r3/formal'


class Native(common.Native):
    def __init__(self):
        super().__init__()
        self.owners = actor_ids(self.dynamic)
        if set(self.owners.unique().tolist()) != {6, 37}:
            raise ValueError('Expected both persistent actor identities')
        self.rows = {str(i): int((self.owners == i).sum()) for i in (6, 37)}
        self.keep_a6 = self.owners == 6

    def actor_kwargs(self, camera, delete_a37):
        dy = common.prepare_rasterization(camera, self.dynamic, self.args, dynamic=True)
        dy['means2D'] = torch.zeros_like(dy['means3D'])
        if delete_a37:
            dy['mask'] = dy['mask'] & self.keep_a6
        return dy

    def render(self, camera, road=None, ids=None, gates=None, dynamic=False):
        # The historic base() call defaults to deletion. 'dynamic=True' keeps
        # both instances. Road and gates are identical across the edit states.
        kw = common.prepare_rasterization(camera, self.static, self.args)
        kw['means2D'] = torch.zeros_like(kw['means3D'])
        if road is not None:
            kw['opacities'] = kw['opacities'].index_copy(
                0, ids, kw['opacities'][ids] * gates[:, None])
        dy = self.actor_kwargs(camera, delete_a37=not dynamic)

        def tag(values, flags):
            values['features'] = torch.cat((values['features'], flags), 1)
            return values

        tag(kw, kw['features'].new_zeros((len(kw['features']), 3)))
        actor_tags = torch.stack((torch.zeros_like(self.keep_a6), self.keep_a6, ~self.keep_a6), 1)
        tag(dy, actor_tags.to(dy['features'].dtype))
        kw = common.merge_rows(kw, dy)
        if road is not None:
            channels = kw['shs'].shape[1] if kw['shs'] is not None else 1
            rr = common.road_kwargs(camera, road, self.args, channels,
                                    precomputed_color=kw['colors_precomp'] is not None)
            flags = rr['features'].new_zeros((len(rr['features']), 3))
            flags[:, 0] = 1
            kw = common.merge_rows(kw, tag(rr, flags))
        out = common.raster(camera, self.args, self.static.active_sh_degree, kw, self.bg)
        env = self.env(camera.get_world_directions(False).permute(1, 2, 0)).permute(2, 0, 1)
        out['rgb'] = out['rgb'] + (1 - out['alpha']) * env
        return out


@torch.no_grad()
def capture_parent():
    from gaussian_renderer import render
    from dual_actor_runtime import hide_actors
    target = ROOT / 'native_delete'
    target.mkdir(exist_ok=False)
    n = Native()
    largest = 0.
    for raw in range(89, 138):
        camera = n.camera(raw)
        image = n.render(camera)['rgb']
        with hide_actors(n.dynamic, [37]):
            direct = render(camera, n.static, n.dynamic, n.args, n.bg, env_map=n.env)['render']
        largest = max(largest, float((image - direct).abs().max()))
        d = target / f'{raw:03d}'
        d.mkdir()
        common.save(d/'delete_a37.png', image)
        (d/'gt.png').symlink_to(OLD_NATIVE / 'evaluation' / f'{raw:03d}' / 'gt.png')
    if largest > 2e-6:
        raise RuntimeError(f'Parent deletion differs from direct native raster: {largest}')
    import json
    (ROOT / 'parent_binding.json').write_text(json.dumps(
        dict(frames=49, max_native_rgb_difference=largest, actor_rows=n.rows, parent=n.before,
             temporal_envelopes_preserved=True), indent=2) + '\n')


if __name__ == '__main__':
    capture_parent()

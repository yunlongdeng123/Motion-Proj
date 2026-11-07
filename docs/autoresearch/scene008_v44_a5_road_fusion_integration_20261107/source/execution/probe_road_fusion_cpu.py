"""只验证真实道路包和特征接线；不是CUDA渲染或画质验收。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch

from scripts.worldsim_v82.drivingeditor_dynamic_edit.road_fusion_integration import (
    AA_MANIFEST, C_MANIFEST, RoadFusionV44Integration, load_road_payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    ma, aa, _ = load_road_payload(AA_MANIFEST)
    mc, c, _ = load_road_payload(C_MANIFEST)
    geometry = ['render_model_xyz', 'render_scales_model', 'render_rotation_wxyz',
                'render_opacity_logit', 'active', 'child_to_parent_ids',
                'parent_render_model_xyz', 'parent_render_scales_model', 'parent_render_rotation_wxyz']
    for key in geometry:
        assert np.array_equal(aa[key], c[key]), key

    def rows(n):
        return dict(means3D=torch.arange(n*3, dtype=torch.float32).reshape(n, 3),
            means2D=torch.zeros(n, 3), shs=None, colors_precomp=torch.full((n, 3), .3),
            opacities=torch.full((n, 1), .5), scales=torch.ones(n, 3), rotations=torch.ones(n, 4),
            cov3D_precomp=None, mask=torch.ones(n, dtype=torch.bool), features=torch.zeros(n, 24))

    original = rows(7)
    for i in range(7):
        original['features'][i, 2+i] = 1
    # 当前语义模式只有a5。18/26空组不能暗中变成道路。
    original['features'][5:, 7:9] = 0
    original['features'][5:, 5] = 1
    original['features'][4, 9:12] = torch.tensor([.1, .2, .3])
    extra = rows(3)
    extra['features'][:, 9] = 1
    extra['features'][:, 13:16] = extra['colors_precomp']
    merge = lambda left, right: {k: None if left[k] is None else torch.cat((left[k], right[k]), 0) for k in left}
    toy = RoadFusionV44Integration.__new__(RoadFusionV44Integration)
    toy.torch, toy.np = torch, np
    toy.manifest_path, toy.manifest = C_MANIFEST, mc
    toy.semantic = SimpleNamespace(scene=lambda raw, delete: dict(rows=original, meta={}, cam=None))
    toy.base = SimpleNamespace(merge=merge)
    toy.assert_unchanged = lambda full=False: True
    toy.extension_rows = lambda raw: extra
    for delete in (False, True):
        scene = toy.scene(105, delete=delete)
        joint = scene['rows']
        assert joint['features'].shape == (10, 24)
        for key in original:
            if key != 'features' and original[key] is not None:
                assert torch.equal(joint[key][:7], original[key]), key
        assert torch.equal(joint['features'][:7, :9], original['features'][:, :9])
        assert torch.equal(joint['features'][:7, 10:13], original['features'][:, 9:12])
        assert bool((joint['features'][:7, 9] == 0).all())
        assert bool((joint['features'][7:, 9] == 1).all())
        assert bool((joint['features'][:, 16:] == 0).all())
        assert bool((joint['features'][:, 2:10].sum(1) == 1).all())
        assert torch.equal(joint['features'][7:, 13:16], extra['colors_precomp'])
    assert not torch.cuda.is_initialized()
    module_path = Path(__import__(RoadFusionV44Integration.__module__, fromlist=['']).__file__)
    result = dict(status='PASS_CPU_ROAD_PAYLOAD_AND_FEATURE_ROUTING',
        adapter_sha256=hashlib.sha256(module_path.read_bytes()).hexdigest(),
        aa_manifest=str(AA_MANIFEST), c_manifest=str(C_MANIFEST),
        aa_asset_sha256=ma['new_road']['sha256'], c_asset_sha256=mc['new_road']['sha256'],
        road_rows=40392, aa_c_exact_geometry_fields=geometry,
        changed_child_colour_rows=int(np.any(aa['render_colours_srgb'] != c['render_colours_srgb'], axis=1).sum()),
        feature_count=24, eight_alpha_groups_onehot=True,
        semantic5_rgb_preserved_in_explicit_repacking=True, physical_prefix_exact=True,
        newroad_rgb_routed=True, cuda_initialized=False, optimizer_updates=0,
        GPU_render_or_visual_quality_test=False,
        original_road_1k_local_x_m=[8.17,19.75], added_road_parent_local_x_m=[19.8,31.8],
        direct_material_transfer_to_old_road_has_no_overlapping_centres=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

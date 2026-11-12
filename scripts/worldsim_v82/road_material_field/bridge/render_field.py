"""在同一米制道路几何上，用原生 CUDA 高斯光栅比较不同材质场。"""
import argparse
import importlib.util
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

if __package__:
    from .material_field import WorldRoadAppearanceField, check_coordinate_bridge, linear_to_srgb
else:
    from material_field import WorldRoadAppearanceField, check_coordinate_bridge, linear_to_srgb


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def camera_for_tile(extent, oblique, device):
    # 所有分支共用一次构造的相机。模型单位是米×0.1，近裁剪可追溯。
    scale = .1
    side = max(extent)
    eye = np.array([0., -side * .7 if oblique else 0., max(side * 1.25, 2.8)]) * scale
    forward = -eye / np.linalg.norm(eye)
    right = np.array([1., 0., 0.])
    down = np.cross(forward, right)
    view = np.eye(4)
    view[:3, :3] = np.stack((right, down, forward))
    view[:3, 3] = -view[:3, :3] @ eye
    near, far = .01, 100.
    fov = math.radians(54)
    projection = np.zeros((4, 4))
    projection[0, 0] = projection[1, 1] = 1 / math.tan(fov / 2)
    projection[2, 2] = far / (far - near)
    projection[2, 3] = -far * near / (far - near)
    projection[3, 2] = 1
    transpose_view = torch.tensor(view.T, dtype=torch.float32, device=device)
    transpose_projection = torch.tensor(projection.T, dtype=torch.float32, device=device)
    return SimpleNamespace(image_width=256, image_height=256, FoVx=fov, FoVy=fov,
                           world_view_transform=transpose_view,
                           full_proj_transform=transpose_view @ transpose_projection,
                           camera_center=torch.tensor(eye, dtype=torch.float32, device=device)), dict(
                               local_camera_origin_m=(eye / scale).tolist(), meter_to_model=scale,
                               physical_hard_near_m=.2 / scale, projective_near_m=near / scale,
                               world_to_camera_local_m=view.tolist(), FoV_radians=fov)


def geometry_rows(field):
    values = field.gaussian_rows()
    matrix = field.world_from_uv
    u = matrix[:3, 0] / torch.linalg.norm(matrix[:3, 0])
    v = matrix[:3, 1] / torch.linalg.norm(matrix[:3, 1])
    normal = torch.linalg.cross(u, v)
    normal = normal / torch.linalg.norm(normal)
    basis = torch.stack((u, v, normal), 1)
    centre = values['world_xyz'].mean(0)
    xyz = ((values['world_xyz'] - centre) @ basis * .1).to(torch.float32).contiguous()
    dimensions = [float(torch.linalg.norm(matrix[:3, 0])), float(torch.linalg.norm(matrix[:3, 1]))]
    scales = xyz.new_tensor([dimensions[0] * .65, dimensions[1] * .65, .002]).repeat(len(xyz), 1) * .1
    quaternion = xyz.new_zeros((len(xyz), 4)); quaternion[:, 0] = 1
    values.update(xyz=xyz, scales=scales, quaternion=quaternion,
                  local_basis=basis, world_origin=centre,
                  extent=[dimensions[0] * field.width, dimensions[1] * field.height])
    return values


def load_raster(path):
    spec = importlib.util.spec_from_file_location('inherited_road_raster', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.raster


def render(field, fixed, camera, raster):
    values = field.query(fixed['world_xyz'])
    colours = linear_to_srgb(values['rgb_linear']).contiguous()
    xyz = fixed['xyz']
    homogeneous = torch.cat((xyz, torch.ones_like(xyz[:, :1])), 1)
    depth = homogeneous @ camera.world_view_transform[:, 2:3]
    kwargs = dict(means3D=xyz, means2D=torch.zeros_like(xyz), shs=None,
                  colors_precomp=colours, features=torch.cat((depth, torch.ones_like(depth)), 1),
                  opacities=torch.full_like(depth, .99), scales=fixed['scales'],
                  rotations=fixed['quaternion'], cov3D_precomp=None, mask=fixed['active'])
    pipe = SimpleNamespace(neg_fov=False, debug=False)
    with torch.no_grad():
        return raster(camera, pipe, 0, kwargs, xyz.new_zeros(3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, required=True, help='JSON 4x4 world_from_uv or sample with that key')
    parser.add_argument('--support', type=Path, required=True)
    parser.add_argument('--world-xyz', type=Path, help='HxWx3 fixed measured surface coordinates')
    parser.add_argument('--hole-mask', type=Path, help='255 means generated/inferred, outside is observed context')
    parser.add_argument('--uv-corner', action='store_true', help='input transform maps [u+.5,v+.5,0,1]')
    parser.add_argument('--texture', type=Path, action='append', required=True, help='same-shaped native/adapted texture')
    parser.add_argument('--raster', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    device = 'cuda'
    torch.set_num_threads(2)
    matrix = json.loads(args.matrix.read_text())
    if isinstance(matrix, dict):
        matrix = matrix['world_from_uv']
    matrix = np.asarray(matrix, dtype=np.float64)
    if args.uv_corner:
        offset = np.eye(4); offset[:2, 3] = .5
        matrix = matrix @ offset
    bridge_test = check_coordinate_bridge(device)
    world = None if args.world_xyz is None else np.load(args.world_xyz)
    provenance = None
    if args.hole_mask is not None:
        support = np.asarray(Image.open(args.support).convert('L')) > 127
        holes = np.asarray(Image.open(args.hole_mask).convert('L')) > 127
        provenance = np.where(support, np.where(holes, 3, 1), 0).astype(np.uint8)
    fields = [WorldRoadAppearanceField.from_files(p, args.support, matrix, provenance, world).to(device) for p in args.texture]
    fixed = geometry_rows(fields[0])
    roundtrip = []
    for i, field in enumerate(fields):
        sampled = field.query(fixed['world_xyz'])['rgb_linear']
        expected = field.reference_linear[0].permute(1, 2, 0).reshape(-1, 3)
        delta = (sampled - expected).abs()
        roundtrip.append(dict(texture=str(args.texture[i]),
                              real_float32_geometry_max_linear_error=float(delta.max()),
                              real_float32_geometry_mean_linear_error=float(delta.mean())))
        np.savez_compressed(args.output / f'world_material_field_{i}.npz',
            reference_linear=field.reference_linear[0].permute(1, 2, 0).cpu().numpy(),
            support=field.support[0, 0].cpu().numpy(), provenance=field.provenance.cpu().numpy(),
            world_xyz=field.surface_world_xyz.cpu().numpy(), world_from_uv=field.world_from_uv.cpu().numpy())
    raster = load_raster(args.raster)
    receipts = []
    for angle in [False, True]:
        camera, metadata = camera_for_tile(fixed['extent'], angle, device)
        geometry_before = {k: fixed[k].clone() for k in ['xyz', 'scales', 'quaternion', 'active']}
        alpha_first = None
        for i, field in enumerate(fields):
            package = render(field, fixed, camera, raster)
            name = f'{i}_{"oblique" if angle else "overhead"}'
            rgb = package['rgb'].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            alpha = package['alpha'].detach().cpu().numpy()
            Image.fromarray(np.uint8(rgb * 255 + .5)).save(args.output / (name + '.png'))
            np.savez_compressed(args.output / (name + '.npz'), rgb=rgb, alpha=alpha,
                                depth_sum=package['depth_sum'].detach().cpu().numpy())
            if alpha_first is None:
                alpha_first = alpha
            assert np.array_equal(alpha_first, alpha), '材质分支不得改变alpha/遮挡'
            assert all(torch.equal(fixed[k], v) for k, v in geometry_before.items())
            assert np.isfinite(rgb).all() and (alpha > .95).any()
            receipts.append(dict(texture=str(args.texture[i]), file=name + '.png', camera=metadata,
                                 finite=True, same_alpha_as_first=True, geometry_unchanged=True,
                                 visible_pixel_count=int((alpha > .95).sum())))
    write(args.output / 'receipt.json', dict(status='PASS_REAL_CUDA_MATERIAL_TO_GAUSSIAN_BRIDGE',
          rows=len(fixed['xyz']), bridge_test=bridge_test, cameras=receipts,
          raster_source=str(args.raster), learning_updates=0,
          colour_pipeline='reference linear RGB sampled by world coordinate, sRGB attributes in inherited alpha raster',
          field_files=[f'world_material_field_{i}.npz' for i in range(len(fields))],
          actual_geometry_texture_roundtrip=roundtrip,
          provenance_legend={'0': 'unknown', '1': 'observed context', '2': 'transferred', '3': 'model inferred'},
          includes_vehicles=False, full_scene_edit_quality_evaluated=False,
          human_verdict=None))
    source = args.output / 'source'; source.mkdir()
    for name in ['material_field.py', 'render_field.py']:
        shutil.copy2(Path(__file__).with_name(name), source / name)


if __name__ == '__main__':
    main()

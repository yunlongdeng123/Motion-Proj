"""Existing TRELLIS material -> physical anti-alias grid -> 2cm H source. CPU only."""
from pathlib import Path
import os, sys, json, time
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
from trellis_hf_seed import sha, blur, bilinear, field_stats

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / 'source_snapshot'
sys.path.insert(0, str(S))
from exact_mesh_uv import WorldMeshUVMap


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def frame():
    transforms = json.loads((S / 'transforms.json').read_text())
    M = np.asarray(transforms['world_to_canonical_yup'])
    origin = np.linalg.solve(M[:3, :3], -M[:3, 3])
    rotation = M[:3, :3] / np.linalg.norm(M[:3, :3], axis=1)[:, None]
    basis = np.stack([rotation[0], -rotation[2]], axis=1)
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12, rtol=0):
        raise ValueError('original physical frame is not orthonormal')
    return origin, basis, rotation[1]


def audit():
    with np.load(S / 'mesh_uv_source.npz') as z:
        d = {k: z[k] for k in z.files}
    with np.load(S / 'fixed_geometry.npz') as z:
        original = {k: z[k] for k in z.files}
    prior = json.loads((S / 'previous_native_preparation.json').read_text())
    pins = prior['source_sha256']
    source_checks = {}
    for name in ['mesh_uv_source.npz', 'basecolor_rgba.png']:
        expected = [value for path, value in pins.items() if path.endswith('/world_prior_r1/' + name)]
        source_checks[name] = len(expected) == 1 and sha(S / name) == expected[0]
    mapped = d['generated_vertex_to_fixed'][d['generated_faces'][d['fixed_to_generated_face']]]
    source_checks['fixed_face_vertex_identity'] = np.array_equal(np.sort(mapped, axis=1), np.sort(d['fixed_faces'], axis=1))
    source_checks['fixed_world_vertices_exact'] = np.array_equal(original['world_vertices'], d['world_vertices'])
    source_checks['fixed_faces_exact'] = np.array_equal(original['faces'], d['fixed_faces'])
    color = json.loads((S / 'color_semantics_receipt.json').read_text())
    midpoint = json.loads((S / 'previous_midpoint_receipt.json').read_text())
    source_checks['previous_official_UV_exporter_checks'] = all(color['checks'].values())
    source_checks['previous_fixed_midpoint_qualification'] = midpoint['status'] == 'PASS_MIDPOINT_CPU_QUALIFICATION'
    if not all(source_checks.values()):
        raise AssertionError(source_checks)
    rgba = np.asarray(Image.open(S / 'basecolor_rgba.png').convert('RGBA'), np.float64) / 255
    origin, basis, normal = frame()
    uv = (d['world_vertices'][:, :2] - origin[:2]) @ np.linalg.inv(basis[:2]).T
    plane_height = (d['world_vertices'] - origin) @ normal
    maximum = float(np.linalg.norm(d['generated_vertices_world'] - d['world_vertices'][d['generated_vertex_to_fixed']], axis=1).max())
    result = dict(status='NO_NEW_DEMONSTRATED_UV_COLOR_SCALE_ALPHA_BUG', checks=source_checks,
        source_sha256={name: sha(S / name) for name in ['mesh_uv_source.npz', 'basecolor_rgba.png', 'fixed_geometry.npz', 'transforms.json']},
        geometry_identity_max_m=maximum, raw_PNG_RGB_mean=rgba[..., :3].mean((0, 1)).tolist(),
        raw_PNG_alpha_min_mean_max=[float(rgba[..., 3].min()), float(rgba[..., 3].mean()), float(rgba[..., 3].max())],
        physical_mesh_UV_extent_m=[uv.min(0).tolist(), uv.max(0).tolist()],
        fixed_mean_plane_height_deviation_m=[float(plane_height.min()), float(plane_height.max())],
        limitations=['baseColor direct native sRGB prior is not identified intrinsic albedo/radiance',
                     'gray unknown reference conditioning can affect generated style; no causal attribution claimed',
                     'original GH hard-boundary numerical failure was real and already repaired; fixed midpoint check is sample-scoped',
                     '2.56m is nominal patch footprint; actual mesh center-to-center span is recorded; mean-plane metric is not geodesic'],
        generated_alpha_changes_opacity=False, new_inference_calls=0, optimizer_updates=0, GPU_operations=0)
    save_json(ROOT / 'receipts/bug_review.json', result)
    return result


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('explicit CPU-only environment required')
    output = ROOT / 'extraction_r1'
    if output.exists():
        raise FileExistsError('preserve old extraction')
    ROOT.joinpath('receipts').mkdir(exist_ok=True)
    bug = audit()
    start = time.monotonic()
    mapper = WorldMeshUVMap.load(S)
    origin, basis, normal = frame()
    side, step, aa_sigma = 2.56, .0025, .02
    N = int(round(side / step))
    axis = (np.arange(N) + .5 - N / 2) * step
    rgb = np.zeros((N, N, 3), np.float32)
    valid = np.zeros((N, N), bool)
    reference_unknown = np.zeros((N, N), np.float32)
    with np.load(S / 'mesh_uv_source.npz') as z:
        source_ref = z['reference_valid']; faces = z['fixed_faces']
    inverse_max = 0.
    for begin in range(0, N, 64):
        end = min(begin + 64, N)
        xx, yy = np.meshgrid(axis, axis[begin:end])
        uv = np.stack([xx, yy], -1)
        world = origin[:2] + uv @ basis[:2].T
        q = mapper.query(world)
        good = q['valid']
        rgb[begin:end][good] = q['basecolor_rgba'][good, :3]
        valid[begin:end] = good
        reference_unknown[begin:end][good] = 1 - np.sum(q['barycentric'][good] * source_ref[faces[q['fixed_face_id'][good]]], axis=1)
        inverse_max = max(inverse_max, float(q['worldXY_reconstruction_error'][good].max()))
    mass = ndi.gaussian_filter(valid.astype(np.float64), aa_sigma / step, mode='constant', cval=0, truncate=4.)
    numerator = ndi.gaussian_filter(rgb.astype(np.float64) * valid[..., None], (aa_sigma / step, aa_sigma / step, 0), mode='constant', cval=0, truncate=4.)
    aa = numerator / np.maximum(mass[..., None], 1e-12)
    with np.load(S / 'fixed_geometry.npz') as z:
        xyz, keys, rows = z['world_vertices'], z['world_keys'], z['source_world_rows']
    plane_uv = (xyz[:, :2] - origin[:2]) @ np.linalg.inv(basis[:2]).T
    grid_uv = (plane_uv + side / 2) / step - .5
    samples = bilinear(aa, grid_uv)
    sampled_mass = bilinear(mass[..., None], grid_uv)[..., 0]
    minimum = keys.min(0)
    ij = keys - minimum
    H, W = ij.max(0)[::-1] + 1
    banded = np.zeros((H, W, 3), np.float32)
    world = np.zeros((H, W, 3), np.float64)
    world_rows = np.full((H, W), -1, np.int64)
    source_mass = np.zeros((H, W), np.float64)
    banded[ij[:, 1], ij[:, 0]] = samples
    world[ij[:, 1], ij[:, 0]] = xyz
    world_rows[ij[:, 1], ij[:, 0]] = rows
    source_mass[ij[:, 1], ij[:, 0]] = sampled_mass
    if np.any(world_rows < 0):
        raise ValueError('fixed H patch is not a complete chart block')
    output.mkdir()
    np.savez_compressed(output / 'nodes_2cm.npz', bandlimited_rgb=banded, source_world_xyz=world,
        source_world_rows=world_rows, source_valid_mass=source_mass, source_geometry_valid=source_mass > 1e-12,
        minimum_world_keys=minimum, physical_origin=origin, physical_basis=basis)
    np.savez_compressed(output / 'physical_grid.npz', rgb_srgb=rgb, geometry_valid=valid,
        antialiased_rgb_srgb=aa.astype(np.float32), aa_source_valid_mass=mass.astype(np.float32),
        colocated_reference_unknown=reference_unknown, origin=origin, basis=basis, step_m=step, side_m=side)
    high = banded.astype(np.float64) - blur(banded, 3.)
    recipe = dict(schema='T2_HF_only_CPU_source_candidate.v1', status='EXTRACTED_NOT_PERMISSION_QUALIFIED',
        nominal_patch_side_m=side, physical_step_m=step, H_node_spacing_nominal_m=.02,
        anti_alias_gaussian_sigma_m=aa_sigma, anti_alias_truncate=4.,
        anti_alias_amplitude_at_H_Nyquist25cycles_per_m=float(np.exp(-2 * np.pi**2 * aa_sigma**2 * 25**2)),
        highpass_sigma_m=.06, highpass_sigma_cells=3., highpass_truncate=3.,
        frequency_policy='soft Gaussian anti-alias then 2cm sampling then (I-G6cm); not a hard frequency cutoff',
        desired_prior_amplitude_candidate=.02, common_model_H_bound_candidate=.04,
        mean_plane_coordinate_metric_not_geodesic=True, actual_face_UV_query_inverse_max_m=inverse_max,
        fine_mesh_valid_pixels=int(valid.sum()), fine_total_pixels=int(valid.size),
        source_mass_at_H_nodes_quantiles=np.quantile(source_mass, [0, .01, .5, .99, 1]).tolist(),
        unqualified_raw_HF=field_stats(high, np.ones((H, W), bool)),
        source_geometry_identity_max_m=bug['geometry_identity_max_m'],
        permission='PENDING independent Q teacher ledger; no old valid/source/known hard gate',
        use='H initialization only at common A200/B entry; no extra prior loss; A0/low/A200/Adam-H state unchanged',
        source_hashes=bug['source_sha256'], output_hashes={'nodes_2cm.npz': sha(output / 'nodes_2cm.npz'), 'physical_grid.npz': sha(output / 'physical_grid.npz')},
        CPU_seconds=time.monotonic() - start, new_inference_calls=0, optimizer_updates=0, GPU_operations=0,
        central_contract_locked=False, GPU_entry_permitted=False)
    save_json(output / 'recipe.json', recipe)
    print(json.dumps(recipe), flush=True)


if __name__ == '__main__':
    main()

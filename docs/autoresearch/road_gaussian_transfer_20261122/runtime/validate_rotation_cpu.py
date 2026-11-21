"""显式世界四元数 ABI 的 CPU 验收；不创建 native renderer 或 CUDA context。"""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import sys
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
sys.path.insert(0, str(REPO / 'scripts/worldsim_v82'))


def module(name, path):
    spec = importlib.util.spec_from_file_location('drivingeditor_dynamic_edit.' + name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def matrices(q):
    return Rotation.from_quat(np.asarray(q)[:, [1, 2, 3, 0]]).as_matrix()


def covariances(basis, sigma):
    return (basis * np.asarray(sigma)[:, None, :] ** 2) @ basis.transpose(0, 2, 1)


def equal_provider(first, second):
    import torch
    checks = {key: bool(torch.equal(getattr(first, key), getattr(second, key)))
              for key in ('xyz', 'scaling', 'rotation', 'rgb', 'opacity', 'active')}
    assert all(checks.values()), checks
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, default=Path(__file__).with_name('v44_factory.py'))
    parser.add_argument('--before', type=Path, default=Path(__file__).with_name('v44_factory.before.py'))
    parser.add_argument('--asset', type=Path, default=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_surface_residual_20261121_r2/appearance_fit/asset/road.npz'))
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('rotation_validation.json'))
    args = parser.parse_args()
    before = module('_rotation_before', args.before)
    candidate = module('_rotation_candidate', args.candidate)
    import torch
    torch.set_num_threads(2)
    actual_stat = args.asset.stat()
    results = {}
    with tempfile.TemporaryDirectory(prefix='rotation_cpu_', dir=args.out.parent) as temporary:
        folder = Path(temporary)
        actual = load(args.asset)
        old = before.RoadRows(args.asset, 'cpu')
        current = candidate.RoadRows(args.asset, 'cpu')
        results['actual_asset_default_byte_exact'] = equal_provider(old, current)
        world_q = candidate._legacy_road_world_quaternions(actual['normal_world'], actual['model_from_world'], actual['meter_to_model'])
        restored = matrices(world_q)
        normal = actual['normal_world'] / np.linalg.norm(actual['normal_world'], axis=1, keepdims=True)
        normal_error = float(np.abs(restored[:, :, 2] - normal).max())
        assert normal_error < 1e-7
        actual['gaussian_rotation_world_wxyz'] = world_q
        actual['gaussian_rotation_world_override'] = np.zeros(len(actual['active']), bool)
        path = folder / 'actual_zero_override.npz'
        np.savez_compressed(path, **actual)
        zero = candidate.RoadRows(path, 'cpu')
        results['actual_asset_zero_override_byte_exact'] = equal_provider(old, zero)
        results['legacy_world_helper_normal_max_error'] = normal_error
        del old, current, zero, actual, restored, normal, world_q

        scale = 0.018849925866127503
        unit = Rotation.from_euler('zyx', [.4, -.2, .7]).as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = scale * unit
        transform[:3, 3] = [3., -2., .4]
        world_basis = Rotation.from_rotvec(np.array([[0., 0., 0.], [0., 0., np.pi / 2],
            [.4, -.5, .9], [np.pi, 0., 0.], [1e-8, np.pi - 1e-8, 0.]])).as_matrix()
        n = len(world_basis)
        sigma = np.tile([.023, .007, .003], (n, 1))
        active = np.array([True, True, True, True, False])
        a = dict(active=active, model_xyz=np.zeros((n, 3)), gaussian_sigma_m=sigma,
                 normal_world=world_basis[:, :, 2], reference_srgb=np.full((n, 3), .3, np.float32),
                 road_opacity=np.full(n, .98, np.float32), model_from_world=transform,
                 meter_to_model=np.array(scale), gaussian_rotation_world_wxyz=Rotation.from_matrix(world_basis).as_quat()[:, [3, 0, 1, 2]])
        path = folder / 'explicit_all.npz'
        np.savez_compressed(path, **a)
        value = candidate.RoadRows(path, 'cpu')
        got_basis = matrices(value.rotation.numpy())
        wanted_basis = np.einsum('ij,njk->nik', unit, world_basis[active])
        roundtrip = float(np.abs(got_basis - wanted_basis).max())
        assert roundtrip < 5e-7
        got = covariances(got_basis, value.scaling.numpy())
        world_cov = covariances(world_basis[active], sigma[active])
        expected = scale ** 2 * (unit[None] @ world_cov @ unit.T[None])
        covariance_relative = float(np.max(np.linalg.norm(got - expected, axis=(1, 2)) / np.linalg.norm(expected, axis=(1, 2))))
        assert covariance_relative < 2e-6 and np.linalg.eigvalsh(expected).min() > 0
        results['explicit_rotation_roundtrip_max_error'] = roundtrip
        results['explicit_covariance_relative_error'] = covariance_relative

        a['gaussian_rotation_world_override'] = np.array([True, True, False, True, True])
        np.savez_compressed(path, **a)
        mixed = candidate.RoadRows(path, 'cpu')
        legacy_a = {k: v for k, v in a.items() if not k.startswith('gaussian_rotation_world')}
        old_path = folder / 'legacy_small.npz'
        np.savez_compressed(old_path, **legacy_a)
        legacy = before.RoadRows(old_path, 'cpu')
        assert torch.equal(mixed.rotation[2], legacy.rotation[2])
        assert not torch.equal(mixed.rotation[1], legacy.rotation[1])
        results['mixed_override_keeps_old_row_exact_and_rotates_new_row'] = True

        source_basis = world_basis[2]
        target_basis = Rotation.from_euler('xyz', [.15, -.2, .6]).as_matrix()
        jacobian = target_basis @ np.array([[1.2, .18, 0], [0, .8, 0], [0, 0, .9]]) @ source_basis.T
        source_covariance = covariances(source_basis[None], sigma[2:3])[0]
        transferred = jacobian @ source_covariance @ jacobian.T
        tangent_cov = target_basis[:, :2].T @ transferred @ target_basis[:, :2]
        eigen, tangent = np.linalg.eigh(tangent_cov)
        basis = np.column_stack((target_basis[:, :2] @ tangent, target_basis[:, 2]))
        if np.linalg.det(basis) < 0:
            basis[:, 0] *= -1
        axis_variance = np.r_[eigen, target_basis[:, 2] @ transferred @ target_basis[:, 2]]
        transferred_a = dict(legacy_a)
        transferred_a.update(active=np.ones(1, bool), model_xyz=np.zeros((1, 3)),
            gaussian_sigma_m=np.sqrt(axis_variance)[None], normal_world=target_basis[:, 2][None],
            reference_srgb=np.full((1, 3), .3, np.float32), road_opacity=np.array([.98], np.float32),
            gaussian_rotation_world_wxyz=Rotation.from_matrix(basis).as_quat()[[3, 0, 1, 2]][None])
        transferred_path = folder / 'transported.npz'
        np.savez_compressed(transferred_path, **transferred_a)
        loaded = candidate.RoadRows(transferred_path, 'cpu')
        loaded_covariance = covariances(matrices(loaded.rotation.numpy()), loaded.scaling.numpy())[0]
        wanted = scale ** 2 * unit @ transferred @ unit.T
        affine_error = float(np.linalg.norm(loaded_covariance - wanted) / np.linalg.norm(wanted))
        assert affine_error < 2e-6 and np.linalg.eigvalsh(transferred).min() > 0
        results['J_Sigma_JT_relative_error'] = affine_error

        invalid = []
        def rejects(name, mutate):
            bad = {k: v.copy() if hasattr(v, 'copy') else v for k, v in a.items()}
            mutate(bad)
            bad_path = folder / (name + '.npz')
            np.savez_compressed(bad_path, **bad)
            try:
                candidate._load_road_arrays(bad_path)
            except ValueError:
                invalid.append(name)
            else:
                raise AssertionError('Expected rejection: ' + name)
        rejects('quaternion_shape', lambda b: b.update(gaussian_rotation_world_wxyz=np.ones((n, 3))))
        rejects('quaternion_nan', lambda b: b['gaussian_rotation_world_wxyz'].__setitem__((0, 0), np.nan))
        rejects('quaternion_nonunit', lambda b: b['gaussian_rotation_world_wxyz'].__setitem__(0, b['gaussian_rotation_world_wxyz'][0] * 1.1))
        rejects('quaternion_zero', lambda b: b['gaussian_rotation_world_wxyz'].__setitem__(0, 0))
        rejects('normal_axis_mismatch', lambda b: b['normal_world'].__setitem__(0, [1., 0., 0.]))
        rejects('nonpositive_sigma_not_SPD', lambda b: b['gaussian_sigma_m'].__setitem__((0, 0), 0))
        rejects('override_requires_bool', lambda b: b.update(gaussian_rotation_world_override=np.ones(n, np.int32)))
        rejects('override_shape', lambda b: b.update(gaussian_rotation_world_override=np.ones(n + 1, bool)))
        rejects('override_without_quaternion', lambda b: b.pop('gaussian_rotation_world_wxyz'))
        results['invalid_inputs_rejected'] = invalid

        empty = {k: v[:0] if getattr(v, 'shape', ()) and v.shape[0] == n else v for k, v in a.items()}
        empty_path = folder / 'empty.npz'
        np.savez_compressed(empty_path, **empty)
        empty_provider = candidate.RoadRows(empty_path, 'cpu')
        assert empty_provider.rotation.shape == (0, 4) and empty_provider.rows(None, None, None) is None
        results['empty_explicit_rotation_supported'] = True
    stat = args.asset.stat()
    assert stat.st_mtime_ns == actual_stat.st_mtime_ns and stat.st_size == actual_stat.st_size
    receipt = dict(schema='road_explicit_world_rotation_CPU_validation.v1', passed=True,
                   GPU_used=False, native_renderer_created=False, actual_source_asset_unchanged=True,
                   actual_asset=str(args.asset), checks=results)
    args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

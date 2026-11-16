"""失败模式检查及真实scene065浮点fixture；不把无目标域误报为质量通过。"""
from pathlib import Path
import numpy as np
import pytest

from .road_asset_quality import (audit_completion, audit_atlas_sampling, audit_training_asset,
                                 audit_local_texture_correction, audit_render, world_continuity)

BASE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene065_closedloop_20261116_r1/handoff_coverage_r2')


def test_write_permission_does_not_hide_illegal_pixel():
    observation = np.full((4, 4, 3), .4, np.float32)
    decode = np.full_like(observation, .5)
    write = np.zeros((4, 4), bool)
    write[1:3, 1:3] = True
    composite = observation.copy()
    composite[write] = decode[write]
    composite[0, 0] += .01
    record = audit_completion(observation, decode, composite, write, np.zeros_like(write))
    assert record['status'] == 'failed'
    assert record['metrics']['illegal_changed_pixels'] == 1


def test_conservation_cannot_replace_support_or_protection():
    domain = np.ones((4, 4), bool)
    road = np.full((4, 4), .1, np.float32)
    static = np.full_like(road, .9)
    zero = np.zeros_like(road)
    rgb = np.full((4, 4, 3), .4, np.float32)
    baseline = rgb.copy()
    baseline[0, 0] -= .02
    protected = np.zeros_like(domain)
    protected[0, 0] = True
    record, masks = audit_render(np.ones_like(road), road, static, zero, np.ones_like(road), domain,
                                 rgb=rgb, protected=protected, baseline_rgb=baseline)
    assert record['checks']['joint_branch_alphaT_sum']
    assert record['road_only_status'] == 'pass'
    assert record['protected_status'] == 'failed'
    assert record['status'] == 'failed'
    assert masks['static_dominated'].sum() == 16
    assert record['metrics']['physical_depth_order_verified'] is False


def test_world_neighbors_are_not_row_neighbors():
    keys = np.array([[100, 0], [0, 0], [1, 0], [99, 0]], np.int64)
    rgb = np.array([[.1] * 3, [.7] * 3, [.7] * 3, [.1] * 3], np.float32)
    metrics = world_continuity(rgb, keys, np.ones(4, bool))
    assert metrics['world_neighbor_edges'] == 2
    assert metrics['edge_RGB_jump_gt0p15'] == 0
    assert metrics['bright_neutral_endpoint_cells'] == 2


def test_actual_saved_atlas_off_grid_sampling_and_reload():
    path = Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_asset_20261116_r1/asset_r1/atlas.npz')
    if not path.exists():
        pytest.skip('scene065 remote atlas fixture not mounted')
    with np.load(path.parent / 'provenance.npz') as z:
        provenance = {key: z[key].copy() for key in ('world_xyz', 'row_rgb', 'unknown')}
    record = audit_atlas_sampling(path, provenance, max_samples=256)
    assert record['status'] == 'pass', record
    assert record['metrics']['midpoint_samples'] == 256
    assert record['visual_quality_status'] == 'incomplete'


def test_training_export_cannot_change_known_texture_or_geometry():
    before = dict(active=np.ones(4, bool), model_xyz=np.zeros((4, 3)),
                  gaussian_sigma_m=np.full((4, 3), .02), road_opacity=np.full(4, .99, np.float32),
                  reference_srgb=np.full((4, 3), .4, np.float32))
    trained = {key: value.copy() for key, value in before.items()}
    known = np.array([1, 0, 0, 0], bool)
    trained['reference_srgb'][0] += .01
    trained['model_xyz'][1, 2] += .01
    record = audit_training_asset(2, before, trained, np.ones(4, bool), observed=known)
    assert record['status'] == 'failed'
    assert record['checks']['observed_RGB_exact'] is False
    assert record['checks']['model_xyz_frozen_exact'] is False


def test_opacity_permission_does_not_allow_trusted_tangent_changes():
    before=dict(active=np.ones(4,bool),road_opacity=np.full(4,.99,np.float32),
                gaussian_sigma_m=np.full((4,3),.01),reference_srgb=np.full((4,3),.3,np.float32))
    trained={key:value.copy() for key,value in before.items()}
    trained['gaussian_sigma_m'][2,:2]*=1.1
    record=audit_training_asset(1,before,trained,np.ones(4,bool),max_tangent_scale_factor=1.15,
                                allowed_tangent_world_rows=np.array([1,1,0,0],bool))
    assert record['status']=='failed'
    assert record['checks']['gaussian_sigma_m_only_permitted_rows'] is False


def test_saved_linear_correction_reconstructs_and_keeps_known_guard_zero():
    grid=np.arange(4).reshape(2,2)
    base_rgb=np.full((2,2,3),.3,np.float32)
    known=np.array([1,0,0,0],bool);guard=np.array([0,1,0,0],bool)
    eligible=~known&~guard
    delta=np.zeros((2,2,3),np.float32);delta[1,0]=.04
    linear=((base_rgb.astype(np.float64)+.055)/1.055)**2.4+delta
    effective_rgb=(1.055*linear**(1/2.4)-.055).astype(np.float32)
    effective_rgb[~eligible[grid]]=base_rgb[~eligible[grid]]
    atlas=dict(rgb_srgb=base_rgb,grid_world_row=grid,world_from_uv=np.eye(4),surface_world_xyz=np.zeros((2,2,3)))
    effective={**atlas,'rgb_srgb':effective_rgb}
    correction=dict(delta_linear=delta,eligible=eligible[grid],grid_world_row=grid,
                    world_from_uv=np.eye(4),surface_world_xyz=np.zeros((2,2,3)))
    before=dict(active=np.ones(4,bool),road_opacity=np.full(4,.99,np.float32),reference_srgb=base_rgb.reshape(4,3).copy())
    trained={**before,'reference_srgb':effective_rgb.reshape(4,3).copy()}
    provenance=dict(texture_correction_eligible=eligible,known=known,guarded_world_rows=guard,
        requested=np.ones(4,bool),surface_supported=np.ones(4,bool),unknown=np.zeros(4,bool),material_updated=np.ones(4,bool),
        base_row_rgb=base_rgb.reshape(4,3).copy(),texture_delta_linear=delta.reshape(4,3).copy(),
        row_rgb=effective_rgb.reshape(4,3).copy(),effective_row_rgb=trained['reference_srgb'])
    record=audit_local_texture_correction(atlas,effective,correction,provenance,before,trained)
    assert record['status']=='pass',record
    correction['delta_linear'][0,0]=.01
    broken=audit_local_texture_correction(atlas,effective,correction,provenance,before,trained)
    assert broken['status']=='failed'
    assert broken['checks']['observed_delta_exact_zero'] is False


@pytest.mark.parametrize('raw', [86, 110, 134, 179, 198])
def test_actual_float_render_support_visibility_and_full_mask(raw):
    path = BASE / f'rendered/{raw:03d}/arrays.npz'
    if not path.exists():
        pytest.skip('scene065 remote float fixture not mounted')
    with np.load(path) as z:
        data = {key: z[key].copy() for key in z.files}
    with np.load(BASE / f'contribution_diagnostics_20261116/float/{raw:03d}.npz') as z:
        diagnostics = {key: z[key].copy() for key in z.files}
    record, masks = audit_render(data['road_only_alpha'], diagnostics['road_alphaT'], diagnostics['static_alphaT'],
                                 diagnostics['dynamic_alphaT'], data['alpha'], diagnostics['domain'])
    assert record['checks']['float_arrays_finite']
    assert record['checks']['joint_branch_alphaT_sum']
    assert record['checks']['joint_road_not_above_road_only']
    assert record['metrics']['domain_pixels'] == int(diagnostics['domain'].sum())
    assert np.array_equal(masks['gap'], diagnostics['domain'] & (data['road_only_alpha'].squeeze() < .05))
    assert record['visual_quality_status'] == 'incomplete'
    if raw == 179:
        assert record['road_only_status'] == 'failed'
        assert record['metrics']['gap_pixels'] > 0

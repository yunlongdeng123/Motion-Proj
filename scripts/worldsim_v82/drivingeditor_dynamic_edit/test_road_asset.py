"""材质接线和覆盖权限的解析控制。"""
import numpy as np
from .road_asset import (WorldRoadMaterialAtlas, build_observation_priority_atlas,
                         repair_road_coverage, interpolate_legal_surface, srgb_to_linear)
from .road_asset import infer_measured_plane_surface


def grid(size=5, cell=.02):
    yy, xx = np.mgrid[:size, :size]
    keys = np.c_[xx.ravel(), yy.ravel()]
    world = np.c_[2320 + keys[:, 0] * cell, 1365 + keys[:, 1] * cell, np.full(size * size, -20.)]
    return keys, world


def test_world_xy_inverse_preserves_measured_nodes_and_reload(tmp_path):
    keys, world = grid()
    world[:, 0] += .003 * np.sin(keys[:, 1])
    world[:, 1] += .002 * np.cos(keys[:, 0])
    rgb = np.random.default_rng(5).uniform(.1, .6, (len(world), 3)).astype(np.float32)
    atlas, _ = build_observation_priority_atlas(world, keys, rgb, np.ones(len(world), bool), np.ones(len(world), bool))
    assert np.array_equal(atlas.query(world)["rgb_srgb"], rgb)
    atlas.save(tmp_path / "atlas.npz")
    assert np.array_equal(WorldRoadMaterialAtlas.load(tmp_path / "atlas.npz").query(world)["rgb_srgb"], rgb)
    off = world[:1].copy(); off[:, 2] += .1
    assert atlas.query(off)["unknown"].all()


def test_bilinear_sampling_is_linear_light_and_unknown_is_conservative():
    keys, world = grid(2)
    rgb = np.array([[.1, .1, .1], [.9, .9, .9], [.1, .1, .1], [.9, .9, .9]], np.float32)
    atlas, _ = build_observation_priority_atlas(world, keys, rgb, np.ones(4, bool), np.ones(4, bool))
    midpoint = world.mean(axis=0, keepdims=True)
    sample = atlas.query(midpoint)
    assert np.allclose(sample["rgb_linear"][0], srgb_to_linear(rgb).mean(axis=0), atol=1e-6)
    assert np.all(sample["rgb_srgb"][0] > rgb.mean(axis=0))
    atlas.source[0, 0] = 0; atlas.confidence[0, 0] = 0
    assert atlas.query(midpoint)["unknown"].all()


def test_priority_never_overwrites_observed_or_illegal_cells():
    keys, world = grid(5, .5)
    known = np.zeros(25, bool); known[0] = True
    rgb = np.full((25, 3), .3, np.float32)
    legal = np.ones(25, bool); legal[-1] = False
    generated = np.full((25, 3), .8, np.float32)
    atlas, _ = build_observation_priority_atlas(world, keys, rgb, known, legal,
        generated_rgb=generated, generated_mask=np.ones(25, bool), generated_source_raw=np.full(25, 86))
    result = atlas.query(world)
    assert result["source"][0] == 1 and np.array_equal(result["rgb_srgb"][0], rgb[0])
    assert result["source"][1] == 2 and result["source_world_row"][1] == 0
    assert result["source"][2] == 3 and result["source_raw"][2] == 86
    assert result["source"][-1] == 0 and result["unknown"][-1]


def test_coverage_repairs_only_authorized_nontrusted_geometry():
    n = 9
    base = dict(active=np.array([True] * 8 + [False]), model_xyz=np.zeros((n, 3)),
        normal_world=np.tile([0., 0., 1.], (n, 1)), gaussian_sigma_m=np.full((n, 3), .005),
        road_opacity=np.full(n, .3), reference_srgb=np.full((n, 3), .5))
    requested = np.ones(n, bool); support = np.ones(n, bool); support[5] = False
    trusted = np.zeros(n, bool); trusted[0] = True
    protected = np.zeros(n, bool); protected[1] = True
    unknown = np.zeros(n, bool); unknown[3] = True
    result, provenance, _ = repair_road_coverage(base, dict(unknown=unknown), requested, support, trusted, protected)
    assert np.array_equal(result["model_xyz"], base["model_xyz"])
    assert np.array_equal(result["normal_world"], base["normal_world"])
    assert np.array_equal(result["gaussian_sigma_m"][[0, 1, 3, 5]], base["gaussian_sigma_m"][[0, 1, 3, 5]])
    assert result["active"][8] and provenance["coverage_added"][8]


def test_bounded_interpolation_accepts_plane_and_rejects_step():
    keys, world = grid(11)
    world[:, 2] += .01 * (world[:, 0] - 2320)
    normal = np.tile([-.01, 0., 1.], (len(world), 1))
    center = 60
    anchor = np.ones(len(world), bool); anchor[center] = False
    candidate = ~anchor
    broken = world.copy(); broken[center, 2] = -12
    fixed, _, proof = interpolate_legal_surface(broken, normal, anchor, candidate)
    assert proof["surface_interpolated"][center]
    assert abs(fixed[center, 2] - world[center, 2]) < 1e-10
    assert np.array_equal(fixed[anchor], world[anchor])
    stepped = broken.copy(); stepped[(keys[:, 0] > 5) & anchor, 2] += .12
    _, _, proof = interpolate_legal_surface(stepped, normal, anchor, candidate)
    assert not proof["surface_interpolated"].any()


def test_independent_plane_support_hull_rejects_unknown_road_extent():
    measurement = np.array([[2319., 1364.5, -20.], [2321., 1364.5, -20.],
                            [2319., 1365.5, -20.], [2321., 1365.5, -20.]])
    world = np.array([[2320., 1365., -12.], [2323., 1365., -12.], [2319., 1364.5, -20.]])
    normal = np.tile([0., 0., 1.], (3, 1))
    result, _, proof, _ = infer_measured_plane_surface(world, normal, np.array([True, True, False]),
        measurement, [0., 0., 1.], -20.)
    assert proof["surface_interpolated"].tolist() == [True, False, False]
    assert result[0, 2] == -20 and np.array_equal(result[2], world[2])

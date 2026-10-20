"""世界道路单元 / 原始多视角证据的固定合同。"""

ROAD_VERSION = 'V3_WORLD_ROAD_MULTIVIEW_20261020'
ROAD_POLICY = {
    'cell_m': .2, 'context_halo_m': 1.2, 'surface_support_radius_m': .8,
    'mask_margin_px': 2, 'depth_tolerance_m': .3, 'minimum_cosine': .12,
    'minimum_area_px': .25, 'minimum_quality': .2, 'minimum_views': 2,
    'source_batch_size': 2, 'robust_rgb_delta': .05,
    'reliability_floor': .1, 'reliability_color_scale': .15,
    'surface_weight': .1, 'normal_weight': .01, 'thin_weight': .01,
    'anchor_weight': .01, 'coverage_weight': .1,
    'normal_sigma_m': .015, 'maximum_normal_sigma_m': .03,
    'maximum_unit_drift_m': .15, 'audit_interval': 100,
    'normal_source': 'finite_difference_measured_LiDAR_height_field',
    'prior': 'same_world_four_neighbor_continuity_with_observed_context_boundary',
    'initial_color': [.18, .18, .18],
    'road_topology': 'one_fixed_thin_Gaussian_per_supported_world_unit_in_static_budget',
    'observation_color_space': 'unfused_source_sRGB_same_as_native_reconstruction',
}

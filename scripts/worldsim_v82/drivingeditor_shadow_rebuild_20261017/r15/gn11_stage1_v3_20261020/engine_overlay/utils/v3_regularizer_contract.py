"""V1 静态寿命/速度/NPS 合同；公式仍调用原生损失，不另写替代项。"""

TEMPORAL_VERSION = "V1_BACKGROUND_REGULARIZERS_RESTORED_20261020"
V1_REGULARIZER_CONFIG = {
    "lambda_t_reg": 0.1,
    "lambda_v_reg": 0.01,
    "paper_lambda_nps": 0.01,
    "hybrid_nps_enabled": True,
    "hybrid_time_reg_eps": 1e-8,
}
TEMPORAL_POLICY = {
    "config": V1_REGULARIZER_CONFIG,
    "duration_velocity_normalization": "joint_alpha_clamp_min_1e-5",
    "nps_normalization": "independent_static_alpha_clamp_min_1e-5",
    "nps_domain": "unchanged_five_ID_mask_four_axial_neighbors_both_parities_sum",
    "selective_temporal_objective_enabled": False,
}


def validate_v1_regularizers(config):
    for key, expected in V1_REGULARIZER_CONFIG.items():
        actual = config.get(key)
        if isinstance(expected, bool):
            matches = actual is expected
        else:
            matches = not isinstance(actual, bool) and actual == expected
        if not matches:
            raise RuntimeError(f"V1 regularizer mismatch: {key}={actual!r}, expected {expected!r}")

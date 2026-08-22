"""P4 Hunyuan3D-Omni smoke 的冻结合同测试。"""

from pathlib import Path

import yaml


def test_p4_is_one_off_offline_voxel_capability_smoke() -> None:
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (root / "configs/worldsim_v61/p4_hy3d_omni_smoke_v1.yaml").read_text(encoding="utf-8")
    )
    assert config["task_id"] == "WS-V61-P4-HY3D-OMNI-3090-SMOKE-01"
    assert config["hypothesis_id"] == "WS-V61-H-P4-001"
    assert config["inference"] == {
        "control_type": "voxel",
        "input_image": "demos/voxel/imgs/1c1ff58afbf4455ca80228d280f86aef.png",
        "input_voxel": "demos/voxel/plys/1c1ff58afbf4455ca80228d280f86aef.ply",
        "surface_points": 81920,
        "num_inference_steps": 50,
        "octree_resolution": 512,
        "mc_level": 0.0,
        "guidance_scale": 4.5,
        "use_ema": False,
        "fast_decode": False,
    }
    assert config["resources"]["network_during_formal_run"] == "disabled"
    assert config["environment"]["installation_contract"] == (
        "fixed_official_shape_inference_dependency_closure"
    )
    assert config["environment"]["excluded_non_inference_groups"] == [
        "training",
        "demo_ui",
        "texture_and_mesh_postprocess",
    ]
    assert config["environment"]["dino_cache_repo_dir"] == (
        "hub/models--facebook--dinov2-large"
    )
    assert config["environment"]["dino_cache_ref"] == "refs/main"
    assert len(config["sources"]["dino_revision"]) == 40
    assert config["license_boundary"]["no_model_or_output_distribution"]
    assert config["license_boundary"]["no_output_for_training_other_models"]
    assert all(
        len(value) == 64 and set(value) <= set("0123456789abcdef")
        for value in config["sources"]["model_files"].values()
    )

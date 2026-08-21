from pathlib import Path

import yaml

from motion_proj.worldsim_v6.r1_capability import REQUIRED_MATRIX_FIELDS, build_matrix


def _repo(commit: str, required: list[str]) -> dict:
    return {
        "exists": True,
        "commit": commit,
        "clean": True,
        "license_files": ["LICENSE"],
        "required_files": {item: True for item in required},
    }


def test_real_config_uses_only_logical_local_uris() -> None:
    root = Path(__file__).resolve().parents[2]
    config_path = root / "configs/worldsim_v6/r1_frontend_capability_v1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["schema_version"] == "worldsim_v6.r1_frontend_capability.v1"
    for candidate in config["candidates"]:
        for key in ("repo_uri", "env_uri"):
            if key in candidate:
                assert candidate[key].split("://", 1)[0] in {"third_party", "env"}
                assert not candidate[key].startswith("/")
                assert "\\" not in candidate[key]


def test_gate_accepts_executable_optimization_and_adaptable_feed_forward() -> None:
    optimization_commit = "a" * 40
    feed_commit = "b" * 40
    required = ["entry.py"]
    config = {
        "schema_version": "worldsim_v6.r1_frontend_capability.v1",
        "candidates": [
            {
                "id": "opt",
                "kind": "optimization",
                "paper": "opt",
                "official_source": "https://example.invalid/opt",
                "expected_commit": optimization_commit,
                "required_files": required,
                "license": {"declared": "MIT"},
                "weights": {"remote_available": False},
                "minimum_checkpoint_count": 1,
                "input_schema": "images",
                "output_schema": "gaussians",
                "gpu_requirement": {"minimum_vram_mib": 1000},
                "adapter_cost": "low",
                "selected_role": "reference",
            },
            {
                "id": "ff",
                "kind": "feed_forward",
                "paper": "ff",
                "official_source": "https://example.invalid/ff",
                "expected_commit": feed_commit,
                "required_files": required,
                "license": {"declared": "Apache-2.0"},
                "weights": {"remote_available": True},
                "input_schema": "images",
                "output_schema": "gaussians",
                "gpu_requirement": {"minimum_vram_mib": 1000},
                "adapter_cost": "medium",
                "adapter_possible": True,
                "selected_role": "candidate",
            },
        ],
    }
    local = {
        "gpu": {"available": True, "total_vram_mib": 24576},
        "frontends": {
            "opt": {"repo_key": "opt", "env_key": "opt", "checkpoint_count": 1, "input_ready": True},
            "ff": {"repo_key": "ff", "env_key": "ff", "checkpoint_key": "ff", "input_ready": False},
        },
        "third_party": {"opt": _repo(optimization_commit, required), "ff": _repo(feed_commit, required)},
        "envs": {"opt": {"exists": True}, "ff": {"exists": False}},
        "checkpoints": {"ff": {"exists": False}},
    }
    matrix = build_matrix(config, local)
    assert matrix["gate"] == {
        "optimization_executable": 1,
        "feed_forward_executable_or_adaptable": 1,
        "passed": True,
    }
    assert all(REQUIRED_MATRIX_FIELDS.issubset(row) for row in matrix["frontends"])
    assert matrix["frontends"][0]["local_status"] == "executable"
    assert matrix["frontends"][1]["local_status"] == "adaptable"

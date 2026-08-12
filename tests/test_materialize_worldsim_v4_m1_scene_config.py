from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.materialize_worldsim_v4_m1_scene_config import (
    materialize_scene_config,
    sha256_file,
)


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _config(tmp_path: Path, *, abstain: bool) -> Path:
    project = tmp_path
    run = project / "run"
    chain_path = run / "chain.json"
    matrix_path = project / "matrix.yaml"
    config_path = project / "m1.yaml"
    if abstain:
        chain = {
            "scene": "scene-a",
            "test_quality_read": False,
            "stages": {"instance_field": {"status": "abstain", "reason": "ABSTAIN_NO_ACTOR"}},
        }
    else:
        checkpoint = run / "checkpoint.pth"
        source_config = run / "source.yaml"
        field = run / "field.npz"
        sidecar = run / "semantic.npz"
        masks = run / "masks.json"
        for path in (checkpoint, source_config, field, sidecar, masks):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(path.name.encode())
        resolved = {
            "inputs": {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "source_config": str(source_config),
                "source_config_sha256": sha256_file(source_config),
            },
            "scene": {"processed_scene_dir": str(run / "processed")},
            "actors": {
                "high_support": {
                    "instance_token": "token",
                    "dataset_instance_id": 7,
                    "rigid_model_index": 0,
                    "semantic_sidecar": str(sidecar),
                    "semantic_sidecar_sha256": sha256_file(sidecar),
                }
            },
            "runtimes": {"drivestudio_python": "python", "drivestudio_checkout": "/code"},
        }
        _yaml(run / "instance" / "resolved.yaml", resolved)
        summary = {
            "evaluation_source": {
                "manifest": str(masks),
                "manifest_sha256": sha256_file(masks),
                "optimization_forbidden": True,
                "partition": "development",
            },
            "arms": {
                "O1_dual_opacity": {
                    "instance_field": str(field),
                    "instance_field_sha256": sha256_file(field),
                    "evaluation": {"aggregate": {"boundary_f1": 0.1}, "rows": []},
                }
            },
        }
        _json(run / "instance" / "instance_field" / "summary.json", summary)
        stage_summary = {
            "selected_arm": "O1_dual_opacity",
            "development_optimization_read": False,
            "development_content_read": True,
            "heldout_content_read": False,
            "test_quality_read": False,
        }
        _json(run / "instance" / "stage_summary.json", stage_summary)
        chain = {
            "scene": "scene-a",
            "test_quality_read": False,
            "stages": {
                "instance_field": {
                    "status": "done",
                    "run": str(run / "instance"),
                    "summary_sha256": sha256_file(run / "instance" / "stage_summary.json"),
                }
            },
        }
    _json(chain_path, chain)
    matrix = {
        "baselines": {
            "v33_frozen": {
                "executable_scene_chains": {
                    "scene-a": {
                        "algorithm_commit": "frozen",
                        "files": {
                            "scene_chain.json": {
                                "path": str(chain_path),
                                "sha256": sha256_file(chain_path),
                            }
                        },
                    }
                }
            }
        }
    }
    _yaml(matrix_path, matrix)
    config = {
        "schema_version": "worldsim_v4_m1_evidence_v1",
        "task_id": "WS-V4-M1-EVIDENCE-FIELD-01",
        "protocol": {
            "development_scenes": ["scene-a"],
            "abstain_no_actor_scenes": ["scene-a"] if abstain else [],
            "test_quality_read": False,
        },
        "inputs": {
            "baseline_matrix": "matrix.yaml",
            "required_v33_arm": "O1_dual_opacity",
        },
        "evidence": {},
        "calibration": {},
        "evaluation": {},
        "immutability": {},
        "gates": {},
    }
    _yaml(config_path, config)
    return config_path


def test_materialize_active_scene_binds_o1_and_partition(tmp_path: Path) -> None:
    config = _config(tmp_path, abstain=False)
    payload = materialize_scene_config(
        project_root=tmp_path, config_path=config, scene="scene-a"
    )
    assert payload["status"] == "ready"
    assert payload["inputs"]["v33_o1_instance_field"]["sha256"]
    assert payload["actors"]["high_support"]["dataset_instance_id"] == 7
    assert payload["test_quality_read"] is False


def test_materialize_preserves_registered_no_actor_abstention(tmp_path: Path) -> None:
    config = _config(tmp_path, abstain=True)
    payload = materialize_scene_config(
        project_root=tmp_path, config_path=config, scene="scene-a"
    )
    assert payload["status"] == "abstain"
    assert payload["reason"] == "ABSTAIN_NO_ACTOR"

import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_worldsim_v51_f0_source_preflight import (
    initialize_cuda_peak_tracking,
    observation_schema_report,
    repository_source_identity,
    summarize_instance_metadata,
)


def test_f0_source_preflight_help_works_from_repo_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_worldsim_v51_f0_source_preflight.py",
            "--help",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--run-dir" in result.stdout


def test_summarize_instance_metadata_reports_stable_train_only_tracks() -> None:
    instances = {
        "0": {"id": "actor-a", "class_name": "vehicle.car"},
        "1": {"id": "actor-b", "class_name": "vehicle.truck"},
    }
    frames = {"0": [0, 1], "40": [0], "80": [], "120": [1], "160": []}
    report = summarize_instance_metadata(instances, frames, [0, 40, 80, 120, 160])
    assert report["train_frame_active_union_count"] == 2
    assert report["train_frame_repeated_instance_count"] == 2
    assert report["stable_instance_tokens_present_unique"] is True


def test_observation_schema_does_not_treat_binary_sam_as_identity(tmp_path: Path) -> None:
    paths = []
    for index in range(2):
        path = tmp_path / f"view-{index}.npz"
        np.savez(
            path,
            gaussian_id=np.arange(2, dtype=np.int64),
            sam_probability=np.asarray([0.1, 0.9], dtype=np.float32),
            visibility=np.ones(2, dtype=np.float32),
        )
        paths.append(path)
    report = observation_schema_report(paths)
    assert report["associated_instance_identity_labels_present"] is False
    assert report["identity_label_fields"] == []


def test_observation_schema_detects_explicit_identity_field(tmp_path: Path) -> None:
    path = tmp_path / "view.npz"
    np.savez(path, gaussian_id=np.arange(2), instance_identity=np.asarray([1, 2]))
    report = observation_schema_report([path])
    assert report["associated_instance_identity_labels_present"] is True
    assert report["identity_label_fields"] == ["instance_identity"]


def test_repository_source_identity_binds_project_before_git_args(monkeypatch) -> None:
    calls = []

    def fake_git(project: Path, *args: str) -> str:
        calls.append((project, args))
        return "commit-value" if args[-1] == "HEAD" else "tree-value"

    monkeypatch.setattr(
        "scripts.run_worldsim_v51_f0_source_preflight._git", fake_git
    )
    project = Path("/tmp/frozen-project")
    assert repository_source_identity(project) == {
        "commit": "commit-value",
        "tree": "tree-value",
    }
    assert calls == [
        (project, ("rev-parse", "HEAD")),
        (project, ("rev-parse", "HEAD^{tree}")),
    ]


def test_cuda_peak_tracking_initializes_device_before_reset(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "scripts.run_worldsim_v51_f0_source_preflight.torch.cuda.set_device",
        lambda device: calls.append(("set_device", str(device))),
    )
    monkeypatch.setattr(
        "scripts.run_worldsim_v51_f0_source_preflight.torch.empty",
        lambda *args, **kwargs: calls.append(("allocate", str(kwargs["device"]))) or object(),
    )
    monkeypatch.setattr(
        "scripts.run_worldsim_v51_f0_source_preflight.torch.cuda.reset_peak_memory_stats",
        lambda device: calls.append(("reset", str(device))),
    )
    assert str(initialize_cuda_peak_tracking("cuda:0")) == "cuda:0"
    assert calls == [
        ("set_device", "cuda:0"),
        ("allocate", "cuda:0"),
        ("reset", "cuda:0"),
    ]

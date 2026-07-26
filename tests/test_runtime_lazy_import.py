"""运行时包的轻量导入回归测试。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_atomic_import_does_not_eagerly_load_torch() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import motion_proj.runtime.atomic; assert 'torch' not in sys.modules",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_runtime_public_symbols_remain_available() -> None:
    from motion_proj.runtime import (
        ExperimentRegistry,
        JsonlMetrics,
        ResumableRandomSampler,
        RunManifest,
        StageManifest,
    )

    assert [
        ExperimentRegistry.__name__,
        JsonlMetrics.__name__,
        ResumableRandomSampler.__name__,
        RunManifest.__name__,
        StageManifest.__name__,
    ] == [
        "ExperimentRegistry",
        "JsonlMetrics",
        "ResumableRandomSampler",
        "RunManifest",
        "StageManifest",
    ]

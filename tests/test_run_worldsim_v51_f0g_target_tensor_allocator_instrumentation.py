from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_worldsim_v51_f0g_target_tensor_allocator_instrumentation import (
    _outcome,
    _trace_command,
    _validate_config,
)
from scripts.worldsim_v51_cuda_trace_launcher import tensor_metadata


CONFIG = ROOT / "configs/worldsim_v51/stage_f_f0g_target_tensor_allocator_instrumentation_v1.yaml"


def test_f0g_help_works_from_repo_root() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_worldsim_v51_f0g_target_tensor_allocator_instrumentation.py", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--run-dir" in result.stdout


def test_trace_launcher_help_works_from_repo_root() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/worldsim_v51_cuda_trace_launcher.py", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--trace-output" in result.stdout


def test_f0g_config_and_wrapper_preserve_method_contract() -> None:
    config = _validate_config(CONFIG)
    attempt = config["execution"]["attempts"][0]
    command = _trace_command(config, attempt, Path("/input"), Path("/output"), Path("/trace.json"))
    assert "worldsim_v51_cuda_trace_launcher.py" in command[1]
    assert command[command.index("--SAM_NUM_POINTS_PER_SIDE") + 1] == "32"
    assert command[command.index("--SAM_NUM_POINTS_PER_BATCH") + 1] == "64"
    assert config["execution"]["source_neutral_contract"]["tensor_content_read"] is False
    assert config["locks"]["quality_read"] is False


def test_trace_tensor_metadata_reads_schema_not_content() -> None:
    tensor = torch.zeros((1, 3, 4), dtype=torch.float16)
    record = tensor_metadata(tensor)
    assert record == {
        "shape": [1, 3, 4],
        "dtype": "torch.float16",
        "device": "cpu",
        "stride": [12, 4, 1],
        "contiguous": True,
        "storage_offset": 0,
        "numel": 12,
        "element_size": 2,
        "logical_bytes": 24,
        "requires_grad": False,
    }


def test_f0g_outcome_matrix_is_diagnostic_only() -> None:
    success = {"classification": "success"}
    failure = {"classification": "expected_cublas_internal_failure"}
    assert _outcome([success, failure])[0] == "control_success_target_failure"
    assert _outcome([success, success])[0] == "both_success"
    assert _outcome([failure, success])[0] == "control_failure"

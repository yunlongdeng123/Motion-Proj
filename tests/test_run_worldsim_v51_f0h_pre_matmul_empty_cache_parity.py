from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_worldsim_v51_f0g_target_tensor_allocator_instrumentation import _trace_command
from scripts.run_worldsim_v51_f0h_pre_matmul_empty_cache_parity import (
    _outcome,
    _validate_config,
)
from scripts.audit_worldsim_v51_f0h_pre_matmul_empty_cache_parity import audit


CONFIG = ROOT / "configs/worldsim_v51/stage_f_f0h_pre_matmul_empty_cache_parity_v1.yaml"


def test_f0h_config_locks_empty_cache_abab_recovery() -> None:
    config = _validate_config(CONFIG)
    assert [row["name"] for row in config["execution"]["attempts"]] == [
        "control_cache_1",
        "target_cache_1",
        "control_cache_2",
        "target_cache_2",
    ]
    assert config["execution"]["pre_matmul_empty_cache"] is True
    assert config["locks"]["smaller_batch_retry"] is False
    assert config["decision"]["full_materialization_authorized"] is False


def test_f0h_wrapper_enables_empty_cache_without_method_change() -> None:
    config = _validate_config(CONFIG)
    attempt = config["execution"]["attempts"][0]
    command = _trace_command(config, attempt, Path("/input"), Path("/output"), Path("/trace.json"))
    assert "--pre-matmul-empty-cache" in command
    assert command[command.index("--SAM_NUM_POINTS_PER_SIDE") + 1] == "32"
    assert command[command.index("--SAM_NUM_POINTS_PER_BATCH") + 1] == "64"


def _success(group: str, config: dict) -> dict:
    reference = config["input_groups"][group]["reference"]
    trace = {
        "pre_matmul_empty_cache": True,
        "events": [
            {"event": "pre_matmul", "empty_cache": {"before": {"free_bytes": 1}, "after": {"free_bytes": 2}}},
            {"event": "post_matmul"},
            {"event": "pre_matmul", "empty_cache": {"before": {"free_bytes": 1}, "after": {"free_bytes": 2}}},
            {"event": "post_matmul"},
        ],
    }
    return {
        "classification": "success",
        "input_group": group,
        "masks": [{"sha256": value} for value in reference["mask_sha256"]],
        "metadata": {"sha256": reference["metadata_sha256"]},
        "trace": {"payload": trace},
    }


def test_f0h_outcome_requires_reference_exact_and_empty_cache_evidence() -> None:
    config = _validate_config(CONFIG)
    attempts = [_success("control", config), _success("target", config)] * 2
    assert _outcome(config, attempts)[0] == "recovery_pass"
    drifted = [dict(row) for row in attempts]
    drifted[1] = dict(drifted[1])
    drifted[1]["masks"] = [dict(row) for row in drifted[1]["masks"]]
    drifted[1]["masks"][0]["sha256"] = "drift"
    assert _outcome(config, drifted)[0] == "recovery_nonexact"


def test_f0h_r039_independent_audit_replays_recovery_parity() -> None:
    result = audit(
        CONFIG,
        Path(
            "/root/autodl-tmp/runs/worldsim_v51/"
            "WS-V51-M1-F-IDENTITY-EMBEDDING-01/"
            "20260818T160000Z__m1-stage-f-f0h-empty-cache-parity-s20260814-r039"
        ),
    )
    assert result["status"] == "pass"
    assert result["outcome"] == "recovery_pass"
    assert result["empty_cache_call_count"] == 8
    assert result["parity_checks"]["reference_checks"] == [True] * 4
    assert result["quality_read"] is False
    assert result["full_materialization"] is False

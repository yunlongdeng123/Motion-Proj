import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_dr_v2_m4.py"
SPEC = importlib.util.spec_from_file_location("run_dr_v2_m4", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def valid_report() -> dict:
    return {
        "status": "done",
        "task_id": "DR-V2-M4-EDIT-PILOT-01",
        "scene": "scene-0230",
        "instance_token": "af663976db5e412e83db033d309c5c29",
        "frames": list(range(196)),
        "cameras": [0, 1, 2],
        "checks": {"coverage": True, "hash": True},
        "mask_truth_claim": False,
        "quality_claim": False,
    }


def test_m4_report_validator_accepts_complete_protocol() -> None:
    MODULE.validate_report(valid_report())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("instance_token", "changed", "actor"),
        ("frames", list(range(195)), "coverage"),
        ("mask_truth_claim", True, "真实观测"),
        ("quality_claim", True, "质量结论"),
    ],
)
def test_m4_report_validator_fails_closed(field: str, value, message: str) -> None:
    report = valid_report()
    report[field] = value
    with pytest.raises(RuntimeError, match=message):
        MODULE.validate_report(report)

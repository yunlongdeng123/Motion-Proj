import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_dr_v2_m3_post_training.py"
SPEC = importlib.util.spec_from_file_location("run_dr_v2_m3_post_training", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_require_input_stages_enforces_frozen_actor(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stages"
    stage_dir.mkdir()
    for name in MODULE.REQUIRED_INPUT_STAGES:
        payload = {"stage": name, "status": "done"}
        if name == "native_actor_mapping_probe":
            payload.update(
                {
                    "status": "available",
                    "instance_token": MODULE.SELECTED_TOKEN,
                    "checkpoint_gaussian_count": 10,
                }
            )
        (stage_dir / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    assert MODULE.require_input_stages(tmp_path)["native_actor_mapping_probe"][
        "instance_token"
    ] == MODULE.SELECTED_TOKEN
    probe_path = stage_dir / "native_actor_mapping_probe.json"
    probe = json.loads(probe_path.read_text())
    probe["instance_token"] = "wrong"
    probe_path.write_text(json.dumps(probe), encoding="utf-8")
    with pytest.raises(RuntimeError, match="frozen actor"):
        MODULE.require_input_stages(tmp_path)


def test_validate_smoke_report_is_fail_closed() -> None:
    images = [
        {"variant": variant}
        for variant in ("original", "lateral_plus_1m", "remove")
        for _ in range(9)
    ]
    report = {"status": "done", "checks": {"all": True}, "images": images}
    MODULE.validate_smoke_report(report)
    report["checks"]["all"] = False
    with pytest.raises(RuntimeError, match="checks failed"):
        MODULE.validate_smoke_report(report)


def test_atomic_json_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "stage.json"
    MODULE.atomic_json(output, {"status": "done"})
    with pytest.raises(FileExistsError, match="refuse to overwrite"):
        MODULE.atomic_json(output, {"status": "done"})

import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

import resim.event_first_n1_cutin as cutin_runner
from resim.event_first_n1_cutin import _validate_config_contract


CONFIG_PATH = Path("configs/resim/event_first_n1_cutin_v1.yaml")


def _config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_formal_k4_config_contract_is_complete():
    config = _config()
    _validate_config_contract(config)
    assert config["kinematics_control"]["min_median_speed_mps"] == 0.5
    assert config["kinematics_control"]["max_acceleration_mps2"] == 12.0


@pytest.mark.parametrize(
    "missing_key",
    ["min_median_speed_mps", "max_acceleration_mps2"],
)
def test_k4_config_contract_rejects_missing_lane_keep_key(missing_key):
    config = deepcopy(_config())
    del config["kinematics_control"][missing_key]
    with pytest.raises(ValueError, match=f"kinematics_control.{missing_key}"):
        _validate_config_contract(config)


def test_k4_config_contract_rejects_n2_authorization():
    config = deepcopy(_config())
    config["stop_rule"]["never_start_n2_from_this_run"] = False
    with pytest.raises(ValueError, match="N2 fail-closed"):
        _validate_config_contract(config)


def test_uncaught_post_creation_error_marks_run_failed(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "RUNNING").write_text("running\n", encoding="utf-8")
    monkeypatch.setattr(cutin_runner, "_ACTIVE_RUN_DIR", run_dir)
    try:
        raise KeyError("delayed_runtime_key")
    except KeyError as exc:
        cutin_runner._mark_active_run_failed(exc)
    failure = json.loads((run_dir / "failure.json").read_text(encoding="utf-8"))
    assert failure["terminal_status"] == "FAILED"
    assert failure["exception_type"] == "KeyError"
    assert failure["n2_authorized"] is False
    assert (run_dir / "FAILED").is_file()
    assert not (run_dir / "RUNNING").exists()

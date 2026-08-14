from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_worldsim_v5_m1_formal_batch.py"
SPEC = importlib.util.spec_from_file_location("worldsim_v5_m1_formal_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_formal_batch_config_freezes_exact_eight_scene_denominator() -> None:
    config = AUDIT.load_config(
        ROOT / "configs/worldsim_v5/m1_formal_batch_audit_v1.yaml"
    )
    rows = config["runs"]
    assert len(rows) == 8
    assert len({row["scene"] for row in rows}) == 8
    assert [row["run_id"].rsplit("-", 1)[-1] for row in rows] == [
        f"r{index:03d}" for index in range(27, 35)
    ]
    assert config["formal_source_commit"] == "267faba485148a6e60e79864848c8376016b6369"
    assert (
        config["formal_config_sha256"]
        == "b13453d25621e1ba7d5fbe01c69bf0560c506822a2337fdac62d3de05fc1abf3"
    )

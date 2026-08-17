from pathlib import Path

import yaml

from motion_proj.worldsim_v51.protocol import sha256_file


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "configs/worldsim_v51/stage_e_e0b_same_propagation_freeze_v1.yaml"


def test_e0b_freeze_binds_canonical_operator_audit_and_locks() -> None:
    freeze = yaml.safe_load(FREEZE.read_text(encoding="utf-8"))
    assert freeze["status"] == "done"
    assert freeze["canonical_run"]["status"] == "done"
    assert freeze["canonical_run"]["manifest_inventory_entries"] == 13
    for name in ("config", "module", "runner"):
        spec = freeze["source"][name]
        assert sha256_file(ROOT / spec["path"]) == spec["sha256"]
    run = Path(freeze["canonical_run"]["path"])
    for name, expected in freeze["canonical_run"]["hashes"].items():
        relative = {
            "summary": "summary.json",
            "status": "status.json",
            "manifest": "manifest.json",
            "events": "events.jsonl",
            "resources": "artifacts/resources.json",
            "resource_samples": "artifacts/resource_samples.jsonl",
        }[name]
        assert sha256_file(run / relative) == expected
    audit = freeze["independent_audit"]
    assert sha256_file(Path(audit["report"])) == audit["report_sha256"]
    assert audit["full_operator_replay_exact"] is True
    assert freeze["governance"]["quality_read"] is False
    assert freeze["governance"]["e1_panogs_execution"] is False
    assert freeze["governance"]["m2_status"] == "pending"
    assert freeze["governance"]["m3_status"] == "pending"

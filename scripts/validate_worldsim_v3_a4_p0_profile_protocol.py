#!/usr/bin/env python
"""校验 A4-P0 端到端 profile 冻结协议。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from omegaconf import OmegaConf


PROJECT = Path("/root/autodl-tmp/motion_proj")
DEFAULT_PROTOCOL = PROJECT / "configs/worldsim_v3/a4_p0_profile_protocol_v1.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"A4-P0 protocol invalid: {message}")


def validate_schema(protocol: Mapping[str, Any]) -> None:
    require(protocol["schema_version"] == 1, "schema_version")
    require(protocol["task_id"] == "WS-V3-A4-DEPLOYMENT-01", "task_id")
    require(
        protocol["protocol_status"] == "frozen_before_new_a4_measurements",
        "protocol_status",
    )
    require(protocol["seed"] == 0 and protocol["scene"] == "scene-0230", "split")
    authorization = protocol["authorization"]
    require(not any(bool(value) for value in authorization.values()), "authorization")
    selected = protocol["selected_asset"]
    require(selected["variant"] == "r0-no-refine-exact-alias", "selected variant")
    require(selected["role"] == "A3-star-R0-off-D2-immutable-exact-alias", "role")
    require(
        selected["rejected_r1_checkpoint_policy"]
        == "forbidden_as_profile_or_production_input",
        "R1 exclusion",
    )
    probe = protocol["new_probe"]
    require(probe["resolution"] == {
        "width": 1600,
        "height": 900,
        "policy": "native_source_config_no_resize",
    }, "resolution")
    require(probe["warmup"] == {
        "frame": 0,
        "camera": 0,
        "repeats": 2,
        "output_hash_must_repeat_exactly": True,
    }, "warmup")
    measured = probe["measured"]
    require(measured["frames"] == [10, 100, 190], "measured frames")
    require(measured["cameras"] == [0, 1, 2], "measured cameras")
    require(measured["expected_samples"] == 9, "measured sample count")
    require(measured["edit"] == "original_only", "edit scope")
    require(probe["load_semantics"]["os_cache_eviction"] == "forbidden", "cache eviction")
    ceilings = protocol["resource_ceilings"]
    require(ceilings == {
        "wall_time_seconds": 600,
        "peak_torch_allocated_mib": 16384,
        "peak_torch_reserved_mib": 24576,
        "peak_nvidia_process_memory_mib_sampled": 24000,
        "peak_cgroup_memory_bytes": 34359738368,
        "run_bytes": 50000000,
        "disk_free_floor_bytes": 30000000000,
        "oom_events_delta": 0,
        "oom_kill_events_delta": 0,
    }, "resource ceilings")
    reuse = protocol["historical_evidence"]["reuse_policy"]
    require(reuse["train"] == "immutable_stage_reuse_no_rerun", "train reuse")
    require(reuse["render_eval"] == "immutable_stage_reuse_no_rerun", "eval reuse")
    require(
        reuse["convert"] == "inventory_manifest_only_no_parameter_conversion",
        "convert scope",
    )
    recovery = protocol["recovery_contract"]
    require(
        recovery["stage_order"]
        == ["inventory", "runtime_probe", "aggregate", "resume_audit"],
        "stage order",
    )
    require(recovery["completed_stage_policy"] == "never_overwrite", "overwrite policy")
    precedents = set(protocol["failure_precedents"])
    require({"PIVOT-F05", "PIVOT-F22", "PIVOT-F24"} <= precedents, "failure precedents")
    require(all(bool(value) for value in protocol["required_audits"].values()), "audits")
    require(all(bool(value) for value in protocol["claim_boundary"].values()), "claims")


def iter_fingerprinted_inputs(protocol: Mapping[str, Any]):
    selected = protocol["selected_asset"]
    for name in ("checkpoint", "source_config", "actor_registry"):
        yield f"selected_asset.{name}", selected[name]
    evidence = protocol["historical_evidence"]
    for name in ("manifest", "summary", "resource_log"):
        yield f"historical_evidence.{name}", evidence[name]
    for name, value in evidence["stages"].items():
        yield f"historical_evidence.stages.{name}", value


def validate_inputs(protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    audits = {}
    for name, spec in iter_fingerprinted_inputs(protocol):
        path = Path(spec["path"])
        require(path.is_file(), f"missing {name}: {path}")
        actual_sha = sha256_file(path)
        require(actual_sha == spec["sha256"], f"hash drift {name}")
        actual_bytes = path.stat().st_size
        if "bytes" in spec:
            require(actual_bytes == int(spec["bytes"]), f"byte drift {name}")
        audits[name] = {
            "path": str(path),
            "sha256": actual_sha,
            "bytes": actual_bytes,
        }
    closeout = protocol["selected_asset"]["a3_closeout_commit"]
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", closeout, "HEAD"],
        cwd=PROJECT,
        check=False,
    ).returncode == 0
    require(ancestor, "A3 closeout commit is not an ancestor of HEAD")
    return audits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--schema-only", action="store_true")
    args = parser.parse_args()
    protocol = OmegaConf.to_container(OmegaConf.load(args.protocol), resolve=True)
    validate_schema(protocol)
    inputs = {} if args.schema_only else validate_inputs(protocol)
    print(json.dumps({
        "status": "done",
        "protocol": str(args.protocol),
        "protocol_sha256": sha256_file(args.protocol),
        "schema_only": args.schema_only,
        "input_audits": inputs,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

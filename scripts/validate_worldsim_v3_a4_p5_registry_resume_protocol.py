#!/usr/bin/env python
"""校验 A4-P5 registry/resume 结果前冻结协议。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from omegaconf import OmegaConf


PROJECT = Path("/root/autodl-tmp/motion_proj")
DEFAULT_PROTOCOL = PROJECT / "configs/worldsim_v3/a4_p5_registry_resume_protocol_v1.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"A4-P5 protocol invalid: {message}")


def validate_schema(protocol: Mapping[str, Any]) -> None:
    require(protocol["schema_version"] == 1, "schema_version")
    require(protocol["task_id"] == "WS-V3-A4-DEPLOYMENT-01", "task_id")
    require(protocol["profile_id"] == "A4-P5-REGISTRY-RESUME-v1", "profile_id")
    require(
        protocol["protocol_status"] == "frozen_before_new_p5_measurements",
        "protocol_status",
    )
    require(protocol["seed"] == 0 and protocol["scene"] == "scene-0230", "split")
    authorization = protocol["authorization"]
    require(authorization["p5_registry_resume_execution_authorized"], "P5 authorization")
    require(
        not any(
            bool(value)
            for name, value in authorization.items()
            if name != "p5_registry_resume_execution_authorized"
        ),
        "forbidden authorization",
    )
    selected = protocol["selected_asset"]
    require(selected["role"] == "A3-star-R0-off-D2-immutable-exact-alias", "role")
    p0 = protocol["p0_canonical_evidence"]
    require(p0["source_commit"] == "b191afaaa88d5d356506fc29a36e6128959d8897", "P0 source commit")
    registry = protocol["registry_contract"]
    require(registry["schema_version"] == "worldsim-v3-deployment-registry-v1", "registry schema")
    require(registry["mode"] == "reference_only_immutable_manifest", "registry mode")
    require(registry["checkpoint_copy_forbidden"], "checkpoint copy")
    require(registry["checkpoint_rewrite_forbidden"], "checkpoint rewrite")
    require(registry["output_bytes_ceiling"] == 2_000_000, "registry bytes")
    static = registry["static_asset"]
    require(
        static
        == {
            "asset_id": "static-background-000",
            "model_key": "models.Background",
            "gaussian_count": 1_205_164,
            "asset_count": 1,
            "storage": "monolithic_checkpoint_reference",
            "independently_extractable": False,
            "independent_extraction_missing_reason": "p3_chunk_not_authorized",
        },
        "static asset",
    )
    actors = registry["actor_assets"]
    require(
        {
            "model_key": actors["model_key"],
            "gaussian_count": actors["gaussian_count"],
            "actor_count": actors["actor_count"],
            "available_actor_count": actors["available_actor_count"],
            "unavailable_actor_count": actors["unavailable_actor_count"],
            "unavailable_policy": actors["unavailable_policy"],
        }
        == {
            "model_key": "models.RigidNodes",
            "gaussian_count": 104_704,
            "actor_count": 24,
            "available_actor_count": 23,
            "unavailable_actor_count": 1,
            "unavailable_policy": "preserve_unavailable_empty_checkpoint_slice",
        },
        "actor assets",
    )
    reload_smoke = protocol["reload_smoke"]
    require(reload_smoke["checkpoint_loads"] == 1, "checkpoint loads")
    require(reload_smoke["render_count"] == 0, "render count")
    require(reload_smoke["no_optimizer_construction_or_step"], "optimizer")
    recovery = protocol["recovery_contract"]
    require(
        recovery["stage_order"]
        == ["input_audit", "registry_materialize", "reload_smoke", "aggregate", "resume_audit"],
        "stage order",
    )
    require(recovery["completed_stage_policy"] == "never_overwrite", "overwrite policy")
    require(recovery["resume_probe_process"] == "no_torch_no_gpu", "resume process")
    require(
        protocol["resource_ceilings"]
        == {
            "wall_time_seconds": 180,
            "peak_torch_allocated_mib": 16_384,
            "peak_torch_reserved_mib": 24_576,
            "peak_nvidia_process_memory_mib_sampled": 24_000,
            "peak_cgroup_memory_bytes": 34_359_738_368,
            "run_bytes": 5_000_000,
            "disk_free_floor_bytes": 30_000_000_000,
            "oom_events_delta": 0,
            "oom_kill_events_delta": 0,
        },
        "resource ceilings",
    )
    require(len(protocol["required_audits"]) == 14, "audit count")
    require(all(bool(value) for value in protocol["required_audits"].values()), "audits")
    require(all(bool(value) for value in protocol["claim_boundary"].values()), "claims")
    require(
        {"PIVOT-F05", "PIVOT-F22", "PIVOT-F25"}
        <= set(protocol["failure_precedents"]),
        "failure precedents",
    )


def iter_fingerprinted_inputs(protocol: Mapping[str, Any]):
    selected = protocol["selected_asset"]
    for name in ("checkpoint", "source_config", "actor_registry"):
        yield f"selected_asset.{name}", selected[name]
    p0 = protocol["p0_canonical_evidence"]
    for name in ("protocol", "manifest", "summary", "resource_audit", "runtime_rows", "terminal"):
        yield f"p0_canonical_evidence.{name}", p0[name]


def validate_inputs(protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    audits = {}
    for name, spec in iter_fingerprinted_inputs(protocol):
        path = Path(spec["path"])
        require(path.is_file(), f"missing {name}: {path}")
        actual_sha = sha256_file(path)
        require(actual_sha == spec["sha256"], f"hash drift {name}")
        actual_bytes = path.stat().st_size
        require(actual_bytes == int(spec["bytes"]), f"byte drift {name}")
        audits[name] = {
            "path": str(path),
            "sha256": actual_sha,
            "bytes": actual_bytes,
        }
    terminal = json.loads(
        Path(protocol["p0_canonical_evidence"]["terminal"]["path"]).read_text(encoding="utf-8")
    )
    summary = json.loads(
        Path(protocol["p0_canonical_evidence"]["summary"]["path"]).read_text(encoding="utf-8")
    )
    require(terminal == {"failure": None, "status": "done"}, "P0 terminal")
    require(summary["status"] == "done" and all(summary["audits"].values()), "P0 summary")
    closeout = protocol["p0_closeout_commit"]
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", closeout, "HEAD"],
        cwd=PROJECT,
        check=False,
    ).returncode == 0
    require(ancestor, "P0 closeout commit is not an ancestor of HEAD")
    return audits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--schema-only", action="store_true")
    args = parser.parse_args()
    protocol = OmegaConf.to_container(OmegaConf.load(args.protocol), resolve=True)
    validate_schema(protocol)
    inputs = {} if args.schema_only else validate_inputs(protocol)
    print(
        json.dumps(
            {
                "status": "done",
                "protocol": str(args.protocol),
                "protocol_sha256": sha256_file(args.protocol),
                "schema_only": args.schema_only,
                "input_audits": inputs,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

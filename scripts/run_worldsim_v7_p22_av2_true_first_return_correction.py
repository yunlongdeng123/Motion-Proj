"""Correct consumed AV2 visibility attribution to literal first return."""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from motion_proj.worldsim_v7.av2_four_action_compiler import compile_log
from motion_proj.worldsim_v7.true_first_return_attribution import (
    attribute_actor_true_first_return,
)
from motion_proj.worldsim_v7.visible_failure_attribution import summarize_attributions


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _identity(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["log_id"]), str(row["track_id"])


def _scope(rows: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    if name == "all":
        return rows
    hazardous = name == "hazard"
    return [row for row in rows if bool(row["hazardous"]) == hazardous]


def _comparison(
    legacy_rows: list[dict[str, Any]],
    literal_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    result = {}
    for name in ("all", "hazard", "clear"):
        legacy = summarize_attributions(_scope(legacy_rows, name))
        literal = summarize_attributions(_scope(literal_rows, name))
        old_rate = float(legacy["new_early_fraction_of_target_rays"])
        new_rate = float(literal["new_early_fraction_of_target_rays"])
        result[name] = {
            "actor_count": int(literal["actor_count"]),
            "target_ray_count": int(literal["target_ray_count"]),
            "proxy_new_early_count": int(legacy["new_early_count"]),
            "literal_new_early_count": int(literal["new_early_count"]),
            "proxy_new_early_rate": old_rate,
            "literal_new_early_rate": new_rate,
            "literal_to_proxy_rate_ratio": (
                float(new_rate / old_rate) if old_rate > 0.0 else None
            ),
            "proxy_new_hit_count": int(legacy["new_hit_count"]),
            "literal_new_hit_count": int(literal["new_hit_count"]),
            "literal_new_early_by_provenance": literal[
                "new_early_by_provenance"
            ],
            "literal_new_hit_by_provenance": literal["new_hit_by_provenance"],
        }
    return result


def run(config_path: Path, repo_root: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    compiler = yaml.safe_load(
        (repo_root / str(config["p2_config"])).read_text(encoding="utf-8")
    )
    compiler["compiler_geometry"].update(config.get("compiler_overrides", {}))
    cohort = json.loads(
        (repo_root / str(config["cohort_config"])).read_text(encoding="utf-8")
    )
    roles = set(config["cohort_roles"])
    selected = [row for row in cohort["logs"] if row["role"] in roles]
    if len(selected) != int(config["expected_log_count"]):
        raise RuntimeError("frozen P22 cohort count changed")

    run_dir = Path(config["runs_root"]) / "worldsim_v7" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        run_dir / "status.json",
        {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P22 literal first-return correction requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    literal_rows: list[dict[str, Any]] = []

    try:
        for position, cohort_row in enumerate(selected):
            log_id = str(cohort_row["log_id"])
            result = compile_log(
                Path(compiler["dataset_root"]) / log_id,
                compiler,
                device,
                include_diagnostics=True,
            )
            diagnostics = result["compiled"]["diagnostics"]
            for actor_row in result["actor_rows"]:
                track_id = str(actor_row["track_id"])
                row = attribute_actor_true_first_return(
                    actor_row,
                    diagnostics[track_id],
                    config["attribution"],
                    float(compiler["compiler_geometry"]["output_voxel_size_m"]),
                    device,
                )
                row["log_id"] = log_id
                literal_rows.append(row)
            print(
                json.dumps(
                    {"progress": f"{position + 1}/{len(selected)}", "log_id": log_id},
                    ensure_ascii=False,
                ),
                flush=True,
            )

        legacy_rows = _read_jsonl(
            Path(config["legacy_attribution_run"])
            / "ACTOR_PROVENANCE_ATTRIBUTION.jsonl"
        )
        if {_identity(row) for row in legacy_rows} != {
            _identity(row) for row in literal_rows
        }:
            raise RuntimeError("P15 proxy and P22 literal Actor identities differ")
        legacy_rows = sorted(legacy_rows, key=_identity)
        literal_rows = sorted(literal_rows, key=_identity)
        comparison = _comparison(legacy_rows, literal_rows)
        literal_summary = summarize_attributions(literal_rows)
        output_path = run_dir / "TRUE_FIRST_RETURN_ACTORS.jsonl"
        output_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in literal_rows
            ),
            encoding="utf-8",
        )
        summary = {
            "schema_version": "worldsim_v7.p22_av2_true_first_return_correction.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": config["verdict"],
            "input_evidence_status": "consumed_external_diagnostic_only",
            "cohort": {
                "log_count": len(selected),
                "actor_count": len(literal_rows),
                "failed_log_deletion": False,
            },
            "ray_operator": "minimum_positive_depth_within_lateral_tolerance",
            "proxy_vs_literal": comparison,
            "literal_attribution": literal_summary,
            "claim_boundary": config["claim_boundary"],
            "consumed_target_data_reread": True,
            "fresh_target_data_read": False,
            "resources": {
                "device": str(device),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device)
                / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "status.json",
            {
                "status": "done",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return {
            "run_dir": str(run_dir),
            "verdict": summary["verdict"],
            "proxy_vs_literal": comparison,
        }
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.repo_root.resolve(), args.run_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

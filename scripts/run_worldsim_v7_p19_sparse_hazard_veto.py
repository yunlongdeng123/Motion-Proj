"""Evaluate the frozen V7 P19 one-slot sparse hazard veto."""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from motion_proj.worldsim_v7.av2_four_action_compiler import compile_log
from motion_proj.worldsim_v7.completion_expert_router import sparse_hazard_veto_states
from motion_proj.worldsim_v7.completion_responsibility import (
    FeatureStandardizer,
    apply_completion_policy,
    summarize_actor_policy,
)
from motion_proj.worldsim_v7.nuscenes_actor_surface import (
    build_selected_index,
    compile_nuscenes_scene,
)
from motion_proj.worldsim_v7.ray_set_completion import (
    RaySetCompletionMLP,
    predict_ray_set,
)
from scripts.run_worldsim_v7_p18_two_expert_router import _load_p17r_expert


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _load_configs(config_path: Path, repo_root: Path):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    compiler = yaml.safe_load((repo_root / config["p2_config"]).read_text(encoding="utf-8"))
    source = yaml.safe_load((repo_root / config["p4_config"]).read_text(encoding="utf-8"))
    expert = yaml.safe_load(
        (repo_root / config["base_expert_config"]).read_text(encoding="utf-8")
    )
    return config, compiler, source, expert


def _source_test_bundles(
    compiler: Mapping[str, Any],
    source: Mapping[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    test_scenes = {"test": source["nuscenes"]["role_scenes"]["test"]}
    index = build_selected_index(
        Path(source["nuscenes"]["dataset_root"]),
        test_scenes,
        source["nuscenes"]["allowed_category_prefixes"],
    )
    output = []
    scenes = test_scenes["test"]
    for position, scene in enumerate(scenes):
        rows, diagnostics = compile_nuscenes_scene(
            scene,
            index["scenes"][scene],
            Path(source["nuscenes"]["dataset_root"]),
            source["nuscenes"]["actors"],
            compiler,
            device,
            include_diagnostics=True,
        )
        for row in rows:
            output.append(
                {"row": row, "diagnostics": diagnostics[row["track_id"]], "scene": scene}
            )
        print(
            json.dumps(
                {
                    "stage": "source_test_corpus",
                    "progress": f"{position + 1}/{len(scenes)}",
                    "scene": scene,
                    "actors": len(rows),
                }
            ),
            flush=True,
        )
    return output


def _evaluate_bundles(
    bundles: list[dict[str, Any]],
    model: RaySetCompletionMLP,
    standardizer: FeatureStandardizer,
    expert_config: Mapping[str, Any],
    compiler: Mapping[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    rows = []
    threshold = float(expert_config["model"]["forward_selection_threshold"])
    for bundle in bundles:
        diagnostics = bundle["diagnostics"]
        _, _, occupancy = predict_ray_set(
            model,
            standardizer,
            np.asarray(diagnostics["completion_features"], dtype=np.float32),
            threshold,
            device,
        )
        states, probabilities, veto_index = sparse_hazard_veto_states(
            occupancy, bool(bundle["row"]["hazardous"]), threshold
        )
        row = apply_completion_policy(
            bundle["row"],
            diagnostics,
            states,
            probabilities,
            compiler,
            expert_config["attribution"],
            device,
        )
        row["sparse_veto_count"] = int(veto_index is not None)
        row["sparse_veto_score"] = (
            float(occupancy[veto_index]) if veto_index is not None else None
        )
        if "scene" in bundle:
            row["scene_name"] = bundle["scene"]
        if "log_id" in bundle:
            row["log_id"] = bundle["log_id"]
        rows.append(row)
    return rows


def _decisions(policy: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "hazard_new_early_strictly_lower": (
            policy["hazard"]["p16"]["new_early_rate"]
            < policy["hazard"]["baseline"]["new_early_rate"]
        ),
        "population_chamfer_no_worse_than_frozen_baseline": (
            policy["p16"]["mean_chamfer_m"] <= policy["baseline"]["mean_chamfer_m"]
        ),
    }


def run_source(config_path: Path, repo_root: Path, run_id: str) -> dict[str, Any]:
    config, compiler, source, expert_config = _load_configs(config_path, repo_root)
    run_dir = Path(config["runs_root"]) / "worldsim_v7" / config["source_task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "source"})
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P19 source evaluation requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        model, standardizer = _load_p17r_expert(
            Path(config["base_expert_fit_run"]), expert_config, device
        )
        bundles = _source_test_bundles(compiler, source, device)
        rows = _evaluate_bundles(
            bundles, model, standardizer, expert_config, compiler, device
        )
        policy = summarize_actor_policy(rows)
        decisions = _decisions(policy)
        _write_jsonl(run_dir / "SOURCE_TEST_ACTORS.jsonl", rows)
        summary = {
            "schema_version": "worldsim_v7.p19_sparse_hazard_veto_source.v1",
            "task_id": config["source_task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": (
                "source_development_pass_external_authorized"
                if all(decisions.values())
                else "source_development_rejected_external_unread"
            ),
            "source_evidence_status": "consumed_development_only",
            "actor_count": len(rows),
            "veto_actor_count": int(sum(row["sparse_veto_count"] for row in rows)),
            "policy": policy,
            "decisions": decisions,
            "target_data_read": False,
            "resources": {
                "device": str(device),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "status.json",
            {
                "status": "done",
                "phase": "source",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return {
            "run_dir": str(run_dir),
            "verdict": summary["verdict"],
            "decisions": decisions,
        }
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "phase": "source",
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


def run_external(
    config_path: Path,
    repo_root: Path,
    run_id: str,
    source_run: Path,
) -> dict[str, Any]:
    config, compiler, _, expert_config = _load_configs(config_path, repo_root)
    source_summary = json.loads((source_run / "summary.json").read_text(encoding="utf-8"))
    if source_summary.get("verdict") != "source_development_pass_external_authorized":
        raise RuntimeError("P19 source development did not authorize external read")
    cohort = json.loads((repo_root / config["fresh_av2_cohort"]).read_text(encoding="utf-8"))
    state_root = Path(config["fresh_download_state"])
    missing = [
        row["log_id"]
        for row in cohort["logs"]
        if not (state_root / f"{row['log_id']}.complete").is_file()
    ]
    if missing:
        raise RuntimeError(f"P19 fresh download incomplete: {len(missing)} logs")
    run_dir = Path(config["runs_root"]) / "worldsim_v7" / config["external_task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "external"})
    device = torch.device(config["device"])
    torch.cuda.reset_peak_memory_stats(device)
    model, standardizer = _load_p17r_expert(
        Path(config["base_expert_fit_run"]), expert_config, device
    )
    started = time.monotonic()
    rows = []
    try:
        for position, cohort_row in enumerate(cohort["logs"]):
            log_id = cohort_row["log_id"]
            compiled = compile_log(
                Path(compiler["dataset_root"]) / log_id,
                compiler,
                device,
                include_diagnostics=True,
            )
            bundles = [
                {
                    "row": actor_row,
                    "diagnostics": compiled["compiled"]["diagnostics"][actor_row["track_id"]],
                    "log_id": log_id,
                }
                for actor_row in compiled["actor_rows"]
            ]
            log_rows = _evaluate_bundles(
                bundles, model, standardizer, expert_config, compiler, device
            )
            rows.extend(log_rows)
            print(
                json.dumps(
                    {
                        "stage": "fresh_AV2",
                        "progress": f"{position + 1}/{len(cohort['logs'])}",
                        "log_id": log_id,
                        "actors": len(log_rows),
                    }
                ),
                flush=True,
            )
        policy = summarize_actor_policy(rows)
        decisions = _decisions(policy)
        verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
        _write_jsonl(run_dir / "FRESH_AV2_ACTORS.jsonl", rows)
        summary = {
            "schema_version": "worldsim_v7.p19_sparse_hazard_veto_external.v1",
            "task_id": config["external_task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": verdict,
            "source_run": str(source_run),
            "fresh_log_count": len(cohort["logs"]),
            "actor_count": len(rows),
            "veto_actor_count": int(sum(row["sparse_veto_count"] for row in rows)),
            "policy": policy,
            "decisions": decisions,
            "resources": {
                "device": str(device),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "status.json",
            {
                "status": "done",
                "phase": "external",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return {"run_dir": str(run_dir), "verdict": verdict, "decisions": decisions}
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "phase": "external",
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--phase", choices=("source", "external"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run", type=Path)
    args = parser.parse_args()
    if args.phase == "source":
        result = run_source(args.config.resolve(), args.repo_root.resolve(), args.run_id)
    else:
        if args.source_run is None:
            parser.error("--source-run is required for external")
        result = run_external(
            args.config.resolve(),
            args.repo_root.resolve(),
            args.run_id,
            args.source_run.resolve(),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

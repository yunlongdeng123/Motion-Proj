#!/usr/bin/env python3
"""Build and replay-freeze the result-blind WorldSim V5 nuScenes cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.worldsim_v4.datasets.nuscenes import (
    CohortError,
    canonical_json_bytes,
    canonical_json_sha256,
)
from motion_proj.worldsim_v5.datasets.nuscenes import (
    SELECTION_FIELDS,
    TASK_ID,
    build_fresh_cohort,
    select_fresh_scene_cohort,
)
from worldsim_v5_forensics_common import (
    atomic_json,
    atomic_text,
    copy_source_snapshot,
    inventory_files,
    load_json_mapping,
    prepare_formal_run,
    sha256_file,
    utc_now,
    verify_file,
    write_events,
    write_resolved_config,
)


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("task_id") != TASK_ID:
        raise CohortError("V5 D0 fresh cohort config task_id drift")
    if payload.get("status") not in {"running", "done"}:
        raise CohortError("V5 D0 fresh cohort config status 必须为 running/done")
    if payload["project"].get("p0_status") != "done":
        raise CohortError("V5 D0 fresh cohort 需要 P0 done")
    restrictions = payload["restrictions"]
    for key in (
        "sensor_blob_expansion_for_selection",
        "training",
        "model_inference",
        "parameter_search",
        "fresh_test_quality_read",
        "v4_scene_confirmatory_reuse",
    ):
        if restrictions.get(key) is not False:
            raise CohortError(f"V5 D0 restriction violated: {key}")
    if payload["selection"].get("selection_uses_model_results") is not False:
        raise CohortError("V5 D0 selection 不得使用模型结果")
    if tuple(payload["selection"]["allowed_metadata_fields"]) != SELECTION_FIELDS:
        raise CohortError("V5 D0 allowed metadata fields drift")
    return payload


def _scene_roles(manifest: Mapping[str, Any]) -> dict[str, list[str]]:
    return {
        role: [str(row["scene"]) for row in manifest["scenes"] if row["role"] == role]
        for role in ("development", "validation", "test")
    }


def _frozen_scene_records(manifest: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "scene": str(row["scene"]),
            "scene_token": str(row["scene_token"]),
            "role": str(row["role"]),
            "official_split": str(row["official_split"]),
        }
        for row in manifest["scenes"]
    ]


def _candidate_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "scene",
            "scene_token",
            "official_split",
            "location",
            "time_of_day",
            "weather",
            "road_geometry",
            "actor_class",
            "speed_regime",
            "distance_regime",
            "occlusion",
            "donor_support",
            "eligible_actor_count",
            "sensor_contract_complete",
            "sample_count",
        )
    }


def build(config: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    excluded = list(config["v4_exclusion"]["scenes"])
    if len(excluded) != int(config["v4_exclusion"]["expected_count"]):
        raise CohortError("V5 D0 V4 exclusion count drift")
    protocol = dict(config["protocol"])
    protocol["version"] = config["dataset"]["version"]
    protocol.setdefault("continuous_clip_keyframes", 7)
    manifest, candidates = build_fresh_cohort(
        str(config["dataset"]["metadata_root"]),
        protocol,
        seed=int(config["selection"]["seed"]),
        excluded_scenes=excluded,
    )
    replay = select_fresh_scene_cohort(
        candidates,
        seed=int(config["selection"]["seed"]),
        excluded_scenes=excluded,
    )
    if canonical_json_bytes(replay) != canonical_json_bytes(manifest["scenes"]):
        raise CohortError("V5 D0 deterministic replay drift")
    cohort_sha = canonical_json_sha256(manifest)
    roles = _scene_roles(manifest)
    freeze = config["freeze"]
    frozen = (
        config["status"] == "done"
        and freeze.get("selection_status") == "frozen"
        and freeze.get("cohort_sha256") is not None
    )
    if frozen:
        if cohort_sha != freeze["cohort_sha256"]:
            raise CohortError("V5 D0 frozen cohort SHA drift")
        if roles != freeze["scene_roles"]:
            raise CohortError("V5 D0 frozen scene roles drift")
        if _frozen_scene_records(manifest) != freeze["scene_records"]:
            raise CohortError("V5 D0 frozen scene records drift")
        if manifest["metadata_inventory_sha256"] != freeze["metadata_inventory_sha256"]:
            raise CohortError("V5 D0 metadata inventory SHA drift")
        diagnostic = freeze.get("source_diagnostic", {})
        diagnostic_run = Path(str(diagnostic.get("run", "")))
        cohort_record = verify_file(
            diagnostic_run / "artifacts/nuscenes_fresh_cohort.json",
            str(diagnostic.get("cohort_artifact_sha256")),
        )
        summary_record = verify_file(
            diagnostic_run / "summary.json", str(diagnostic.get("summary_sha256"))
        )
        verify_file(
            diagnostic_run
            / "source_snapshot/scripts/build_worldsim_v5_nuscenes_fresh_cohort.py",
            str(freeze["diagnostic_builder_source_sha256"]),
        )
        verify_file(
            diagnostic_run / "source_snapshot/motion_proj/worldsim_v5/datasets/nuscenes.py",
            str(freeze["diagnostic_selector_source_sha256"]),
        )
        diagnostic_manifest = load_json_mapping(Path(cohort_record["path"]))
        diagnostic_summary = load_json_mapping(Path(summary_record["path"]))
        if diagnostic_manifest != manifest:
            raise CohortError("V5 D0 diagnostic/full frozen manifest drift")
        if (
            diagnostic_summary.get("conclusion") != "metadata_selection_diagnostic"
            or diagnostic_summary.get("fresh_quality_read") is not False
            or diagnostic_summary.get("parameter_search_performed") is not False
        ):
            raise CohortError("V5 D0 diagnostic provenance drift")
    return manifest, candidates, frozen


def run(config_path: Path, run_dir: Path) -> dict[str, Any]:
    project = Path(__file__).resolve().parents[1]
    project_head = prepare_formal_run(run_dir, TASK_ID, project)
    config = _load_config(config_path)
    resolved_config = write_resolved_config(run_dir, config)
    source_inventory = copy_source_snapshot(
        run_dir,
        [
            Path(__file__),
            Path(__file__).with_name("worldsim_v5_forensics_common.py"),
            project / "motion_proj/worldsim_v5/datasets/nuscenes.py",
            project / "motion_proj/worldsim_v4/datasets/nuscenes.py",
            config_path,
            project / "configs/worldsim_v4/nuscenes_cohort_v1.yaml",
            project / "tests/test_worldsim_v5_nuscenes_fresh_cohort.py",
        ],
        project,
    )
    started = utc_now()
    manifest, candidates, frozen = build(config)
    cohort_sha = canonical_json_sha256(manifest)
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    cohort_path = artifacts / "nuscenes_fresh_cohort.json"
    cohort_path.write_bytes(canonical_json_bytes(manifest))
    metadata_path = artifacts / "metadata_inventory.json"
    atomic_json(metadata_path, manifest["metadata_fingerprints"])
    candidate_path = artifacts / "nuscenes_candidates.jsonl"
    atomic_text(
        candidate_path,
        "".join(
            json.dumps(
                _candidate_summary(row),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in sorted(candidates, key=lambda item: item["scene"])
        ),
    )
    if frozen and sha256_file(candidate_path) != config["freeze"]["candidate_inventory_sha256"]:
        raise CohortError("V5 D0 frozen candidate inventory SHA drift")
    atomic_text(artifacts / "split.sha256", f"{cohort_sha}  nuscenes_fresh_cohort.json\n")
    events = write_events(
        run_dir,
        [
            {"event": "metadata_selection_started", "task_id": TASK_ID, "timestamp_utc": started},
            {
                "event": "metadata_selection_completed",
                "task_id": TASK_ID,
                "frozen_replay": frozen,
                "timestamp_utc": utc_now(),
            },
        ],
    )
    conclusion = "fresh_cohort_frozen" if frozen else "metadata_selection_diagnostic"
    task_status = "done" if frozen else "running"
    summary = {
        "schema_version": "worldsim_v5_nuscenes_fresh_cohort_summary_v1",
        "task_id": TASK_ID,
        "task_status": task_status,
        "status": "done",
        "conclusion": conclusion,
        "phase": "frozen_replay" if frozen else "diagnostic_selection",
        "project_git_head": project_head,
        "cohort_sha256": cohort_sha,
        "scene_counts": manifest["scene_counts"],
        "scene_roles": _scene_roles(manifest),
        "candidate_scene_count": manifest["candidate_scene_count"],
        "excluded_v4_scene_count": manifest["excluded_v4_scene_count"],
        "metadata_inventory_sha256": manifest["metadata_inventory_sha256"],
        "candidate_inventory_sha256": sha256_file(candidate_path),
        "source_diagnostic": config["freeze"].get("source_diagnostic") if frozen else None,
        "deterministic_replay_exact": True,
        "selection_uses_model_results": False,
        "sensor_blob_expansion_for_selection": False,
        "training_performed": False,
        "model_inference_performed": False,
        "fresh_quality_read": False,
        "test_quality_read": False,
        "parameter_search_performed": False,
        "source_snapshot": source_inventory,
        "finished_at_utc": utc_now(),
    }
    atomic_json(run_dir / "summary.json", summary)
    fingerprint = {
        "schema_version": "worldsim_v5_nuscenes_fresh_cohort_fingerprint_v1",
        "task_id": TASK_ID,
        "project_git_head": project_head,
        "resolved_config": resolved_config,
        "metadata_fingerprints": manifest["metadata_fingerprints"],
        "cohort_sha256": cohort_sha,
        "checkpoint": {"applicable": False, "reason": "metadata_only_selection"},
        "fresh_quality_read": False,
        "test_quality_read": False,
        "parameter_search_performed": False,
    }
    atomic_json(run_dir / "fingerprint.json", fingerprint)
    run_manifest = {
        "schema_version": "worldsim_v5_nuscenes_fresh_cohort_manifest_v1",
        "task_id": TASK_ID,
        "task_status": task_status,
        "status": "done",
        "inventory": inventory_files(run_dir, {"manifest.json", "status.json"}),
    }
    atomic_json(run_dir / "manifest.json", run_manifest)
    status = {
        "schema_version": "worldsim_v5_nuscenes_fresh_cohort_status_v1",
        "task_id": TASK_ID,
        "task_status": task_status,
        "status": "done",
        "conclusion": conclusion,
        "project_git_head": project_head,
        "summary_sha256": sha256_file(run_dir / "summary.json"),
        "manifest_sha256": sha256_file(run_dir / "manifest.json"),
        "resolved_config_sha256": resolved_config["sha256"],
        "events_sha256": events["sha256"],
        "fresh_quality_read": False,
        "test_quality_read": False,
        "parameter_search_performed": False,
        "finished_at_utc": utc_now(),
    }
    atomic_json(run_dir / "status.json", status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.run_dir), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

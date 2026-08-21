"""WorldSim V6 R13 同场景 SceneIR actor 与 sensor/perception 证据绑定实验。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from motion_proj.worldsim_v6.r12_dynamic_logsim import _replay_once
from motion_proj.worldsim_v6.sceneir import load_sceneir, write_sceneir
from motion_proj.worldsim_v6.sceneir_adapters import streetgs_to_sceneir


TASK_ID = "WS-V6-R13-WORLDSIM-01"
FACTORS = ("actor_states", "trajectories", "semantic_labels", "collision_labels")


class R13BindingError(RuntimeError):
    """R13 SceneIR/sensor binding 正式合同失败。"""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _resolve_runs_uri(uri: str) -> Path:
    prefix = "runs://"
    if not uri.startswith(prefix) or ".." in Path(uri[len(prefix) :]).parts:
        raise R13BindingError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / uri[len(prefix) :]).resolve()


def _comparable_replay(replay: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(replay)
    value.pop("repeat_index", None)
    return value


def _without_actor(rows: list[dict[str, Any]], actor_id: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("actor_id") != actor_id]


def _collisions_without_actor(rows: list[dict[str, Any]], actor_id: str) -> list[dict[str, Any]]:
    return [row for row in rows if actor_id not in row["actor_pair"]]


def _remove_actor_document(
    document: Mapping[str, Any],
    arrays: Mapping[str, Mapping[str, np.ndarray]],
    actor_id: str,
) -> tuple[dict[str, Any], dict[str, Mapping[str, np.ndarray]]]:
    edited = copy.deepcopy(document)
    actor = next(row for row in edited["actors"] if row["id"] == actor_id)
    chunk_ids = set(actor["chunk_ids"])
    canonical_frame = actor["canonical_frame"]
    edited["actors"] = [row for row in edited["actors"] if row["id"] != actor_id]
    edited["chunks"] = [row for row in edited["chunks"] if row["id"] not in chunk_ids]
    edited["frames"] = [row for row in edited["frames"] if row["id"] != canonical_frame]
    edited["transforms"] = [
        row
        for row in edited["transforms"]
        if row["src_frame"] != canonical_frame and row["dst_frame"] != canonical_frame
    ]
    edited.pop("content_sha256", None)
    edited_arrays = {key: value for key, value in arrays.items() if key not in chunk_ids}
    return edited, edited_arrays


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R13BindingError("正式 R13 SceneIR binding run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R13BindingError("R13 SceneIR binding task_id 漂移")
    sources = config["sources"]
    cohort_run = _resolve_runs_uri(sources["actor_cohort_run"])
    dynamic_run = _resolve_runs_uri(sources["typed_dynamic_run"])
    checkpoint_path = Path(sources["streetgs_checkpoint"])
    frozen = {
        cohort_run / "MANIFEST.json": sources["actor_cohort_manifest_sha256"],
        cohort_run / "R13_ACTOR_COHORT_GATE.json": sources["actor_cohort_gate_sha256"],
        cohort_run / "ACTOR_VERDICTS.jsonl": sources["actor_verdicts_sha256"],
        dynamic_run / "R13_DYNAMIC_EDIT_GATE.json": sources["typed_dynamic_gate_sha256"],
        checkpoint_path: sources["streetgs_checkpoint_sha256"],
    }
    for path, expected in frozen.items():
        if _sha256(path) != expected:
            raise R13BindingError(f"冻结输入漂移：{path}")
    cohort_gate = json.loads(
        (cohort_run / "R13_ACTOR_COHORT_GATE.json").read_text(encoding="utf-8")
    )
    dynamic_gate = json.loads(
        (dynamic_run / "R13_DYNAMIC_EDIT_GATE.json").read_text(encoding="utf-8")
    )
    binding = config["binding"]
    model_index = int(binding["accepted_frontend_model_index"])
    verdicts = _read_jsonl(cohort_run / "ACTOR_VERDICTS.jsonl")
    selected_verdict = next(row for row in verdicts if int(row["model_index"]) == model_index)
    if not selected_verdict["v6_accepted"] or not cohort_gate["checks"]["passed"]:
        raise R13BindingError("冻结 actor cohort 未接受目标 model index")
    if not dynamic_gate["checks"]["passed"]:
        raise R13BindingError("冻结 typed dynamic gate 未通过")
    if shutil.disk_usage(run_root).free / (1024**3) < float(config["resources"]["minimum_disk_free_gib"]):
        raise R13BindingError("R13 SceneIR binding 磁盘资源不足")
    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / f"{now.strftime('%Y%m%dT%H%M%SZ')}__sceneir-sensor-binding-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        immutable_before = {str(path): _sha256(path) for path in frozen}
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        document, arrays = streetgs_to_sceneir(
            checkpoint,
            source_sha256=sources["streetgs_checkpoint_sha256"],
            source_uri="checkpoint://streetgs_scene0242_matched",
            reconstructor_version=sources["streetgs_reconstructor_version"],
            seed=int(config["seed"]),
        )
        base_package = run_dir / "sceneir_packages/streetgs_scene0242_base"
        base_written = write_sceneir(base_package, document, arrays)
        actor_id = binding["expected_sceneir_actor_id"]
        chunk_id = binding["expected_sceneir_chunk_id"]
        actor = next(row for row in base_written["actors"] if row["id"] == actor_id)
        chunk = next(row for row in base_written["chunks"] if row["id"] == chunk_id)
        edited_document, edited_arrays = _remove_actor_document(document, arrays, actor_id)
        edited_package = run_dir / "sceneir_packages/streetgs_scene0242_actor0000_removed"
        edited_written = write_sceneir(edited_package, edited_document, edited_arrays)
        package_hashes_before = {
            "base_manifest": _sha256(base_package / "MANIFEST.json"),
            "base_sceneir": _sha256(base_package / "sceneir.json"),
            "edited_manifest": _sha256(edited_package / "MANIFEST.json"),
            "edited_sceneir": _sha256(edited_package / "sceneir.json"),
        }
        replay_count = int(config["cohort"]["replay_count"])
        base_replays = [_replay_once(base_package, index) for index in range(1, replay_count + 1)]
        edited_replays = [
            _replay_once(edited_package, index) for index in range(1, replay_count + 1)
        ]
        _write_json(run_dir / "BASE_REPLAY.json", base_replays[0])
        _write_json(run_dir / "EDITED_REPLAY.json", edited_replays[0])
        base_repeat_exact = len(
            {_content_sha256(_comparable_replay(row)) for row in base_replays}
        ) == 1
        edited_repeat_exact = len(
            {_content_sha256(_comparable_replay(row)) for row in edited_replays}
        ) == 1
        base_replay = base_replays[0]
        edited_replay = edited_replays[0]
        base_actor_ids = {row["actor_id"] for row in base_replay["actor_states"]}
        edited_actor_ids = {row["actor_id"] for row in edited_replay["actor_states"]}
        timestamps = {int(row["timestamp_us"]) for row in base_replay["actor_states"]}
        cohort = config["cohort"]
        denominators = {
            "base_actor_count": len(base_actor_ids),
            "edited_actor_count": len(edited_actor_ids),
            "timestamp_count": len(timestamps),
            "base_trajectory_rows": len(base_replay["trajectories"]),
            "edited_trajectory_rows": len(edited_replay["trajectories"]),
            "base_collision_rows": len(base_replay["collision_labels"]),
            "edited_collision_rows": len(edited_replay["collision_labels"]),
        }
        denominator_exact = denominators == {
            "base_actor_count": int(cohort["expected_base_actor_count"]),
            "edited_actor_count": int(cohort["expected_edited_actor_count"]),
            "timestamp_count": int(cohort["expected_timestamp_count"]),
            "base_trajectory_rows": int(cohort["expected_base_trajectory_rows"]),
            "edited_trajectory_rows": int(cohort["expected_edited_trajectory_rows"]),
            "base_collision_rows": int(cohort["expected_base_collision_rows"]),
            "edited_collision_rows": int(cohort["expected_edited_collision_rows"]),
        }
        unaffected_checks = {
            "actor_states": _canonical(_without_actor(edited_replay["actor_states"], actor_id))
            == _canonical(_without_actor(base_replay["actor_states"], actor_id)),
            "trajectories": _canonical(_without_actor(edited_replay["trajectories"], actor_id))
            == _canonical(_without_actor(base_replay["trajectories"], actor_id)),
            "semantic_labels": _canonical(
                _without_actor(edited_replay["semantic_labels"], actor_id)
            )
            == _canonical(_without_actor(base_replay["semantic_labels"], actor_id)),
            "collision_labels": _canonical(
                _collisions_without_actor(edited_replay["collision_labels"], actor_id)
            )
            == _canonical(_collisions_without_actor(base_replay["collision_labels"], actor_id)),
        }
        package_hashes_after = {
            "base_manifest": _sha256(base_package / "MANIFEST.json"),
            "base_sceneir": _sha256(base_package / "sceneir.json"),
            "edited_manifest": _sha256(edited_package / "MANIFEST.json"),
            "edited_sceneir": _sha256(edited_package / "sceneir.json"),
        }
        binding_checks = {
            "model_index_to_actor_id": actor_id == f"actor_{model_index:04d}",
            "actor_chunk_id": actor["chunk_ids"] == [chunk_id],
            "primitive_count": int(chunk["primitive_count"])
            == int(binding["expected_actor_primitive_count"])
            == int(selected_verdict["gaussian_count"]),
            "actor_removed_from_document": actor_id
            not in {row["id"] for row in edited_written["actors"]},
            "chunk_removed_from_document": chunk_id
            not in {row["id"] for row in edited_written["chunks"]},
            "actor_removed_from_replay": actor_id not in edited_actor_ids,
        }
        _write_json(
            run_dir / "BINDING_AUDIT.json",
            {
                "schema_version": "worldsim_v6.r13_sceneir_sensor_binding_audit.v1",
                "scene": cohort["scene"],
                "checkpoint_sha256": sources["streetgs_checkpoint_sha256"],
                "frontend_model_index": model_index,
                "sceneir_actor_id": actor_id,
                "sceneir_chunk_id": chunk_id,
                "primitive_count": int(chunk["primitive_count"]),
                "actor_cohort_verdict": selected_verdict,
                "binding_checks": binding_checks,
                "denominators": denominators,
                "unaffected_checks": unaffected_checks,
            },
        )
        factor_rows = []
        for factor in FACTORS:
            factor_rows.append(
                {
                    "factor": factor,
                    "base_sha256": _content_sha256(base_replay[factor]),
                    "edited_sha256": _content_sha256(edited_replay[factor]),
                    "unaffected_exact": unaffected_checks[factor],
                }
            )
        _write_jsonl(run_dir / "FACTOR_BINDING_HASHES.jsonl", factor_rows)
        unsupported = config["unsupported_metrics"]
        wall_seconds = time.monotonic() - started
        checks = {
            "actor_identity_and_primitive_count_binding": all(binding_checks.values()),
            "same_scene_checkpoint": cohort["scene"] == "scene-0242",
            "actor_cohort_accept": selected_verdict["v6_accepted"],
            "typed_sceneir_removal_dependency_closure": denominator_exact,
            "unaffected_state_and_collision_exact": all(unaffected_checks.values()),
            "fresh_replay_exact": base_repeat_exact and edited_repeat_exact,
            "package_immutable_during_replay": package_hashes_before == package_hashes_after,
            "source_immutable": immutable_before == {str(path): _sha256(path) for path in frozen},
            "inherited_v6_false_safe_rate_zero": float(cohort_gate["v6_false_safe_rate"])
            == 0.0,
            "unsupported_metrics_abstain": all(str(value).startswith("ABSTAIN") for value in unsupported.values()),
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True,
            "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        gate = {
            "schema_version": "worldsim_v6.r13_sceneir_sensor_binding_gate.v1",
            "checks": checks,
            "unsupported_metrics": unsupported,
            "decision": "accept_same_scene_actor_dependency_sensor_binding"
            if checks["passed"]
            else "reject_sceneir_sensor_binding_hypothesis",
        }
        _write_json(run_dir / "R13_SCENEIR_SENSOR_BINDING_GATE.json", gate)
        summary = {
            "schema_version": "worldsim_v6.r13_sceneir_sensor_binding_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_same_scene_actor_binding"
            if checks["passed"]
            else "rejected",
            "source_commit": source_commit,
            "scene": cohort["scene"],
            "frontend_model_index": model_index,
            "sceneir_actor_id": actor_id,
            "primitive_count": int(chunk["primitive_count"]),
            "base_actor_count": len(base_actor_ids),
            "edited_actor_count": len(edited_actor_ids),
            "fresh_replay_exact": base_repeat_exact and edited_repeat_exact,
            "inherited_verified_actor_coverage": float(cohort_gate["verified_actor_coverage"]),
            "inherited_v6_false_safe_rate": float(cohort_gate["v6_false_safe_rate"]),
            "wall_seconds": wall_seconds,
            "unsupported_metrics": unsupported,
            "claim_boundary": config["claim_boundary"],
            "training_started": False,
            "confirmation_content_read": False,
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = [
            "BASE_REPLAY.json",
            "EDITED_REPLAY.json",
            "BINDING_AUDIT.json",
            "FACTOR_BINDING_HASHES.jsonl",
            "R13_SCENEIR_SENSOR_BINDING_GATE.json",
            "SUMMARY.json",
            "sceneir_packages/streetgs_scene0242_base/MANIFEST.json",
            "sceneir_packages/streetgs_scene0242_base/sceneir.json",
            "sceneir_packages/streetgs_scene0242_actor0000_removed/MANIFEST.json",
            "sceneir_packages/streetgs_scene0242_actor0000_removed/sceneir.json",
        ]
        _write_json(
            run_dir / "MANIFEST.json",
            {
                "schema_version": "worldsim_v6.r13_sceneir_sensor_binding_manifest.v1",
                "source_commit": source_commit,
                "config": str(config_path),
                "files": {
                    relative: {
                        "bytes": (run_dir / relative).stat().st_size,
                        "sha256": _sha256(run_dir / relative),
                    }
                    for relative in tracked
                },
                "package_manifest_sha256": package_hashes_after,
            },
        )
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": summary["status"],
                "task_id": TASK_ID,
                "hypothesis_id": config["hypothesis_id"],
                "manifest_sha256": _sha256(run_dir / "MANIFEST.json"),
            },
        )
        return run_dir
    except Exception as error:
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": "blocked",
                "task_id": TASK_ID,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/worldsim_v6/r13_sceneir_sensor_binding_v0.yaml"),
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6")
    )
    args = parser.parse_args()
    run_dir = run_experiment(args.repo_root, args.config, args.run_root)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

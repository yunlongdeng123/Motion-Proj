"""Audit P4 physical-Actor alignment with frozen V6.7 reliability rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _instance_map(scene_index: int, roots: list[Path]) -> tuple[dict[str, int], str | None]:
    for root in roots:
        path = root / f"{scene_index:03d}" / "instances" / "instances_info.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {str(row["id"]): int(actor_id) for actor_id, row in payload.items()}, str(root)
    return {}, None


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v7" / str(config["task_id"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(
        run_dir / "status.json",
        {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    try:
        scene_table = json.loads(Path(config["nuscenes_scene_json"]).read_text(encoding="utf-8"))
        scene_index = {str(row["name"]): index for index, row in enumerate(scene_table)}
        p4_run = Path(str(config["p4_run"]))
        role_rows = {
            role: _read_jsonl(p4_run / f"NUSCENES_{role.upper()}_ACTORS.jsonl")
            for role in ("train", "calibration", "test")
        }
        source = np.load(str(config["v67_source_rows"]), allow_pickle=False)
        source_scene = source["scene_index"].astype(np.int64)
        source_actor = source["actor_id"].astype(np.int64)
        source_pair_keys = (source_scene << 32) | source_actor
        unique_pair_keys, pair_counts = np.unique(source_pair_keys, return_counts=True)
        pair_count = {
            int(key): int(count) for key, count in zip(unique_pair_keys, pair_counts)
        }
        source_scenes = sorted(int(value) for value in np.unique(source_scene))
        fold_by_scene = {scene: rank % 5 for rank, scene in enumerate(source_scenes)}
        roots = [Path(value) for value in config["processed_roots"]]

        all_p4_scenes = sorted(
            {str(row["scene_name"]) for rows in role_rows.values() for row in rows}
        )
        identity_maps: dict[int, dict[str, int]] = {}
        identity_roots: dict[int, str | None] = {}
        for scene_name in all_p4_scenes:
            index = int(scene_index[scene_name])
            mapping, root = _instance_map(index, roots)
            identity_maps[index] = mapping
            identity_roots[index] = root

        role_summary: dict[str, Any] = {}
        aligned_records: list[dict[str, Any]] = []
        for role, rows in role_rows.items():
            p4_scene_indices = {int(scene_index[str(row["scene_name"])]) for row in rows}
            aligned_scenes: set[int] = set()
            mapped_actors = 0
            source_aligned_actors = 0
            aligned_source_rows = 0
            aligned_folds: Counter[int] = Counter()
            roots_used: Counter[str] = Counter()
            for row in rows:
                index = int(scene_index[str(row["scene_name"])])
                actor_id = identity_maps[index].get(str(row["track_id"]))
                if actor_id is None:
                    continue
                mapped_actors += 1
                if identity_roots[index] is not None:
                    roots_used[str(identity_roots[index])] += 1
                key = (index << 32) | int(actor_id)
                count = pair_count.get(key, 0)
                if count <= 0:
                    continue
                source_aligned_actors += 1
                aligned_source_rows += count
                aligned_scenes.add(index)
                aligned_folds[fold_by_scene[index]] += 1
                aligned_records.append(
                    {
                        "p4_role": role,
                        "scene_name": str(row["scene_name"]),
                        "scene_index": index,
                        "track_id": str(row["track_id"]),
                        "v67_actor_id": int(actor_id),
                        "v67_source_row_count": count,
                        "v67_scene_fold": int(fold_by_scene[index]),
                    }
                )
            role_summary[role] = {
                "p4_actor_count": len(rows),
                "p4_scene_count": len(p4_scene_indices),
                "p4_scenes_present_in_v67_source": len(p4_scene_indices.intersection(source_scenes)),
                "identity_mapped_actor_count": mapped_actors,
                "source_aligned_actor_count": source_aligned_actors,
                "source_aligned_scene_count": len(aligned_scenes),
                "aligned_source_row_count": aligned_source_rows,
                "aligned_scene_indices": sorted(aligned_scenes),
                "aligned_actor_count_by_v67_scene_fold": {
                    str(key): value for key, value in sorted(aligned_folds.items())
                },
                "identity_roots_used": dict(roots_used),
            }

        minimums = config["direct_joint_fit_minimums"]
        train = role_summary["train"]
        direct_joint_fit = bool(
            train["source_aligned_scene_count"]
            >= int(minimums["aligned_p4_train_scenes"])
            and train["source_aligned_actor_count"]
            >= int(minimums["aligned_p4_train_actors"])
        )
        decision = (
            "direct_joint_fit_supported"
            if direct_joint_fit
            else str(config["fallback_on_failure"])
        )
        _write_json(run_dir / "ALIGNED_ACTORS.json", aligned_records)
        summary = {
            "schema_version": "worldsim_v7.p5_physical_reliability_alignment.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "decision": decision,
            "claim_boundary": config["claim_boundary"],
            "v67_source": {
                "row_count": int(len(source_scene)),
                "scene_count": len(source_scenes),
                "actor_pair_count": int(len(unique_pair_keys)),
                "horizons_seconds": sorted(
                    float(value) for value in np.unique(source["horizon_seconds"])
                ),
                "feature_dimension": int(source["features"].shape[1]),
                "frozen_reliability_run": str(config["v67_reliability_run"]),
            },
            "roles": role_summary,
            "direct_joint_fit_minimums": minimums,
            "direct_joint_fit_allowed": direct_joint_fit,
            "aligned_actor_total": len(aligned_records),
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "status.json",
            {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
        )
        return {"run_dir": str(run_dir), **summary}
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
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.run_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

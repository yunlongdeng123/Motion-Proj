"""V6.4 三臂小型几何风险策略 post-training。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


TASK_ID = "WS-V6-PT2-RISK-POLICY-01"


class PT2RiskPolicyError(RuntimeError):
    """PT2 风险策略正式合同失败。"""


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


def _pose_set_sha256(directory: Path) -> str:
    # 与 preregistration 使用的 `sha256sum files | sha256sum` 字节合同一致。
    rows = []
    for path in sorted(directory.glob("*.txt")):
        rows.append(f"{_sha256(path)}  {path}\n")
    return hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


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


def _load_scene_sources(root: Path, spec: Mapping[str, Any], expected_frames: int) -> tuple[dict[str, Any], dict[int, np.ndarray], list[Path]]:
    scene_dir = root / f"{int(spec['scene_index']):03d}"
    instances_path = scene_dir / "instances" / "instances_info.json"
    pose_dir = scene_dir / "lidar_pose"
    if _sha256(instances_path) != spec["instances_sha256"]:
        raise PT2RiskPolicyError(f"{spec['scene']} instances hash 漂移")
    if _pose_set_sha256(pose_dir) != spec["lidar_pose_set_sha256"]:
        raise PT2RiskPolicyError(f"{spec['scene']} lidar pose set hash 漂移")
    pose_paths = sorted(pose_dir.glob("*.txt"))
    if len(pose_paths) != expected_frames:
        raise PT2RiskPolicyError(f"{spec['scene']} pose 分母漂移: {len(pose_paths)}")
    poses = {int(path.stem): np.loadtxt(path).reshape(4, 4) for path in pose_paths}
    instances = json.loads(instances_path.read_text(encoding="utf-8"))
    return instances, poses, [instances_path, *pose_paths]


def _actors_by_frame(instances: Mapping[str, Any]) -> dict[int, list[tuple[np.ndarray, np.ndarray]]]:
    result: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for info in instances.values():
        annotations = info.get("frame_annotations", {})
        frames = annotations.get("frame_idx", [])
        transforms = annotations.get("obj_to_world", [])
        sizes = annotations.get("box_size", [])
        for frame, transform, size in zip(frames, transforms, sizes):
            array = np.asarray(transform, dtype=np.float64).reshape(4, 4)
            box = np.asarray(size, dtype=np.float64).reshape(-1)
            if box.size < 2 or not np.isfinite(array).all() or not np.isfinite(box[:2]).all():
                continue
            result.setdefault(int(frame), []).append((array, box))
    return result


def _actor_signed_clearance(
    ego_pose: np.ndarray,
    actor_pose: np.ndarray,
    box_size: np.ndarray,
    ego_half: tuple[float, float],
) -> float:
    relative = np.linalg.inv(ego_pose) @ actor_pose
    x, y = float(relative[0, 3]), float(relative[1, 3])
    yaw = math.atan2(float(relative[1, 0]), float(relative[0, 0]))
    actor_hx = 0.5 * float(abs(box_size[0]))
    actor_hy = 0.5 * float(abs(box_size[1]))
    projected_hx = abs(math.cos(yaw)) * actor_hx + abs(math.sin(yaw)) * actor_hy
    projected_hy = abs(math.sin(yaw)) * actor_hx + abs(math.cos(yaw)) * actor_hy
    dx = abs(x) - (ego_half[0] + projected_hx)
    dy = abs(y) - (ego_half[1] + projected_hy)
    return max(dx, dy)


def _clean_clearance(
    ego_pose: np.ndarray,
    actors: list[tuple[np.ndarray, np.ndarray]],
    ego_half: tuple[float, float],
) -> float:
    if not actors:
        return 100.0
    return min(_actor_signed_clearance(ego_pose, pose, size, ego_half) for pose, size in actors)


def _scene_rows(
    spec: Mapping[str, Any],
    instances: Mapping[str, Any],
    poses: Mapping[int, np.ndarray],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    actors = _actors_by_frame(instances)
    part = config["partition"]
    geom = config["geometry"]
    ego_half = (0.5 * float(geom["ego_length_m"]), 0.5 * float(geom["ego_width_m"]))
    actor_size = np.asarray([geom["default_actor_length_m"], geom["default_actor_width_m"]], dtype=float)
    real_rows: list[dict[str, Any]] = []
    synthetic_rows: list[dict[str, Any]] = []
    for frame in sorted(poses):
        if frame < int(part["sample_start"]) or (frame - int(part["sample_start"])) % int(part["sample_stride"]):
            continue
        clean = _clean_clearance(poses[frame], actors.get(frame, []), ego_half)
        clean_label = int(clean <= 0.0)
        real_rows.append(
            {"scene": spec["scene"], "frame": frame, "case_type": "logged_clean", "signed_clearance_m": clean,
             "hazard_label": clean_label, "label_source": "logged_actor_geometry"}
        )
        for offset in geom["synthetic_clone_forward_offsets_m"]:
            # clone 与 ego 同航向，位于冻结纵向 offset；feature 同时保留其他真实 actor 的最危险值。
            clone_local = np.eye(4, dtype=float)
            clone_local[0, 3] = float(offset)
            clone_world = poses[frame] @ clone_local
            clone_clearance = _actor_signed_clearance(poses[frame], clone_world, actor_size, ego_half)
            edited_clearance = min(clean, clone_clearance)
            synthetic_rows.append(
                {"scene": spec["scene"], "frame": frame, "case_type": "typed_actor_clone",
                 "clone_forward_offset_m": float(offset), "signed_clearance_m": edited_clearance,
                 "hazard_label": int(edited_clearance <= 0.0), "stale_naive_label": clean_label,
                 "label_source": "recomputed_projected_aabb_dependency"}
            )
    return real_rows, synthetic_rows


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    recalls = []
    for value in (0, 1):
        mask = labels == value
        if mask.any():
            recalls.append(float((predictions[mask] == value).mean()))
    return float(sum(recalls) / len(recalls)) if recalls else 0.0


def _fit_threshold(rows: list[dict[str, Any]], label_key: str) -> dict[str, Any]:
    features = np.asarray([row["signed_clearance_m"] for row in rows], dtype=float)
    labels = np.asarray([row[label_key] for row in rows], dtype=int)
    unique = np.unique(features)
    candidates = [float(unique[0] - 1.0), float(unique[-1] + 1.0)]
    candidates.extend(float((left + right) * 0.5) for left, right in zip(unique[:-1], unique[1:]))
    candidates.extend(float(value) for value in unique)
    scored = []
    for threshold in candidates:
        predictions = (features <= threshold).astype(int)
        scored.append((_balanced_accuracy(labels, predictions), threshold))
    best_score = max(score for score, _ in scored)
    threshold = min((value for score, value in scored if score == best_score), key=lambda value: (abs(value), value))
    return {"threshold_m": threshold, "train_balanced_accuracy": best_score, "train_rows": len(rows),
            "train_positive_fraction": float(labels.mean())}


def _evaluate(model: Mapping[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = np.asarray([row["hazard_label"] for row in rows], dtype=int)
    predictions = np.asarray([int(row["signed_clearance_m"] <= model["threshold_m"]) for row in rows], dtype=int)
    positive = labels == 1
    negative = labels == 0
    false_safe = int(((predictions == 0) & positive).sum())
    false_brake = int(((predictions == 1) & negative).sum())
    return {
        "row_count": len(rows),
        "hazard_count": int(positive.sum()),
        "safe_count": int(negative.sum()),
        "balanced_accuracy": _balanced_accuracy(labels, predictions),
        "hazard_recall": float((predictions[positive] == 1).mean()) if positive.any() else 1.0,
        "false_safe_rate": false_safe / int(positive.sum()) if positive.any() else 0.0,
        "safe_route_completion": float((predictions[negative] == 0).mean()) if negative.any() else 1.0,
        "false_brake_rate": false_brake / int(negative.sum()) if negative.any() else 0.0,
    }


def _train_and_evaluate(
    train_real: list[dict[str, Any]], train_synthetic: list[dict[str, Any]], heldout: list[dict[str, Any]], arms: list[str]
) -> dict[str, Any]:
    result = {}
    for arm in arms:
        if arm == "real_only":
            rows, label_key = train_real, "hazard_label"
        elif arm == "real_plus_naive_synthetic":
            rows, label_key = [*train_real, *train_synthetic], "stale_naive_label"
            rows = [dict(row, stale_naive_label=row.get("stale_naive_label", row["hazard_label"])) for row in rows]
        elif arm == "real_plus_v6_verified_compiled":
            rows, label_key = [*train_real, *train_synthetic], "hazard_label"
        else:
            raise PT2RiskPolicyError(f"未知方法臂: {arm}")
        model = _fit_threshold(rows, label_key)
        result[arm] = {"model": model, "heldout": _evaluate(model, heldout)}
    return result


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    repo_root = repo_root.resolve()
    config_path = (repo_root / config_path).resolve() if not config_path.is_absolute() else config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["task_id"] != TASK_ID:
        raise PT2RiskPolicyError("task_id 不匹配")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{stamp}__risk-policy-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        if shutil.disk_usage(run_root).free / (1024**3) < float(config["resources"]["minimum_disk_free_gib"]):
            raise PT2RiskPolicyError("磁盘资源不足")
        source_commit = _git(repo_root, "rev-parse", "HEAD")
        if _git(repo_root, "status", "--short"):
            raise PT2RiskPolicyError("正式运行要求 clean worktree")
        data_root = Path(config["dataset_root"])
        expected_frames = int(config["partition"]["expected_frame_count_per_scene"])
        frozen_paths = [config_path]
        train_real: list[dict[str, Any]] = []
        train_synthetic: list[dict[str, Any]] = []
        source_audit = []
        for spec in config["train_scenes"]:
            instances, poses, paths = _load_scene_sources(data_root, spec, expected_frames)
            real, synthetic = _scene_rows(spec, instances, poses, config)
            train_real.extend(real)
            train_synthetic.extend(synthetic)
            frozen_paths.extend(paths)
            source_audit.append({"scene": spec["scene"], "real_rows": len(real), "synthetic_rows": len(synthetic)})
        heldout_spec = config["heldout_scene"]
        heldout_instances, heldout_poses, heldout_paths = _load_scene_sources(data_root, heldout_spec, expected_frames)
        heldout_real, heldout_synthetic = _scene_rows(heldout_spec, heldout_instances, heldout_poses, config)
        heldout_rows = [*heldout_real, *heldout_synthetic]
        frozen_paths.extend(heldout_paths)
        hashes_before = {str(path): _sha256(path) for path in frozen_paths}
        arms1 = _train_and_evaluate(train_real, train_synthetic, heldout_rows, list(config["arms"]))
        arms2 = _train_and_evaluate(train_real, train_synthetic, heldout_rows, list(config["arms"]))
        repeat_exact = _canonical(arms1) == _canonical(arms2)
        _write_json(run_dir / "POLICY_ARMS.json", arms1)
        _write_jsonl(run_dir / "TRAIN_REAL_ROWS.jsonl", train_real)
        _write_jsonl(run_dir / "TRAIN_SYNTHETIC_ROWS.jsonl", train_synthetic)
        _write_jsonl(run_dir / "HELDOUT_ROWS.jsonl", heldout_rows)
        _write_json(run_dir / "SOURCE_AUDIT.json", {"train": source_audit, "heldout": {"scene": heldout_spec["scene"],
                    "real_rows": len(heldout_real), "synthetic_rows": len(heldout_synthetic)},
                    "source_sha256": hashes_before})
        v6 = arms1["real_plus_v6_verified_compiled"]["heldout"]
        naive = arms1["real_plus_naive_synthetic"]["heldout"]
        reduction = naive["false_safe_rate"] - v6["false_safe_rate"]
        wall_seconds = time.monotonic() - started
        gate_cfg = config["gate"]
        checks = {
            "independent_heldout_scene": heldout_spec["scene"] not in {row["scene"] for row in config["train_scenes"]},
            "fixed_source_hashes": hashes_before == {str(path): _sha256(path) for path in frozen_paths},
            "nonempty_train_and_heldout": bool(train_real and train_synthetic and heldout_real and heldout_synthetic),
            "heldout_has_both_outcomes": v6["hazard_count"] > 0 and v6["safe_count"] > 0,
            "v6_balanced_accuracy": v6["balanced_accuracy"] >= float(gate_cfg["require_v6_balanced_accuracy_at_least"]),
            "v6_false_safe": v6["false_safe_rate"] <= float(gate_cfg["require_v6_false_safe_rate_at_most"]),
            "v6_safe_route_completion": v6["safe_route_completion"] >= float(gate_cfg["require_v6_safe_route_completion_at_least"]),
            "false_safe_reduction_vs_naive": reduction >= float(gate_cfg["require_false_safe_reduction_vs_naive_at_least"]),
            "training_repeat_exact": repeat_exact,
            "unsupported_metrics_abstain": all(str(value).startswith("ABSTAIN") for value in config["unsupported_metrics"].values()),
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
            "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        gate = {"schema_version": "worldsim_v6.pt2_risk_policy_gate.v1", "checks": checks,
                "decision": "accept_small_risk_policy_post_training" if checks["passed"] else "reject_pt2_risk_policy",
                "false_safe_reduction_vs_naive": reduction, "unsupported_metrics": config["unsupported_metrics"]}
        _write_json(run_dir / "PT2_RISK_POLICY_GATE.json", gate)
        summary = {"schema_version": "worldsim_v6.pt2_risk_policy_summary.v1", "task_id": TASK_ID,
                   "hypothesis_id": config["hypothesis_id"], "status": "done" if checks["passed"] else "rejected",
                   "source_commit": source_commit, "train_scenes": [row["scene"] for row in config["train_scenes"]],
                   "heldout_scene": heldout_spec["scene"], "method_arms": arms1,
                   "false_safe_reduction_vs_naive": reduction, "wall_seconds": wall_seconds,
                   "training_started": True, "confirmation_content_read": False,
                   "claim_boundary": config["claim_boundary"], "unsupported_metrics": config["unsupported_metrics"]}
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = ["POLICY_ARMS.json", "TRAIN_REAL_ROWS.jsonl", "TRAIN_SYNTHETIC_ROWS.jsonl", "HELDOUT_ROWS.jsonl",
                   "SOURCE_AUDIT.json", "PT2_RISK_POLICY_GATE.json", "SUMMARY.json"]
        _write_json(run_dir / "MANIFEST.json", {"schema_version": "worldsim_v6.pt2_risk_policy_manifest.v1",
                    "source_commit": source_commit, "config": str(config_path),
                    "files": {name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)} for name in tracked}})
        _write_json(run_dir / "TERMINAL.json", {"schema_version": "worldsim_v6.terminal.v1",
                    "status": summary["status"], "task_id": TASK_ID, "hypothesis_id": config["hypothesis_id"],
                    "manifest_sha256": _sha256(run_dir / "MANIFEST.json")})
        return run_dir
    except Exception as error:
        _write_json(run_dir / "TERMINAL.json", {"schema_version": "worldsim_v6.terminal.v1", "status": "blocked",
                    "task_id": TASK_ID, "error_type": type(error).__name__, "error": str(error)})
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/pt2_risk_policy_v0.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    print(run_experiment(args.repo_root, args.config, args.run_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

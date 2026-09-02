"""Train one nuScenes-only visibility-risk head and audit frozen AV2 transfer."""

from __future__ import annotations

import argparse
import json
import math
import resource
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch import nn

from motion_proj.worldsim_v7.av2_four_action_compiler import _compile_actor
from motion_proj.worldsim_v7.nuscenes_actor_surface import (
    _associate_frame,
    _make_track,
    _read_lidar,
    build_selected_index,
)
from motion_proj.worldsim_v7.selective_validity_hazard import (
    SmallMLP,
    Standardizer,
    VALIDITY_FEATURE_NAMES,
)
from motion_proj.worldsim_v7.visibility_certificate import compile_actor_certificate


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _wilson_upper(failures: int, count: int, confidence: float = 0.95) -> float | None:
    if count == 0:
        return None
    z = NormalDist().inv_cdf(confidence)
    rate = failures / count
    denom = 1.0 + z * z / count
    center = rate + z * z / (2.0 * count)
    radius = z * math.sqrt(rate * (1.0 - rate) / count + z * z / (4.0 * count * count))
    return float((center + radius) / denom)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _auroc(scores: list[float], positive: list[bool]) -> float | None:
    y = np.asarray(positive, dtype=np.bool_)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return None
    ranks = _rankdata(np.asarray(scores, dtype=np.float64))
    return float((ranks[y].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def _features(rows: list[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[float(row["runtime_features"][name]) for name in VALIDITY_FEATURE_NAMES] for row in rows],
        dtype=np.float32,
    )


def _compile_source_scene(
    scene_name: str,
    frames: list[dict[str, Any]],
    dataset_root: Path,
    actor_config: Mapping[str, Any],
    compiler_config: Mapping[str, Any],
    visibility_config: Mapping[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    frame_by_sample = {str(frame["sample_token"]): frame for frame in frames}
    annotations_by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        for annotation in frame["annotations"]:
            annotations_by_instance[str(annotation["instance_token"])].append(
                {**annotation, "sample_token": str(frame["sample_token"])}
            )
    tracks = {
        track_id: track
        for track_id, annotations in annotations_by_instance.items()
        if (track := _make_track(track_id, annotations, frame_by_sample, actor_config)) is not None
    }
    frame_ranks = {
        track_id: {
            str(row["sample_token"]): rank
            for rank, row in enumerate(
                sorted(
                    annotations_by_instance[track_id],
                    key=lambda item: int(frame_by_sample[str(item["sample_token"])]["timestamp_us"]),
                )
            )
        }
        for track_id in tracks
    }
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        _associate_frame(
            _read_lidar(dataset_root / str(frame["lidar_filename"])),
            frame,
            tracks,
            frame_ranks,
            records,
            actor_config,
            device,
        )

    output: list[dict[str, Any]] = []
    for track_id in sorted(tracks):
        compiled = _compile_actor(
            tracks[track_id],
            sorted(records.get(track_id, []), key=lambda item: item["frame_rank"]),
            compiler_config,
            device,
            include_diagnostics=True,
        )
        if compiled is None:
            continue
        actor_row, package = compiled
        certificate = compile_actor_certificate(
            actor_row, package["diagnostics"], visibility_config, device
        )
        output.append(
            {
                **actor_row,
                "dataset": "nuScenes",
                "scene_name": scene_name,
                "role": str(frames[0]["role"]),
                "visibility_safe": bool(certificate["nonnew_visible_violation"]),
                "visibility_certificate": certificate,
            }
        )
    return output


def _train(
    features: np.ndarray,
    labels: np.ndarray,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[SmallMLP, list[dict[str, float]]]:
    torch.manual_seed(int(config["seed"]))
    model = SmallMLP(features.shape[1], int(config["hidden_dim"])).to(device)
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    y = torch.as_tensor(labels, dtype=torch.float32, device=device)
    positives = y.sum().clamp_min(1.0)
    negatives = (len(y) - y.sum()).clamp_min(1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=(negatives / positives).detach())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    history: list[dict[str, float]] = []
    for epoch in range(int(config["epochs"])):
        model.train()
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch in {0, int(config["epochs"]) - 1} or (epoch + 1) % 20 == 0:
            history.append({"epoch": epoch + 1, "loss": float(loss.detach().cpu())})
    return model, history


def _predict(model: SmallMLP, features: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        logits = model(torch.as_tensor(features, dtype=torch.float32, device=device))
    return torch.sigmoid(logits).cpu().numpy()


def _coverage_threshold(scores: np.ndarray, target_coverage: float) -> float:
    count = max(1, int(math.ceil(len(scores) * target_coverage)))
    return float(np.sort(scores)[-count])


def _group(rows: list[dict[str, Any]], total: int) -> dict[str, Any]:
    count = len(rows)
    visible = int(sum(not bool(row["visibility_safe"]) for row in rows))
    chamfer = int(sum(bool(row["chamfer_worsened"]) for row in rows))
    gains = [float(row["query_chamfer_m"]) - float(row["compiled_chamfer_m"]) for row in rows]
    return {
        "actor_count": count,
        "coverage": count / total,
        "visible_violation_count": visible,
        "visible_violation_rate": visible / count if count else None,
        "visible_violation_wilson_upper": _wilson_upper(visible, count),
        "chamfer_worsened_count": chamfer,
        "chamfer_worsened_rate": chamfer / count if count else None,
        "mean_chamfer_gain_m": float(np.mean(gains)) if gains else None,
        "hazard_actor_count": int(sum(bool(row["hazardous"]) for row in rows)),
    }


def _evaluate(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    total = len(rows)
    visibility_selected = [row for row in rows if float(row["visibility_score"]) >= threshold]
    p4_selected = [row for row in rows if bool(row["p4_selected"])]
    dual = [row for row in visibility_selected if bool(row["p4_selected"])]
    hazard_total = sum(bool(row["hazardous"]) for row in rows)
    return {
        "always": _group(rows, total),
        "visibility_selected": _group(visibility_selected, total),
        "p4_selected": _group(p4_selected, total),
        "dual_selected": _group(dual, total),
        "safe_visible_auroc": _auroc(
            [float(row["visibility_score"]) for row in rows],
            [bool(row["visibility_safe"]) for row in rows],
        ),
        "dual_hazard_coverage": (
            sum(bool(row["hazardous"]) for row in dual) / hazard_total if hazard_total else None
        ),
    }


def run(config_path: Path, repo_root: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    p4_config = yaml.safe_load((repo_root / str(config["p4_config"])).read_text(encoding="utf-8"))
    p3_config = yaml.safe_load((repo_root / str(config["p3_config"])).read_text(encoding="utf-8"))
    compiler_config = yaml.safe_load(
        (repo_root / str(p3_config["p2_config"])).read_text(encoding="utf-8")
    )
    compiler_config["compiler_geometry"].update(p3_config.get("compiler_overrides", {}))
    p3c_config = yaml.safe_load((repo_root / str(config["p3c_config"])).read_text(encoding="utf-8"))
    visibility_config = p3c_config["visibility_certificate"]
    run_dir = Path(config["runs_root"]) / "worldsim_v7" / str(config["task_id"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(run_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P12 requires the frozen CUDA device")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        index = build_selected_index(
            Path(str(p4_config["nuscenes"]["dataset_root"])),
            p4_config["nuscenes"]["role_scenes"],
            p4_config["nuscenes"]["allowed_category_prefixes"],
        )
        role_rows: dict[str, list[dict[str, Any]]] = {
            role: [] for role in p4_config["nuscenes"]["role_scenes"]
        }
        scene_order = [
            (role, scene)
            for role, scenes in p4_config["nuscenes"]["role_scenes"].items()
            for scene in scenes
        ]
        for position, (role, scene_name) in enumerate(scene_order):
            rows = _compile_source_scene(
                scene_name,
                index["scenes"][scene_name],
                Path(str(p4_config["nuscenes"]["dataset_root"])),
                p4_config["nuscenes"]["actors"],
                compiler_config,
                visibility_config,
                device,
            )
            role_rows[role].extend(rows)
            print(json.dumps({"stage": "source_visibility", "progress": f"{position + 1}/{len(scene_order)}", "role": role, "scene": scene_name, "actors": len(rows)}), flush=True)

        standardizer = Standardizer.fit(_features(role_rows["train"]))
        train_x = standardizer.transform(_features(role_rows["train"]))
        train_y = np.asarray([bool(row["visibility_safe"]) for row in role_rows["train"]], dtype=np.float32)
        model, history = _train(train_x, train_y, config["model"], device)
        source_scores = {
            role: _predict(model, standardizer.transform(_features(rows)), device)
            for role, rows in role_rows.items()
        }
        threshold = _coverage_threshold(
            source_scores["calibration"], float(config["calibration"]["target_visibility_coverage"])
        )

        p4_scores = _read_jsonl(Path(config["p4_run"]) / "SELECTIVE_SCORES.jsonl")
        source_p4 = {
            (str(row["role"]), str(row["scene_or_log"]), str(row["track_id"])): row
            for row in p4_scores
            if row.get("dataset") == "nuScenes"
        }
        source_eval_rows: dict[str, list[dict[str, Any]]] = {}
        for role in ("calibration", "test"):
            output: list[dict[str, Any]] = []
            for row, score in zip(role_rows[role], source_scores[role]):
                key = (role, str(row["scene_name"]), str(row["track_id"]))
                p4 = source_p4.get(key)
                if p4 is None:
                    raise RuntimeError(f"Missing frozen source P4 score for {key}")
                output.append(
                    {
                        "scene_or_log": str(row["scene_name"]),
                        "track_id": str(row["track_id"]),
                        "role": role,
                        "hazardous": bool(row["hazardous"]),
                        "visibility_safe": bool(row["visibility_safe"]),
                        "visibility_score": float(score),
                        "p4_selected": bool(p4["factorized_selected"]),
                        "query_chamfer_m": float(row["query_only"]["symmetric_chamfer_m"]),
                        "compiled_chamfer_m": float(row["after"]["symmetric_chamfer_m"]),
                        "chamfer_worsened": bool(row["after"]["symmetric_chamfer_m"] > row["query_only"]["symmetric_chamfer_m"]),
                    }
                )
            source_eval_rows[role] = output

        external_actors = _read_jsonl(Path(config["p6c_fresh_run"]) / "FRESH_AV2_ACTORS.jsonl")
        external_scores = _read_jsonl(Path(config["p6c_fresh_run"]) / "FRESH_AV2_SCORES.jsonl")
        external_certificates = _read_jsonl(Path(config["p3c_fresh_run"]) / "ACTOR_VISIBILITY_CERTIFICATES.jsonl")
        identity = lambda row: (str(row["log_id"]), str(row["track_id"]))
        actors_by_id = {identity(row): row for row in external_actors}
        scores_by_id = {identity(row): row for row in external_scores}
        certificates_by_id = {identity(row): row for row in external_certificates}
        if not (set(actors_by_id) == set(scores_by_id) == set(certificates_by_id)):
            raise RuntimeError("External frozen identity sets differ")
        external_order = sorted(actors_by_id)
        external_x = standardizer.transform(_features([actors_by_id[key] for key in external_order]))
        external_visibility_scores = _predict(model, external_x, device)
        external_rows: list[dict[str, Any]] = []
        for key, visibility_score in zip(external_order, external_visibility_scores):
            actor = actors_by_id[key]
            p4 = scores_by_id[key]
            certificate = certificates_by_id[key]
            external_rows.append(
                {
                    "scene_or_log": key[0],
                    "track_id": key[1],
                    "role": "consumed_external_development",
                    "hazardous": bool(actor["hazardous"]),
                    "visibility_safe": bool(certificate["nonnew_visible_violation"]),
                    "visibility_score": float(visibility_score),
                    "p4_selected": bool(p4["p4_selected"]),
                    "query_chamfer_m": float(certificate["query_chamfer_m"]),
                    "compiled_chamfer_m": float(certificate["compiled_chamfer_m"]),
                    "chamfer_worsened": bool(certificate["chamfer_worsened_vs_query"]),
                }
            )

        source_calibration = _evaluate(source_eval_rows["calibration"], threshold)
        source_test = _evaluate(source_eval_rows["test"], threshold)
        external = _evaluate(external_rows, threshold)
        external_dual = external["dual_selected"]
        external_p4 = external["p4_selected"]
        gates = {
            "minimum_source_test_safe_visible_auroc": source_test["safe_visible_auroc"] >= float(config["fixed_gates"]["minimum_source_test_safe_visible_auroc"]),
            "source_selected_visible_risk_below_always": source_test["visibility_selected"]["visible_violation_rate"] < source_test["always"]["visible_violation_rate"],
            "external_dual_visible_risk_below_p4": external_dual["visible_violation_rate"] is not None and external_dual["visible_violation_rate"] < external_p4["visible_violation_rate"],
            "external_dual_visible_upper_below_p4_point_risk": external_dual["visible_violation_wilson_upper"] is not None and external_dual["visible_violation_wilson_upper"] < external_p4["visible_violation_rate"],
            "external_dual_chamfer_worsening_not_above_p4": external_dual["chamfer_worsened_rate"] is not None and external_dual["chamfer_worsened_rate"] <= external_p4["chamfer_worsened_rate"],
            "minimum_external_dual_coverage": external_dual["coverage"] >= float(config["fixed_gates"]["minimum_external_dual_coverage"]),
            "minimum_external_dual_hazard_coverage": external["dual_hazard_coverage"] is not None and external["dual_hazard_coverage"] >= float(config["fixed_gates"]["minimum_external_dual_hazard_coverage"]),
        }
        torch.save(
            {"model_state": model.state_dict(), "standardizer": standardizer.payload(), "feature_names": list(VALIDITY_FEATURE_NAMES), "threshold": threshold, "model_config": dict(config["model"])},
            run_dir / "MODEL.pt",
        )
        _write_jsonl(run_dir / "NUSCENES_VISIBILITY_ACTORS.jsonl", [row for role in role_rows.values() for row in role])
        _write_jsonl(run_dir / "SOURCE_EVALUATION_ROWS.jsonl", source_eval_rows["calibration"] + source_eval_rows["test"])
        _write_jsonl(run_dir / "EXTERNAL_DEVELOPMENT_ROWS.jsonl", external_rows)
        summary = {
            "schema_version": "worldsim_v7.p12_nuscenes_visibility_authority.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "supported_visibility_targeted_source_head_on_consumed_external" if all(gates.values()) else "rejected_visibility_targeted_source_head",
            "source_actor_counts": {role: len(rows) for role, rows in role_rows.items()},
            "source_safe_rates": {role: float(np.mean([bool(row["visibility_safe"]) for row in rows])) for role, rows in role_rows.items()},
            "training_history": history,
            "calibration_target_coverage": float(config["calibration"]["target_visibility_coverage"]),
            "frozen_visibility_threshold": threshold,
            "source_calibration": source_calibration,
            "source_test": source_test,
            "external_consumed_development": external,
            "fixed_gates": gates,
            "resources": {
                "gpu_used": True,
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
            "claim_boundary": config["claim_boundary"],
            "external_target_fit_or_threshold_change": False,
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(run_dir / "status.json", {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        return summary
    except Exception as exc:
        _write_json(run_dir / "status.json", {"status": "failed", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "error": repr(exc)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.repo_root, args.run_id), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

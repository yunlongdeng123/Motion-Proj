"""Fit and exactly-once evaluate the V7 P16 completion responsibility model."""

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

from motion_proj.worldsim_v7.av2_four_action_compiler import (
    COMPLETION_FEATURE_NAMES,
    compile_log,
)
from motion_proj.worldsim_v7.completion_responsibility import (
    COMPLETION_STATES,
    CompletionResponsibilityMLP,
    FeatureStandardizer,
    apply_completion_policy,
    classification_metrics,
    completion_labels,
    predict_completion,
    summarize_actor_policy,
)
from motion_proj.worldsim_v7.nuscenes_actor_surface import (
    build_selected_index,
    compile_nuscenes_scene,
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load_configs(config_path: Path, repo_root: Path):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    compiler = yaml.safe_load(
        (repo_root / str(config["p2_config"])).read_text(encoding="utf-8")
    )
    source = yaml.safe_load(
        (repo_root / str(config["p4_config"])).read_text(encoding="utf-8")
    )
    return config, compiler, source


def _candidate_arrays(
    bundles: list[dict[str, Any]],
    config: Mapping[str, Any],
    compiler: Mapping[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    for bundle in bundles:
        diagnostics = bundle["diagnostics"]
        candidate_features = np.asarray(diagnostics["completion_features"], dtype=np.float32)
        if not len(candidate_features):
            continue
        features.append(candidate_features)
        labels.append(
            completion_labels(
                diagnostics,
                config["attribution"],
                float(compiler["compiler_geometry"]["target_support_distance_m"]),
                device,
            )
        )
    if not features:
        raise RuntimeError("P16 source corpus has no completion candidates")
    return np.concatenate(features), np.concatenate(labels)


def _train_model(
    features: np.ndarray,
    labels: np.ndarray,
    config: Mapping[str, Any],
    device: torch.device,
):
    standardizer = FeatureStandardizer.fit(features)
    standardized = torch.as_tensor(
        standardizer.transform(features), dtype=torch.float32, device=device
    )
    targets = torch.as_tensor(labels, dtype=torch.long, device=device)
    model_config = config["model"]
    torch.manual_seed(int(model_config["seed"]))
    model = CompletionResponsibilityMLP(
        len(COMPLETION_FEATURE_NAMES), int(model_config["hidden_dim"])
    ).to(device)
    counts = np.bincount(labels, minlength=len(COMPLETION_STATES)).astype(np.float64)
    weights = np.sqrt(len(labels) / np.maximum(len(COMPLETION_STATES) * counts, 1.0))
    weights /= np.mean(weights)
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.as_tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    batch_size = int(model_config["batch_size"])
    history = []
    for epoch in range(int(model_config["epochs"])):
        permutation = torch.randperm(len(standardized), device=device)
        total_loss = 0.0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            loss = criterion(model(standardized[indices]), targets[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        if epoch in {0, int(model_config["epochs"]) - 1}:
            history.append({"epoch": epoch + 1, "mean_loss": total_loss / len(labels)})
    return model, standardizer, counts.astype(np.int64), weights.astype(np.float32), history


def _source_bundles(
    config: Mapping[str, Any],
    compiler: Mapping[str, Any],
    source: Mapping[str, Any],
    device: torch.device,
) -> dict[str, list[dict[str, Any]]]:
    index = build_selected_index(
        Path(str(source["nuscenes"]["dataset_root"])),
        source["nuscenes"]["role_scenes"],
        source["nuscenes"]["allowed_category_prefixes"],
    )
    output = {role: [] for role in source["nuscenes"]["role_scenes"]}
    scene_order = [
        (role, scene)
        for role, scenes in source["nuscenes"]["role_scenes"].items()
        for scene in scenes
    ]
    for position, (role, scene) in enumerate(scene_order):
        rows, diagnostics = compile_nuscenes_scene(
            scene,
            index["scenes"][scene],
            Path(str(source["nuscenes"]["dataset_root"])),
            source["nuscenes"]["actors"],
            compiler,
            device,
            include_diagnostics=True,
        )
        for row in rows:
            output[role].append(
                {"row": row, "diagnostics": diagnostics[str(row["track_id"])], "scene": scene}
            )
        print(
            json.dumps(
                {"stage": "source_corpus", "progress": f"{position + 1}/{len(scene_order)}", "role": role, "scene": scene, "actors": len(rows)}
            ),
            flush=True,
        )
    return output


def run_fit(config_path: Path, repo_root: Path, run_id: str) -> dict[str, Any]:
    config, compiler, source = _load_configs(config_path, repo_root)
    run_dir = Path(config["runs_root"]) / "worldsim_v7" / config["fit_task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "fit"})
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P16 fit requires the frozen CUDA device")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        roles = _source_bundles(config, compiler, source, device)
        fit_bundles = roles["train"] + roles["calibration"]
        fit_features, fit_labels = _candidate_arrays(fit_bundles, config, compiler, device)
        test_features, test_labels = _candidate_arrays(roles["test"], config, compiler, device)
        model, standardizer, class_counts, class_weights, history = _train_model(
            fit_features, fit_labels, config, device
        )
        test_predictions, test_probabilities = predict_completion(
            model, standardizer, test_features, device
        )
        actor_rows = []
        offset = 0
        for bundle in roles["test"]:
            count = len(bundle["diagnostics"]["completion_features"])
            actor = apply_completion_policy(
                bundle["row"],
                bundle["diagnostics"],
                test_predictions[offset : offset + count],
                test_probabilities[offset : offset + count],
                compiler,
                config["attribution"],
                device,
            )
            actor["scene_name"] = bundle["scene"]
            actor_rows.append(actor)
            offset += count
        model_payload = {
            "state_dict": model.state_dict(),
            "standardizer": standardizer.payload(),
            "feature_names": COMPLETION_FEATURE_NAMES,
            "states": COMPLETION_STATES,
            "training_dataset": "nuScenes train+calibration only",
            "seed": int(config["model"]["seed"]),
        }
        torch.save(model_payload, run_dir / "MODEL.pt")
        _write_jsonl(run_dir / "SOURCE_TEST_ACTORS.jsonl", actor_rows)
        summary = {
            "schema_version": "worldsim_v7.p16_completion_fit.v1",
            "task_id": config["fit_task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "training_dataset": "nuScenes train+calibration only",
            "actor_counts": {role: len(rows) for role, rows in roles.items()},
            "fit_candidate_count": int(len(fit_labels)),
            "fit_class_counts": {name: int(class_counts[index]) for index, name in enumerate(COMPLETION_STATES)},
            "class_weights": {name: float(class_weights[index]) for index, name in enumerate(COMPLETION_STATES)},
            "training_history": history,
            "source_test_classification": classification_metrics(test_labels, test_predictions),
            "source_test_policy": summarize_actor_policy(actor_rows),
            "target_data_read": False,
            "resources": {
                "device": str(device),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(run_dir / "status.json", {"status": "done", "phase": "fit", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        return {"run_dir": str(run_dir), "status": "done"}
    except Exception as error:
        _write_json(run_dir / "status.json", {"status": "failed", "phase": "fit", "error": f"{type(error).__name__}: {error}"})
        raise


def _load_model(fit_run: Path, config: Mapping[str, Any], device: torch.device):
    fit_status = json.loads((fit_run / "status.json").read_text(encoding="utf-8"))
    if fit_status.get("status") != "done":
        raise RuntimeError("P16 fit is not complete")
    payload = torch.load(fit_run / "MODEL.pt", map_location=device, weights_only=False)
    if tuple(payload["feature_names"]) != COMPLETION_FEATURE_NAMES:
        raise RuntimeError("P16 completion feature contract changed")
    model = CompletionResponsibilityMLP(
        len(COMPLETION_FEATURE_NAMES), int(config["model"]["hidden_dim"])
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, FeatureStandardizer.from_payload(payload["standardizer"])


def run_external(
    config_path: Path, repo_root: Path, run_id: str, fit_run: Path
) -> dict[str, Any]:
    config, compiler, _ = _load_configs(config_path, repo_root)
    run_dir = Path(config["runs_root"]) / "worldsim_v7" / config["external_task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "external"})
    device = torch.device(str(config["device"]))
    torch.cuda.reset_peak_memory_stats(device)
    model, standardizer = _load_model(fit_run, config, device)
    cohort = json.loads((repo_root / str(config["fresh_av2_cohort"])).read_text(encoding="utf-8"))
    state_root = Path(str(config["fresh_download_state"]))
    missing = [row["log_id"] for row in cohort["logs"] if not (state_root / f"{row['log_id']}.complete").is_file()]
    if missing:
        raise RuntimeError(f"P16 fresh AV2 download incomplete: {len(missing)} logs")
    started = time.monotonic()
    actor_rows = []
    try:
        for position, cohort_row in enumerate(cohort["logs"]):
            log_id = str(cohort_row["log_id"])
            compiled = compile_log(
                Path(str(compiler["dataset_root"])) / log_id,
                compiler,
                device,
                include_diagnostics=True,
            )
            for actor_row in compiled["actor_rows"]:
                diagnostics = compiled["compiled"]["diagnostics"][str(actor_row["track_id"])]
                predictions, probabilities = predict_completion(
                    model,
                    standardizer,
                    np.asarray(diagnostics["completion_features"], dtype=np.float32),
                    device,
                )
                row = apply_completion_policy(
                    actor_row,
                    diagnostics,
                    predictions,
                    probabilities,
                    compiler,
                    config["attribution"],
                    device,
                )
                row["log_id"] = log_id
                actor_rows.append(row)
            print(json.dumps({"stage": "fresh_AV2", "progress": f"{position + 1}/{len(cohort['logs'])}", "log_id": log_id, "actors": len(actor_rows)}), flush=True)
        policy = summarize_actor_policy(actor_rows)
        decisions = {
            "hazard_new_early_strictly_lower": policy["hazard"]["p16"]["new_early_rate"] < policy["hazard"]["baseline"]["new_early_rate"],
            "population_chamfer_no_worse_than_frozen_baseline": policy["p16"]["mean_chamfer_m"] <= policy["baseline"]["mean_chamfer_m"],
        }
        verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
        _write_jsonl(run_dir / "FRESH_AV2_ACTORS.jsonl", actor_rows)
        summary = {
            "schema_version": "worldsim_v7.p16_completion_external.v1",
            "task_id": config["external_task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": verdict,
            "fresh_log_count": len(cohort["logs"]),
            "failed_log_deletion": False,
            "fit_run": str(fit_run),
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
        _write_json(run_dir / "status.json", {"status": "done", "phase": "external", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        return {"run_dir": str(run_dir), "verdict": verdict, "decisions": decisions}
    except Exception as error:
        _write_json(run_dir / "status.json", {"status": "failed", "phase": "external", "error": f"{type(error).__name__}: {error}"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--phase", choices=("fit", "external"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--fit-run", type=Path)
    args = parser.parse_args()
    if args.phase == "fit":
        result = run_fit(args.config.resolve(), args.repo_root.resolve(), args.run_id)
    else:
        if args.fit_run is None:
            parser.error("--fit-run is required for external phase")
        result = run_external(
            args.config.resolve(), args.repo_root.resolve(), args.run_id, args.fit_run.resolve()
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

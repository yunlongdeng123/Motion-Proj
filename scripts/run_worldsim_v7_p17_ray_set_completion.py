"""Train and exactly-once evaluate V7 P17 joint ray-set completion."""

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
    build_ray_package,
    package_to_device,
    predict_ray_set,
    rendered_actor_loss,
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
    compiler = yaml.safe_load((repo_root / config["p2_config"]).read_text(encoding="utf-8"))
    source = yaml.safe_load((repo_root / config["p4_config"]).read_text(encoding="utf-8"))
    return config, compiler, source


def _source_bundles(
    compiler: Mapping[str, Any],
    source: Mapping[str, Any],
    device: torch.device,
) -> dict[str, list[dict[str, Any]]]:
    index = build_selected_index(
        Path(source["nuscenes"]["dataset_root"]),
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
            Path(source["nuscenes"]["dataset_root"]),
            source["nuscenes"]["actors"],
            compiler,
            device,
            include_diagnostics=True,
        )
        for row in rows:
            output[role].append(
                {"row": row, "diagnostics": diagnostics[row["track_id"]], "scene": scene}
            )
        print(json.dumps({"stage": "source_corpus", "progress": f"{position + 1}/{len(scene_order)}", "role": role, "scene": scene, "actors": len(rows)}), flush=True)
    return output


def _decisions(policy: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "hazard_new_early_strictly_lower": policy["hazard"]["p16"]["new_early_rate"] < policy["hazard"]["baseline"]["new_early_rate"],
        "population_chamfer_no_worse_than_frozen_baseline": policy["p16"]["mean_chamfer_m"] <= policy["baseline"]["mean_chamfer_m"],
    }


def _train(
    fit_bundles: list[dict[str, Any]],
    config: Mapping[str, Any],
    device: torch.device,
):
    feature_arrays = [
        np.asarray(bundle["diagnostics"]["completion_features"], dtype=np.float32)
        for bundle in fit_bundles
        if len(bundle["diagnostics"]["completion_features"])
    ]
    standardizer = FeatureStandardizer.fit(np.concatenate(feature_arrays))
    packages = []
    for bundle in fit_bundles:
        package = build_ray_package(
            bundle["diagnostics"],
            config["attribution"],
            int(config["model"]["maximum_training_rays_per_actor"]),
            device,
        )
        if package is not None:
            packages.append(package_to_device(package, standardizer, device))
    if not packages:
        raise RuntimeError("P17 has no influential source ray sets")
    torch.manual_seed(int(config["model"]["seed"]))
    model = RaySetCompletionMLP(int(config["model"]["hidden_dim"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["model"]["learning_rate"]),
        weight_decay=float(config["model"]["weight_decay"]),
    )
    threshold = float(config["model"]["forward_selection_threshold"])
    actor_batch = int(config["model"]["actor_batch_size"])
    history = []
    for epoch in range(int(config["model"]["epochs"])):
        permutation = torch.randperm(len(packages)).cpu().tolist()
        total_loss = 0.0
        for start in range(0, len(permutation), actor_batch):
            batch = permutation[start : start + actor_batch]
            loss = torch.stack(
                [rendered_actor_loss(model, packages[index], threshold) for index in batch]
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
        if epoch in {0, int(config["model"]["epochs"]) - 1}:
            history.append({"epoch": epoch + 1, "mean_actor_loss": total_loss / len(packages)})
    return model, standardizer, len(packages), history


def _evaluate_bundles(
    bundles: list[dict[str, Any]],
    model: RaySetCompletionMLP,
    standardizer: FeatureStandardizer,
    config: Mapping[str, Any],
    compiler: Mapping[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    rows = []
    threshold = float(config["model"]["forward_selection_threshold"])
    for bundle in bundles:
        diagnostics = bundle["diagnostics"]
        states, probabilities, occupancy = predict_ray_set(
            model,
            standardizer,
            np.asarray(diagnostics["completion_features"], dtype=np.float32),
            threshold,
            device,
        )
        row = apply_completion_policy(
            bundle["row"], diagnostics, states, probabilities, compiler, config["attribution"], device
        )
        row["mean_occupancy_score"] = float(np.mean(occupancy)) if len(occupancy) else 0.0
        if "scene" in bundle:
            row["scene_name"] = bundle["scene"]
        if "log_id" in bundle:
            row["log_id"] = bundle["log_id"]
        rows.append(row)
    return rows


def run_fit(config_path: Path, repo_root: Path, run_id: str) -> dict[str, Any]:
    config, compiler, source = _load_configs(config_path, repo_root)
    run_dir = Path(config["runs_root"]) / "worldsim_v7" / config["fit_task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "fit"})
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P17 fit requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        roles = _source_bundles(compiler, source, device)
        model, standardizer, package_count, history = _train(
            roles["train"] + roles["calibration"], config, device
        )
        test_rows = _evaluate_bundles(
            roles["test"], model, standardizer, config, compiler, device
        )
        policy = summarize_actor_policy(test_rows)
        decisions = _decisions(policy)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "standardizer": standardizer.payload(),
                "feature_names": COMPLETION_FEATURE_NAMES,
                "seed": int(config["model"]["seed"]),
                "source_decisions": decisions,
            },
            run_dir / "MODEL.pt",
        )
        _write_jsonl(run_dir / "SOURCE_TEST_ACTORS.jsonl", test_rows)
        summary = {
            "schema_version": "worldsim_v7.p17_ray_set_fit.v1",
            "task_id": config["fit_task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "source_pass_external_authorized" if all(decisions.values()) else "source_rejected_external_unread",
            "actor_counts": {role: len(rows) for role, rows in roles.items()},
            "training_ray_set_actor_count": package_count,
            "training_history": history,
            "source_test_policy": policy,
            "source_decisions": decisions,
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
        return {"run_dir": str(run_dir), "verdict": summary["verdict"], "decisions": decisions}
    except Exception as error:
        _write_json(run_dir / "status.json", {"status": "failed", "phase": "fit", "error": f"{type(error).__name__}: {error}"})
        raise


def _load_model(fit_run: Path, config: Mapping[str, Any], device: torch.device):
    summary = json.loads((fit_run / "summary.json").read_text(encoding="utf-8"))
    if summary.get("verdict") != "source_pass_external_authorized":
        raise RuntimeError("P17 source did not authorize external read")
    payload = torch.load(fit_run / "MODEL.pt", map_location=device, weights_only=False)
    if tuple(payload["feature_names"]) != COMPLETION_FEATURE_NAMES:
        raise RuntimeError("P17 feature contract changed")
    model = RaySetCompletionMLP(int(config["model"]["hidden_dim"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, FeatureStandardizer.from_payload(payload["standardizer"])


def run_external(config_path: Path, repo_root: Path, run_id: str, fit_run: Path) -> dict[str, Any]:
    config, compiler, _ = _load_configs(config_path, repo_root)
    run_dir = Path(config["runs_root"]) / "worldsim_v7" / config["external_task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "external"})
    device = torch.device(config["device"])
    torch.cuda.reset_peak_memory_stats(device)
    model, standardizer = _load_model(fit_run, config, device)
    cohort = json.loads((repo_root / config["fresh_av2_cohort"]).read_text(encoding="utf-8"))
    state_root = Path(config["fresh_download_state"])
    missing = [row["log_id"] for row in cohort["logs"] if not (state_root / f"{row['log_id']}.complete").is_file()]
    if missing:
        raise RuntimeError(f"P17 fresh download incomplete: {len(missing)} logs")
    started = time.monotonic()
    bundles = []
    try:
        for position, cohort_row in enumerate(cohort["logs"]):
            log_id = cohort_row["log_id"]
            compiled = compile_log(Path(compiler["dataset_root"]) / log_id, compiler, device, include_diagnostics=True)
            for row in compiled["actor_rows"]:
                bundles.append({"row": row, "diagnostics": compiled["compiled"]["diagnostics"][row["track_id"]], "log_id": log_id})
            print(json.dumps({"stage": "fresh_AV2", "progress": f"{position + 1}/{len(cohort['logs'])}", "log_id": log_id}), flush=True)
        rows = _evaluate_bundles(bundles, model, standardizer, config, compiler, device)
        policy = summarize_actor_policy(rows)
        decisions = _decisions(policy)
        verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
        _write_jsonl(run_dir / "FRESH_AV2_ACTORS.jsonl", rows)
        summary = {
            "schema_version": "worldsim_v7.p17_ray_set_external.v1",
            "task_id": config["external_task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": verdict,
            "fit_run": str(fit_run),
            "fresh_log_count": len(cohort["logs"]),
            "failed_log_deletion": False,
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
            parser.error("--fit-run is required for external")
        result = run_external(args.config.resolve(), args.repo_root.resolve(), args.run_id, args.fit_run.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

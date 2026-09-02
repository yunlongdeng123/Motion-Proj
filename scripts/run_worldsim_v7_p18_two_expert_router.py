"""Train and exactly-once evaluate the V7 P18 frozen-expert router."""

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
from motion_proj.worldsim_v7.completion_expert_router import (
    EXPERT_ACTIONS,
    ROUTER_FEATURE_NAMES,
    TwoExpertRouter,
    p17r_dominance_label,
    predict_actions,
    route_actor_row,
    router_features,
    routing_metrics,
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
    predict_ray_set,
)


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
        print(
            json.dumps(
                {
                    "stage": "source_corpus",
                    "progress": f"{position + 1}/{len(scene_order)}",
                    "role": role,
                    "scene": scene,
                    "actors": len(rows),
                }
            ),
            flush=True,
        )
    return output


def _load_p17r_expert(
    fit_run: Path,
    expert_config: Mapping[str, Any],
    device: torch.device,
):
    payload = torch.load(fit_run / "MODEL.pt", map_location=device, weights_only=False)
    if tuple(payload["feature_names"]) != COMPLETION_FEATURE_NAMES:
        raise RuntimeError("P17R feature contract changed")
    model = RaySetCompletionMLP(int(expert_config["model"]["hidden_dim"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, FeatureStandardizer.from_payload(payload["standardizer"])


def _evaluate_p17r(
    bundles: list[dict[str, Any]],
    model: RaySetCompletionMLP,
    standardizer: FeatureStandardizer,
    expert_config: Mapping[str, Any],
    compiler: Mapping[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    records = []
    threshold = float(expert_config["model"]["forward_selection_threshold"])
    for bundle in bundles:
        diagnostics = bundle["diagnostics"]
        states, probabilities, occupancy = predict_ray_set(
            model,
            standardizer,
            np.asarray(diagnostics["completion_features"], dtype=np.float32),
            threshold,
            device,
        )
        p17r_row = apply_completion_policy(
            bundle["row"],
            diagnostics,
            states,
            probabilities,
            compiler,
            expert_config["attribution"],
            device,
        )
        if "scene" in bundle:
            p17r_row["scene_name"] = bundle["scene"]
        if "log_id" in bundle:
            p17r_row["log_id"] = bundle["log_id"]
        records.append(
            {
                "actor_row": bundle["row"],
                "p17r_row": p17r_row,
                "features": router_features(bundle["row"], occupancy, threshold),
                "label": p17r_dominance_label(p17r_row),
            }
        )
    return records


def _feature_array(records: list[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([record["features"] for record in records], dtype=np.float32)


def _label_array(records: list[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([record["label"] for record in records], dtype=np.int64)


def _train_router(
    records: list[dict[str, Any]],
    config: Mapping[str, Any],
    device: torch.device,
):
    values = _feature_array(records)
    labels_np = _label_array(records)
    standardizer = FeatureStandardizer.fit(values)
    features = torch.as_tensor(
        standardizer.transform(values), dtype=torch.float32, device=device
    )
    labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)
    counts = np.bincount(labels_np, minlength=len(EXPERT_ACTIONS)).astype(np.float64)
    class_weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    class_weights /= np.mean(class_weights)
    loss_function = torch.nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    )
    router_config = config["router"]
    seed = int(router_config["seed"])
    torch.manual_seed(seed)
    model = TwoExpertRouter(int(router_config["hidden_dim"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(router_config["learning_rate"]),
        weight_decay=float(router_config["weight_decay"]),
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch_size = int(router_config["batch_size"])
    history = []
    model.train()
    for epoch in range(int(router_config["epochs"])):
        permutation = torch.randperm(len(features), generator=generator).to(device)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(permutation), batch_size):
            index = permutation[start : start + batch_size]
            loss = loss_function(model(features[index]), labels[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            batches += 1
        if epoch in {0, int(router_config["epochs"]) - 1}:
            history.append({"epoch": epoch + 1, "mean_batch_loss": epoch_loss / batches})
    model.eval()
    return model, standardizer, history, class_weights.tolist()


def _route_records(
    records: list[dict[str, Any]],
    model: TwoExpertRouter,
    standardizer: FeatureStandardizer,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = _label_array(records)
    actions, probabilities = predict_actions(
        model, standardizer, _feature_array(records), device
    )
    rows = [
        route_actor_row(
            record["actor_row"], record["p17r_row"], int(action), probability
        )
        for record, action, probability in zip(records, actions, probabilities)
    ]
    return rows, routing_metrics(labels, actions)


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


def run_fit(config_path: Path, repo_root: Path, run_id: str) -> dict[str, Any]:
    config, compiler, source, expert_config = _load_configs(config_path, repo_root)
    run_dir = Path(config["runs_root"]) / "worldsim_v7" / config["fit_task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "fit"})
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P18 fit requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        p17r_model, p17r_standardizer = _load_p17r_expert(
            Path(config["base_expert_fit_run"]), expert_config, device
        )
        roles = _source_bundles(compiler, source, device)
        expert_records = {
            role: _evaluate_p17r(
                bundles,
                p17r_model,
                p17r_standardizer,
                expert_config,
                compiler,
                device,
            )
            for role, bundles in roles.items()
        }
        fit_records = expert_records["train"] + expert_records["calibration"]
        model, standardizer, history, class_weights = _train_router(
            fit_records, config, device
        )
        split_routing = {}
        routed = {}
        for role, records in expert_records.items():
            routed[role], split_routing[role] = _route_records(
                records, model, standardizer, device
            )
        policy = summarize_actor_policy(routed["test"])
        decisions = _decisions(policy)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "standardizer": standardizer.payload(),
                "feature_names": ROUTER_FEATURE_NAMES,
                "expert_actions": EXPERT_ACTIONS,
                "seed": int(config["router"]["seed"]),
                "p17r_fit_run": config["base_expert_fit_run"],
                "source_decisions": decisions,
            },
            run_dir / "MODEL.pt",
        )
        _write_jsonl(run_dir / "SOURCE_TEST_ACTORS.jsonl", routed["test"])
        summary = {
            "schema_version": "worldsim_v7.p18_two_expert_router_fit.v1",
            "task_id": config["fit_task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": (
                "source_development_pass_external_authorized"
                if all(decisions.values())
                else "source_development_rejected_external_unread"
            ),
            "source_evidence_status": "consumed_development_only",
            "actor_counts": {role: len(rows) for role, rows in roles.items()},
            "training_history": history,
            "training_class_weights": class_weights,
            "routing_metrics": split_routing,
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
        _write_json(
            run_dir / "status.json",
            {
                "status": "done",
                "phase": "fit",
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
                "phase": "fit",
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


def _load_router(
    fit_run: Path,
    config: Mapping[str, Any],
    device: torch.device,
):
    summary = json.loads((fit_run / "summary.json").read_text(encoding="utf-8"))
    if summary.get("verdict") != "source_development_pass_external_authorized":
        raise RuntimeError("P18 source development did not authorize external read")
    payload = torch.load(fit_run / "MODEL.pt", map_location=device, weights_only=False)
    if tuple(payload["feature_names"]) != ROUTER_FEATURE_NAMES:
        raise RuntimeError("P18 router feature contract changed")
    if tuple(payload["expert_actions"]) != EXPERT_ACTIONS:
        raise RuntimeError("P18 expert action contract changed")
    model = TwoExpertRouter(int(config["router"]["hidden_dim"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, FeatureStandardizer.from_payload(payload["standardizer"])


def run_external(
    config_path: Path,
    repo_root: Path,
    run_id: str,
    fit_run: Path,
) -> dict[str, Any]:
    config, compiler, _, expert_config = _load_configs(config_path, repo_root)
    run_dir = Path(config["runs_root"]) / "worldsim_v7" / config["external_task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "external"})
    device = torch.device(config["device"])
    torch.cuda.reset_peak_memory_stats(device)
    model, standardizer = _load_router(fit_run, config, device)
    p17r_model, p17r_standardizer = _load_p17r_expert(
        Path(config["base_expert_fit_run"]), expert_config, device
    )
    cohort = json.loads((repo_root / config["fresh_av2_cohort"]).read_text(encoding="utf-8"))
    state_root = Path(config["fresh_download_state"])
    missing = [
        row["log_id"]
        for row in cohort["logs"]
        if not (state_root / f"{row['log_id']}.complete").is_file()
    ]
    if missing:
        raise RuntimeError(f"P18 fresh download incomplete: {len(missing)} logs")
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
            records = _evaluate_p17r(
                bundles,
                p17r_model,
                p17r_standardizer,
                expert_config,
                compiler,
                device,
            )
            log_rows, _ = _route_records(records, model, standardizer, device)
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
            "schema_version": "worldsim_v7.p18_two_expert_router_external.v1",
            "task_id": config["external_task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": verdict,
            "fit_run": str(fit_run),
            "fresh_log_count": len(cohort["logs"]),
            "actor_count": len(rows),
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
    parser.add_argument("--phase", choices=("fit", "external"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--fit-run", type=Path)
    args = parser.parse_args()
    if args.phase == "fit":
        result = run_fit(args.config.resolve(), args.repo_root.resolve(), args.run_id)
    else:
        if args.fit_run is None:
            parser.error("--fit-run is required for external")
        result = run_external(
            args.config.resolve(), args.repo_root.resolve(), args.run_id, args.fit_run.resolve()
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

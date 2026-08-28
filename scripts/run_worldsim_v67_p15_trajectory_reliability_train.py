"""Train a trajectory-conditioned future visited-state reliability model."""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from motion_proj.worldsim_v67.trajectory_reliability import (
    FEATURE_NAMES,
    score_lattice_adapter,
    score_head,
    train_lattice_adapter,
    train_head,
)
from scripts.run_worldsim_v65_p10v_action_visited_state_transfer import (
    _pairwise_concordance,
    _within_case_selection,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _metrics(
    arrays: dict[str, np.ndarray], scores: np.ndarray, config: dict
) -> dict[str, object]:
    target = np.asarray(arrays["target_cost"], dtype=np.float32)
    unsafe = np.asarray(arrays["unsafe"], dtype=bool)
    cases = np.asarray(arrays["case_index"])
    scenes = np.asarray(arrays["scene_index"])
    return {
        "spearman": float(spearmanr(target, scores).statistic),
        "unsafe_auroc": float(roc_auc_score(unsafe, scores)),
        "unsafe_auprc": float(average_precision_score(unsafe, scores)),
        "mse": float(np.mean((scores - target) ** 2)),
        "mae": float(np.mean(np.abs(scores - target))),
        "pairwise": _pairwise_concordance(
            target,
            scores,
            cases,
            float(config["evaluation"]["pairwise_minimum_target_gap"]),
        ),
        "within_case_selection": _within_case_selection(
            target,
            scores,
            cases,
            scenes,
            float(config["evaluation"]["within_case_selected_fraction"]),
        ),
    }


def run(config_path: Path, runs_root: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v67" / str(config["task_id"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(
        run_dir / "status.json",
        {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    train = _load(Path(config["inputs"]["train_compact_cache"]))
    selection = _load(Path(config["inputs"]["selection_compact_cache"]))
    if str(config["model"].get("type", "residual_mlp")) == "lattice_residual_adapter":
        model, training = train_lattice_adapter(
            train, config["model"], int(config["seed"])
        )
        mean = scale = None
        train_scores = score_lattice_adapter(model, train)
        selection_scores = score_lattice_adapter(model, selection)
    else:
        model, mean, scale, training = train_head(
            train, config["model"], int(config["seed"])
        )
        train_scores = score_head(model, train, mean, scale)
        selection_scores = score_head(model, selection, mean, scale)
    train_metrics = _metrics(train, train_scores, config)
    selection_metrics = _metrics(selection, selection_scores, config)
    qmean_metrics = _metrics(
        selection, np.asarray(selection["qmean"], dtype=np.float32), config
    )
    improvement = {
        "selected_cost_reduction_delta_over_qmean": float(
            selection_metrics["within_case_selection"]["relative_cost_reduction"]
            - qmean_metrics["within_case_selection"]["relative_cost_reduction"]
        ),
        "spearman_delta_over_qmean": float(
            selection_metrics["spearman"] - qmean_metrics["spearman"]
        ),
        "pairwise_delta_over_qmean": float(
            selection_metrics["pairwise"]["concordance"]
            - qmean_metrics["pairwise"]["concordance"]
        ),
    }
    gate_config = config["gates"]
    gates = {
        "minimum_selection_spearman": selection_metrics["spearman"]
        >= float(gate_config["minimum_selection_spearman"]),
        "minimum_selection_unsafe_auroc": selection_metrics["unsafe_auroc"]
        >= float(gate_config["minimum_selection_unsafe_auroc"]),
        "minimum_selection_pairwise_concordance": selection_metrics["pairwise"]["concordance"]
        >= float(gate_config["minimum_selection_pairwise_concordance"]),
        "minimum_selected_cost_reduction": selection_metrics["within_case_selection"][
            "relative_cost_reduction"
        ]
        >= float(gate_config["minimum_selected_cost_reduction"]),
        "minimum_selected_cost_reduction_delta_over_qmean": improvement[
            "selected_cost_reduction_delta_over_qmean"
        ]
        >= float(gate_config["minimum_selected_cost_reduction_delta_over_qmean"]),
        "minimum_scene_support": selection_metrics["within_case_selection"][
            "scene_nonincreasing_count"
        ]
        >= int(gate_config["minimum_scene_support"]),
    }
    verdict = (
        "supported_trajectory_conditioned_visited_state_reliability_selection"
        if all(gates.values())
        else "rejected_trajectory_conditioned_visited_state_reliability_selection"
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_type": str(config["model"].get("type", "residual_mlp")),
            "feature_names": list(FEATURE_NAMES)
            if str(config["model"].get("type", "residual_mlp")) == "residual_mlp"
            else ["qmean", "action_index", "case_index"],
            "hidden_dimensions": list(config["model"].get("hidden_dimensions", [])),
            "mean": mean,
            "scale": scale,
        },
        run_dir / "TRAJECTORY_RELIABILITY_HEAD.pt",
    )
    (run_dir / "SELECTION_ACTION_SCORES.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "case_index": int(selection["case_index"][index]),
                    "scene_index": int(selection["scene_index"][index]),
                    "action_index": int(selection["action_index"][index]),
                    "progress_ratio": float(selection["progress_ratio"][index]),
                    "lateral_offset_m": float(selection["lateral_offset_m"][index]),
                    "qmean": float(selection["qmean"][index]),
                    "predicted_visited_state_cost": float(selection_scores[index]),
                    "target_cost": float(selection["target_cost"][index]),
                    "unsafe": bool(selection["unsafe"][index]),
                },
                sort_keys=True,
            )
            + "\n"
            for index in range(len(selection_scores))
        ),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "worldsim_v67.p15_trajectory_reliability_summary.v1",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "role": config["role"],
        "claim_boundary": config["claim_boundary"],
        "training": training,
        "train_metrics": train_metrics,
        "selection_metrics": selection_metrics,
        "selection_qmean_baseline": qmean_metrics,
        "selection_improvement": improvement,
        "gate_results": gates,
        "failure_ledger_delta": "pending_result",
        "resources": {
            "gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
            "wall_seconds": time.monotonic() - started,
        },
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(
        run_dir / "status.json",
        {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    return {"run_dir": str(run_dir), "verdict": verdict, "gate_results": gates}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.runs_root.resolve(), args.run_id), indent=2))


if __name__ == "__main__":
    main()

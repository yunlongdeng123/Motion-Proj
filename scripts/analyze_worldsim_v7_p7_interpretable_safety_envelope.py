"""Explain the frozen V7 P4 selector with risk--coverage and input attributions."""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.stats import ks_2samp, wasserstein_distance

from motion_proj.worldsim_v7.selective_validity_hazard import (
    FactorizedTwoHead,
    HAZARD_FEATURE_NAMES,
    SharedTwoHead,
    Standardizer,
    VALIDITY_FEATURE_NAMES,
    predict,
    rows_to_arrays,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


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


def _standardize(
    arrays: Mapping[str, np.ndarray],
    validity: Standardizer,
    hazard: Standardizer,
) -> dict[str, np.ndarray]:
    return {
        **arrays,
        "validity": validity.transform(arrays["validity"]),
        "hazard": hazard.transform(arrays["hazard"]),
    }


def _curve(
    arrays: Mapping[str, np.ndarray],
    scores: np.ndarray,
    coverage_grid: list[float],
    dataset: str,
    model: str,
) -> list[dict[str, Any]]:
    order = np.argsort(-scores, kind="stable")
    failures = 1.0 - arrays["repairable"]
    hazardous = arrays["hazardous"].astype(bool)
    query = arrays["query_chamfer"]
    compiled = arrays["compiled_chamfer"]
    rows = []
    for requested in coverage_grid:
        count = int(round(float(requested) * len(scores)))
        selected = np.zeros(len(scores), dtype=bool)
        selected[order[:count]] = True
        selective = np.where(selected, compiled, query)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "requested_coverage": float(requested),
                "coverage": float(np.mean(selected)),
                "selected_actor_count": int(np.sum(selected)),
                "population_false_repair_rate": float(np.mean(selected * failures)),
                "conditional_selected_failure_rate": float(
                    np.mean(failures[selected]) if np.any(selected) else 0.0
                ),
                "hazard_actor_coverage": float(
                    np.mean(selected[hazardous]) if np.any(hazardous) else 0.0
                ),
                "mean_selective_chamfer_m": float(np.mean(selective)),
                "mean_selected_gain_m": float(
                    np.mean(query[selected] - compiled[selected]) if np.any(selected) else 0.0
                ),
            }
        )
    return rows


def _operating_point(
    arrays: Mapping[str, np.ndarray], scores: np.ndarray, threshold: float
) -> dict[str, float | int]:
    selected = scores >= float(threshold)
    failures = 1.0 - arrays["repairable"]
    hazardous = arrays["hazardous"].astype(bool)
    query = arrays["query_chamfer"]
    compiled = arrays["compiled_chamfer"]
    selective = np.where(selected, compiled, query)
    coverage = float(np.mean(selected))
    conditional = float(np.mean(failures[selected]) if np.any(selected) else 0.0)
    return {
        "threshold": float(threshold),
        "selected_actor_count": int(np.sum(selected)),
        "coverage": coverage,
        "population_false_repair_rate": float(np.mean(selected * failures)),
        "conditional_selected_failure_rate": conditional,
        "hazard_actor_coverage": float(
            np.mean(selected[hazardous]) if np.any(hazardous) else 0.0
        ),
        "mean_query_chamfer_m": float(np.mean(query)),
        "mean_always_repair_chamfer_m": float(np.mean(compiled)),
        "mean_selective_chamfer_m": float(np.mean(selective)),
        "risk_factorization_residual": float(
            np.mean(selected * failures) - coverage * conditional
        ),
        "geometry_factorization_residual_m": float(
            (np.mean(selective) - np.mean(query))
            - coverage * (np.mean(compiled[selected] - query[selected]) if np.any(selected) else 0.0)
        ),
    }


def _integrated_gradients(
    model: FactorizedTwoHead,
    arrays: Mapping[str, np.ndarray],
    device: torch.device,
    steps: int,
) -> dict[str, Any]:
    validity = torch.as_tensor(arrays["validity"], dtype=torch.float32, device=device)
    hazard = torch.as_tensor(arrays["hazard"], dtype=torch.float32, device=device)
    baseline = torch.zeros_like(validity)
    gradient_sum = torch.zeros_like(validity)
    model.eval()
    for alpha in torch.linspace(0.0, 1.0, steps=steps, device=device):
        point = (baseline + alpha * (validity - baseline)).detach().requires_grad_(True)
        repair_logit, _ = model(point, hazard)
        probability = torch.sigmoid(repair_logit)
        gradient = torch.autograd.grad(probability.sum(), point)[0]
        gradient_sum += gradient.detach()
    attribution = (validity - baseline) * gradient_sum / float(steps)
    with torch.inference_mode():
        input_probability = torch.sigmoid(model(validity, hazard)[0])
        baseline_probability = torch.sigmoid(model(baseline, hazard)[0])
    completeness = attribution.sum(dim=1) - (input_probability - baseline_probability)
    absolute = attribution.abs().mean(dim=0).detach().cpu().numpy()
    normalized = absolute / max(float(np.sum(absolute)), 1e-12)
    return {
        "baseline": "nuScenes train standardized mean",
        "steps": int(steps),
        "mean_absolute_completeness_residual": float(
            completeness.abs().mean().detach().cpu()
        ),
        "features": [
            {
                "name": name,
                "mean_absolute_attribution": float(value),
                "normalized_absolute_attribution": float(weight),
            }
            for name, value, weight in zip(VALIDITY_FEATURE_NAMES, absolute, normalized)
        ],
    }


def _score_distribution(scores: np.ndarray) -> dict[str, Any]:
    return {
        "mean": float(np.mean(scores)),
        "standard_deviation": float(np.std(scores)),
        "quantiles_05_25_50_75_95": [
            float(value) for value in np.quantile(scores, [0.05, 0.25, 0.50, 0.75, 0.95])
        ],
    }


def _plot(
    path: Path,
    curves: list[dict[str, Any]],
    operating: Mapping[str, Mapping[str, Mapping[str, float]]],
    attributions: Mapping[str, Mapping[str, Any]],
    top_features: int,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.1))
    colors = {"nuScenes test": "#2563eb", "AV2 zero-shot": "#dc2626"}
    styles = {"shared": "--", "factorized": "-"}
    for dataset in ("nuScenes test", "AV2 zero-shot"):
        for model in ("shared", "factorized"):
            subset = [row for row in curves if row["dataset"] == dataset and row["model"] == model]
            coverage = [100.0 * row["coverage"] for row in subset]
            false_repair = [100.0 * row["population_false_repair_rate"] for row in subset]
            chamfer = [row["mean_selective_chamfer_m"] for row in subset]
            label = f"{dataset} / {model}"
            axes[0].plot(coverage, false_repair, styles[model], color=colors[dataset], label=label)
            axes[1].plot(coverage, chamfer, styles[model], color=colors[dataset], label=label)
            point = operating[dataset][model]
            axes[0].scatter(
                100.0 * point["coverage"],
                100.0 * point["population_false_repair_rate"],
                color=colors[dataset], marker="*", s=85, zorder=4,
            )
            axes[1].scatter(
                100.0 * point["coverage"],
                point["mean_selective_chamfer_m"],
                color=colors[dataset], marker="*", s=85, zorder=4,
            )
    axes[0].set(xlabel="Coverage (%)", ylabel="Population false repair (%)", title="Risk--coverage envelope")
    axes[1].set(xlabel="Coverage (%)", ylabel="Selective Chamfer (m)", title="Geometry--coverage envelope")
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    axes[0].legend(fontsize=7, frameon=False)

    nu = {row["name"]: row["normalized_absolute_attribution"] for row in attributions["nuScenes test"]["features"]}
    av2 = {row["name"]: row["normalized_absolute_attribution"] for row in attributions["AV2 zero-shot"]["features"]}
    names = sorted(av2, key=av2.get, reverse=True)[:top_features][::-1]
    positions = np.arange(len(names))
    axes[2].barh(positions - 0.18, [nu[name] for name in names], height=0.34, color=colors["nuScenes test"], label="nuScenes test")
    axes[2].barh(positions + 0.18, [av2[name] for name in names], height=0.34, color=colors["AV2 zero-shot"], label="AV2 zero-shot")
    axes[2].set_yticks(positions, [name.replace("_", " ") for name in names], fontsize=7)
    axes[2].set(xlabel="Normalized |integrated gradient|", title="Validity-head sensitivity")
    axes[2].grid(axis="x", alpha=0.25)
    axes[2].legend(fontsize=7, frameon=False)
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = Path(str(config["source_p4_run"]))
    run_dir = Path(str(config["runs_root"])) / "worldsim_v7" / str(config["task_id"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    started = time.monotonic()
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P7 interpretability package is frozen to CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    try:
        artifact = torch.load(source / "MODEL.pt", map_location=device, weights_only=False)
        validity_standardizer = Standardizer(
            mean=np.asarray(artifact["validity_standardizer"]["mean"], dtype=np.float32),
            scale=np.asarray(artifact["validity_standardizer"]["scale"], dtype=np.float32),
        )
        hazard_standardizer = Standardizer(
            mean=np.asarray(artifact["hazard_standardizer"]["mean"], dtype=np.float32),
            scale=np.asarray(artifact["hazard_standardizer"]["scale"], dtype=np.float32),
        )
        shared = SharedTwoHead(len(VALIDITY_FEATURE_NAMES), len(HAZARD_FEATURE_NAMES), 32).to(device)
        factorized = FactorizedTwoHead(len(VALIDITY_FEATURE_NAMES), len(HAZARD_FEATURE_NAMES), 32).to(device)
        shared.load_state_dict(artifact["shared_state"])
        factorized.load_state_dict(artifact["factorized_state"])
        files = {
            "nuScenes calibration": source / "NUSCENES_CALIBRATION_ACTORS.jsonl",
            "nuScenes test": source / "NUSCENES_TEST_ACTORS.jsonl",
            "AV2 zero-shot": source / "AV2_ZERO_SHOT_ACTORS.jsonl",
        }
        arrays = {
            name: _standardize(rows_to_arrays(_read_jsonl(path)), validity_standardizer, hazard_standardizer)
            for name, path in files.items()
        }
        scores = {
            name: {
                "shared": predict(shared, values, device)[0],
                "factorized": predict(factorized, values, device)[0],
            }
            for name, values in arrays.items()
        }
        coverage_grid = [float(value) for value in config["coverage_grid"]]
        curves = []
        for dataset in ("nuScenes test", "AV2 zero-shot"):
            for model in ("shared", "factorized"):
                curves.extend(_curve(arrays[dataset], scores[dataset][model], coverage_grid, dataset, model))
        thresholds = {key: float(value) for key, value in artifact["thresholds"].items()}
        operating = {
            dataset: {
                model: _operating_point(arrays[dataset], scores[dataset][model], thresholds[model])
                for model in ("shared", "factorized")
            }
            for dataset in ("nuScenes calibration", "nuScenes test", "AV2 zero-shot")
        }
        attributions = {
            dataset: _integrated_gradients(
                factorized,
                arrays[dataset],
                device,
                int(config["integrated_gradients_steps"]),
            )
            for dataset in ("nuScenes test", "AV2 zero-shot")
        }
        distributions = {
            dataset: _score_distribution(scores[dataset]["factorized"])
            for dataset in scores
        }
        calibration_scores = scores["nuScenes calibration"]["factorized"]
        shift = {}
        for dataset in ("nuScenes test", "AV2 zero-shot"):
            comparison = scores[dataset]["factorized"]
            ks = ks_2samp(calibration_scores, comparison)
            shift[dataset] = {
                "wasserstein_score_distance": float(wasserstein_distance(calibration_scores, comparison)),
                "ks_score_statistic": float(ks.statistic),
                "ks_pvalue_descriptive_only": float(ks.pvalue),
            }
        _write_jsonl(run_dir / "RISK_COVERAGE_CURVES.jsonl", curves)
        _write_json(run_dir / "INTEGRATED_GRADIENTS.json", attributions)
        _plot(
            run_dir / "INTERPRETABLE_SAFETY_ENVELOPE",
            curves,
            operating,
            attributions,
            int(config["attribution_top_features"]),
        )
        summary = {
            "schema_version": "worldsim_v7.p7_interpretable_safety_envelope.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "conclusion": "frozen_descriptive_interpretable_safety_envelope",
            "source_p4_run": str(source),
            "claim_boundary": config["claim_boundary"],
            "actor_counts": {name: int(len(values["repairable"])) for name, values in arrays.items()},
            "operating_points": operating,
            "factorized_score_distributions": distributions,
            "factorized_score_shift_from_calibration": shift,
            "integrated_gradients": attributions,
            "structural_statements": {
                "factorized_repair_score_reads_hazard_features": False,
                "factorized_hazard_score_reads_validity_features": False,
                "cross_input_derivative": 0.0,
                "external_formal_risk_guarantee": False,
            },
            "resources": {
                "gpu_used": True,
                "device": str(device),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(run_dir / "status.json", {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        return {"run_dir": str(run_dir), "conclusion": summary["conclusion"]}
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {"status": "failed", "failed_at_utc": datetime.now(timezone.utc).isoformat(), "error": f"{type(error).__name__}: {error}"},
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.run_id), indent=2))


if __name__ == "__main__":
    main()

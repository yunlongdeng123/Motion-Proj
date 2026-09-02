"""Certify frozen P4 validity decisions under interpretable feature boxes."""

from __future__ import annotations

import argparse
import json
import math
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
from torch import nn

from motion_proj.worldsim_v7.selective_validity_hazard import (
    FactorizedTwoHead,
    HAZARD_FEATURE_NAMES,
    VALIDITY_FEATURE_NAMES,
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


def _linear_interval(layer: nn.Linear, lower: torch.Tensor, upper: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight = layer.weight
    positive = torch.clamp(weight, min=0.0)
    negative = torch.clamp(weight, max=0.0)
    low = lower @ positive.T + upper @ negative.T
    high = upper @ positive.T + lower @ negative.T
    if layer.bias is not None:
        low = low + layer.bias
        high = high + layer.bias
    return low, high


def _interval_forward(network: nn.Sequential, lower: torch.Tensor, upper: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    for layer in network:
        if isinstance(layer, nn.Linear):
            lower, upper = _linear_interval(layer, lower, upper)
        elif isinstance(layer, nn.ReLU):
            lower, upper = torch.relu(lower), torch.relu(upper)
        else:
            raise TypeError(f"unsupported interval layer: {type(layer).__name__}")
    return lower.squeeze(-1), upper.squeeze(-1)


def _state(lower_probability: np.ndarray, upper_probability: np.ndarray, threshold: float) -> np.ndarray:
    values = np.full(len(lower_probability), "interval_unresolved", dtype=object)
    values[lower_probability >= threshold] = "robust_select"
    values[upper_probability < threshold] = "robust_abstain"
    return values


def _summarize(
    states: np.ndarray,
    nominal_selected: np.ndarray,
    repairable: np.ndarray,
    hazardous: np.ndarray,
    lower_logit: np.ndarray,
    upper_logit: np.ndarray,
) -> dict[str, Any]:
    robust_select = states == "robust_select"
    robust_abstain = states == "robust_abstain"
    unresolved = states == "interval_unresolved"
    false_repair = ~repairable.astype(bool)
    widths = upper_logit - lower_logit
    selected_false = int(np.count_nonzero(robust_select & false_repair))
    selected_count = int(np.count_nonzero(robust_select))
    return {
        "actor_count": int(len(states)),
        "nominal_selected_count": int(np.count_nonzero(nominal_selected)),
        "nominal_coverage": float(np.mean(nominal_selected)),
        "robust_select_count": selected_count,
        "robust_select_coverage": float(np.mean(robust_select)),
        "robust_abstain_count": int(np.count_nonzero(robust_abstain)),
        "interval_unresolved_count": int(np.count_nonzero(unresolved)),
        "interval_unresolved_fraction": float(np.mean(unresolved)),
        "nominal_selected_certified_fraction": float(
            selected_count / max(int(np.count_nonzero(nominal_selected)), 1)
        ),
        "robust_selected_false_repair_count": selected_false,
        "robust_selected_false_repair_rate": float(selected_false / max(selected_count, 1)),
        "robust_selected_hazard_count": int(np.count_nonzero(robust_select & hazardous.astype(bool))),
        "logit_interval_width_mean": float(np.mean(widths)),
        "logit_interval_width_q90": float(np.quantile(widths, 0.9)),
    }


def _plot(path: Path, summaries: Mapping[str, Any], radii: list[float], explanation_radius: float) -> None:
    datasets = ("nuScenes test", "AV2 consumed zero-shot")
    colors = {"robust_select": "#16a34a", "interval_unresolved": "#f59e0b", "robust_abstain": "#64748b"}
    figure, axes = plt.subplots(1, 3, figsize=(14.4, 4.1))
    for dataset in datasets:
        selected = [summaries[dataset]["all_features"][str(radius)]["robust_select_coverage"] for radius in radii]
        unresolved = [summaries[dataset]["all_features"][str(radius)]["interval_unresolved_fraction"] for radius in radii]
        axes[0].plot(radii, selected, marker="o", label=dataset)
        axes[1].plot(radii, unresolved, marker="o", label=dataset)
    axes[0].set(title="Certified repair coverage", xlabel="Box radius (train std)", ylabel="Actor fraction")
    axes[1].set(title="Unresolved decisions", xlabel="Box radius (train std)", ylabel="Actor fraction")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes[:2]:
        axis.grid(alpha=0.25)
        axis.set_ylim(bottom=0.0)

    group_names = ("sensor_opportunity", "physical_surface", "all_features")
    x = np.arange(len(group_names))
    width = 0.36
    for offset, dataset in ((-width / 2, datasets[0]), (width / 2, datasets[1])):
        values = [summaries[dataset][group][str(explanation_radius)]["logit_interval_width_mean"] for group in group_names]
        axes[2].bar(x + offset, values, width=width, label=dataset)
    axes[2].set_xticks(x, ["sensor\nopportunity", "physical\nsurface", "all\nfeatures"])
    axes[2].set(title=f"Mean logit interval width (r={explanation_radius:g})", ylabel="Width")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = Path(str(config["source_p4_run"]))
    run_dir = Path(str(config["runs_root"])) / "worldsim_v7" / str(config["task_id"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(run_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    started = time.monotonic()
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P7-C interval certificate is frozen to CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    try:
        artifact = torch.load(source / "MODEL.pt", map_location="cpu", weights_only=False)
        if tuple(artifact["validity_feature_names"]) != tuple(VALIDITY_FEATURE_NAMES):
            raise RuntimeError("P4 validity feature order differs from frozen P7-C contract")
        model = FactorizedTwoHead(len(VALIDITY_FEATURE_NAMES), len(HAZARD_FEATURE_NAMES), 32).to(
            device=device, dtype=torch.float64
        )
        model.load_state_dict(artifact["factorized_state"])
        model.eval()
        mean = np.asarray(artifact["validity_standardizer"]["mean"], dtype=np.float64)
        scale = np.asarray(artifact["validity_standardizer"]["scale"], dtype=np.float64)
        threshold = float(artifact["thresholds"]["factorized"])
        threshold_logit = math.log(threshold / (1.0 - threshold))
        files = {
            "nuScenes calibration": source / "NUSCENES_CALIBRATION_ACTORS.jsonl",
            "nuScenes test": source / "NUSCENES_TEST_ACTORS.jsonl",
            "AV2 consumed zero-shot": source / "AV2_ZERO_SHOT_ACTORS.jsonl",
        }
        groups = {name: tuple(features) for name, features in config["feature_groups"].items()}
        groups["all_features"] = tuple(VALIDITY_FEATURE_NAMES)
        feature_index = {name: index for index, name in enumerate(VALIDITY_FEATURE_NAMES)}
        radii = [float(value) for value in config["standardized_radii"]]
        explanation_radius = float(config["explanation_radius"])
        summaries: dict[str, Any] = {}
        certificate_rows: list[dict[str, Any]] = []
        maximum_nominal_containment_error = 0.0
        for dataset, path in files.items():
            rows = _read_jsonl(path)
            raw = rows_to_arrays(rows)
            standardized = (raw["validity"].astype(np.float64) - mean) / scale
            values = torch.as_tensor(standardized, device=device, dtype=torch.float64)
            with torch.inference_mode():
                nominal_logit = model.repair_head(values).cpu().numpy()
            nominal_probability = 1.0 / (1.0 + np.exp(-nominal_logit))
            nominal_selected = nominal_probability >= threshold
            summaries[dataset] = {}
            bounds: dict[tuple[str, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for group_name, feature_names in groups.items():
                mask = torch.zeros(len(VALIDITY_FEATURE_NAMES), device=device, dtype=torch.float64)
                mask[[feature_index[name] for name in feature_names]] = 1.0
                summaries[dataset][group_name] = {}
                for radius in radii:
                    delta = radius * mask
                    with torch.inference_mode():
                        lower, upper = _interval_forward(model.repair_head.network, values - delta, values + delta)
                    lower_np = lower.cpu().numpy()
                    upper_np = upper.cpu().numpy()
                    lower_probability = 1.0 / (1.0 + np.exp(-lower_np))
                    upper_probability = 1.0 / (1.0 + np.exp(-upper_np))
                    states = _state(lower_probability, upper_probability, threshold)
                    containment = np.maximum(lower_np - nominal_logit, nominal_logit - upper_np)
                    maximum_nominal_containment_error = max(
                        maximum_nominal_containment_error, float(np.max(containment))
                    )
                    summaries[dataset][group_name][str(radius)] = _summarize(
                        states,
                        nominal_selected,
                        raw["repairable"].astype(bool),
                        raw["hazardous"].astype(bool),
                        lower_np,
                        upper_np,
                    )
                    bounds[(group_name, radius)] = (lower_np, upper_np, states)

            per_feature_width = np.zeros((len(rows), len(VALIDITY_FEATURE_NAMES)), dtype=np.float64)
            for index in range(len(VALIDITY_FEATURE_NAMES)):
                delta = torch.zeros(len(VALIDITY_FEATURE_NAMES), device=device, dtype=torch.float64)
                delta[index] = explanation_radius
                with torch.inference_mode():
                    lower, upper = _interval_forward(model.repair_head.network, values - delta, values + delta)
                per_feature_width[:, index] = (upper - lower).cpu().numpy()
            top_indices = np.argsort(-per_feature_width, axis=1, kind="stable")[:, :3]
            all_lower, all_upper, all_states = bounds[("all_features", explanation_radius)]
            opportunity_states = bounds[("sensor_opportunity", explanation_radius)][2]
            physical_states = bounds[("physical_surface", explanation_radius)][2]
            for row_index, row in enumerate(rows):
                certificate_rows.append(
                    {
                        "dataset": dataset,
                        "scene_or_log": row.get("scene_name", row.get("log_id")),
                        "track_id": row["track_id"],
                        "category": row["category"],
                        "hazardous": bool(row["hazardous"]),
                        "repairable_target_retained_description": bool(row["target_supported_repairable"]),
                        "threshold": threshold,
                        "nominal_probability": float(nominal_probability[row_index]),
                        "nominal_selected": bool(nominal_selected[row_index]),
                        "explanation_radius_train_std": explanation_radius,
                        "all_feature_probability_interval": [
                            float(1.0 / (1.0 + math.exp(-all_lower[row_index]))),
                            float(1.0 / (1.0 + math.exp(-all_upper[row_index]))),
                        ],
                        "all_feature_state": str(all_states[row_index]),
                        "sensor_opportunity_state": str(opportunity_states[row_index]),
                        "physical_surface_state": str(physical_states[row_index]),
                        "top_interval_width_features": [
                            {
                                "name": VALIDITY_FEATURE_NAMES[index],
                                "one_feature_logit_width": float(per_feature_width[row_index, index]),
                            }
                            for index in top_indices[row_index]
                        ],
                    }
                )

        _write_jsonl(run_dir / "ACTOR_INTERVAL_CERTIFICATES.jsonl", certificate_rows)
        _plot(run_dir / "VALIDITY_INTERVAL_CERTIFICATE", summaries, radii, explanation_radius)
        summary = {
            "schema_version": "worldsim_v7.p7c_validity_interval_certificate.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "descriptive_frozen_validity_interval_certificate",
            "source_p4_run": str(source),
            "claim_boundary": config["claim_boundary"],
            "threshold_probability": threshold,
            "threshold_logit": threshold_logit,
            "standardized_radii": radii,
            "explanation_radius": explanation_radius,
            "feature_groups": groups,
            "actor_counts": {
                dataset: int(len(_read_jsonl(path))) for dataset, path in files.items()
            },
            "interval_summaries": summaries,
            "soundness": {
                "method": "layerwise interval bound propagation in float64",
                "maximum_nominal_logit_outside_interval": max(maximum_nominal_containment_error, 0.0),
                "network_or_threshold_changed": False,
                "target_used_for_certificate_state": False,
            },
            "resources": {
                "gpu": torch.cuda.get_device_name(0),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                "wall_seconds": time.monotonic() - started,
            },
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(run_dir / "status.json", {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        return {"run_dir": str(run_dir), "verdict": summary["verdict"]}
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
    print(json.dumps(run(args.config.resolve(), args.run_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

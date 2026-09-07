"""受控验证 EAS 的阻挡/检出分解与有序第一回波测度。"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_proj.worldsim_v72.eas_vggt.ordered_returns import (
    ordered_return_distribution,
    split_blocking_probability,
)


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _probabilities(distribution: Any) -> torch.Tensor:
    return torch.cat(
        [distribution.event_probability, distribution.no_return_probability[:, None]],
        dim=1,
    )


def _make_split(count: int, seed: int, device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    event_count = 3
    occupancy = torch.randn(count, event_count, generator=generator)
    reflectivity = torch.rand(count, event_count, generator=generator)
    is_background = torch.zeros(count, event_count)
    is_background[:, -1] = 1.0
    occupancy[:, -1] = 1.5 + 0.2 * torch.randn(count, generator=generator)
    reflectivity[:, -1] = 0.75 + 0.2 * torch.rand(count, generator=generator)
    valid = torch.rand(count, event_count, generator=generator) > 0.12
    valid[:, -1] = True
    base_depth = torch.tensor([8.0, 16.0, 32.0])[None, :]
    depth = base_depth + torch.randn(count, event_count, generator=generator) * torch.tensor(
        [1.2, 1.8, 2.5]
    )[None, :]
    # 训练真值由局部、可辨识的物理阻挡和材料检出因素分别产生。
    blocking_logits = 2.5 * occupancy + 1.1 * is_background - 0.35
    detection_logits = 5.0 * (reflectivity - 0.48) + 0.8 * is_background
    features = torch.stack(
        [occupancy, reflectivity, is_background, depth / 40.0], dim=-1
    )
    target = _probabilities(
        ordered_return_distribution(depth, blocking_logits, detection_logits, valid)
    )
    return {
        "features": features.to(device),
        "depth": depth.to(device),
        "valid": valid.to(device),
        "target": target.to(device),
        "blocking_logits": blocking_logits.to(device),
        "detection_logits": detection_logits.to(device),
    }


class LocalEventHead(nn.Module):
    def __init__(self, output_count: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_count),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def _predict(
    model: LocalEventHead,
    batch: dict[str, torch.Tensor],
    variant: str,
) -> torch.Tensor:
    raw = model(batch["features"])
    if variant == "separated_blocking_detection":
        blocking_logits, detection_logits = raw.unbind(dim=-1)
    elif variant == "single_hazard":
        blocking_logits = raw[..., 0]
        detection_logits = torch.full_like(blocking_logits, 14.0)
    else:
        raise ValueError(variant)
    return _probabilities(
        ordered_return_distribution(
            batch["depth"], blocking_logits, detection_logits, batch["valid"]
        )
    )


def _soft_nll(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * predicted.clamp_min(1.0e-8).log()).sum(dim=1).mean()


def _fit(
    variant: str,
    train: dict[str, torch.Tensor],
    config: dict[str, Any],
    seed: int,
) -> LocalEventHead:
    torch.manual_seed(seed)
    output_count = 2 if variant == "separated_blocking_detection" else 1
    model = LocalEventHead(output_count, int(config["model"]["hidden_dim"])).to(
        train["features"].device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimization"]["learning_rate"]),
        weight_decay=float(config["optimization"]["weight_decay"]),
    )
    batch_size = int(config["optimization"]["batch_size"])
    steps = int(config["optimization"]["steps"])
    generator = torch.Generator(device="cpu").manual_seed(seed + 1000)
    for _ in range(steps):
        indices = torch.randint(
            len(train["target"]), (batch_size,), generator=generator
        ).to(train["features"].device)
        batch = {key: value[indices] for key, value in train.items()}
        loss = _soft_nll(_predict(model, batch, variant), batch["target"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


def _metrics(
    model: LocalEventHead,
    batch: dict[str, torch.Tensor],
    variant: str,
) -> dict[str, float]:
    with torch.inference_mode():
        predicted = _predict(model, batch, variant)
        target = batch["target"]
        low_detected_front = (
            batch["valid"][:, 0]
            & (torch.sigmoid(batch["blocking_logits"][:, 0]) > 0.65)
            & (torch.sigmoid(batch["detection_logits"][:, 0]) < 0.35)
        )

        def subset_metrics(mask: torch.Tensor) -> dict[str, float]:
            pred = predicted[mask]
            truth = target[mask]
            return {
                "ray_count": int(mask.sum()),
                "soft_nll": float(_soft_nll(pred, truth)),
                "multiclass_brier": float(((pred - truth) ** 2).sum(1).mean()),
                "total_variation": float((0.5 * (pred - truth).abs().sum(1)).mean()),
                "background_probability_mae": float((pred[:, 2] - truth[:, 2]).abs().mean()),
                "no_return_probability_mae": float((pred[:, -1] - truth[:, -1]).abs().mean()),
            }

        return {
            **subset_metrics(torch.ones(len(target), dtype=torch.bool, device=target.device)),
            "undetected_front_ray_count": int(low_detected_front.sum()),
            "undetected_front_soft_nll": subset_metrics(low_detected_front)["soft_nll"],
            "undetected_front_background_probability_mae": subset_metrics(low_detected_front)[
                "background_probability_mae"
            ],
            "undetected_front_no_return_probability_mae": subset_metrics(low_detected_front)[
                "no_return_probability_mae"
            ],
        }


def _movement_intervention(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    moved = {key: value.clone() for key, value in batch.items()}
    # 把原本最近的 Actor 移到背景之后；局部事件属性不变，只改变 world-ray 顺序。
    moved["depth"][:, 0] = 38.0 + 2.0 * torch.sigmoid(moved["features"][:, 0, 0])
    moved["features"][:, :, 3] = moved["depth"] / 40.0
    moved["target"] = _probabilities(
        ordered_return_distribution(
            moved["depth"],
            moved["blocking_logits"],
            moved["detection_logits"],
            moved["valid"],
        )
    )
    return moved


def _mass_split_error(device: torch.device) -> float:
    probability = torch.tensor([0.82], device=device)
    split = split_blocking_probability(probability, 7)
    recomposed = 1.0 - torch.prod(1.0 - split.repeat(7))
    return float(torch.abs(recomposed - probability).max())


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v72" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    _write_json(run_dir / "status.json", {"status": "running", "phase": "fit"})
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("ordered-return experiment requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    seed = int(config["seed"])
    train = _make_split(int(config["data"]["train_rays"]), seed, device)
    test = _make_split(int(config["data"]["test_rays"]), seed + 1, device)
    moved = _movement_intervention(test)
    rows: dict[str, Any] = {}
    for offset, variant in enumerate(
        ("separated_blocking_detection", "single_hazard")
    ):
        model = _fit(variant, train, config, seed + offset)
        rows[variant] = {
            "heldout": _metrics(model, test, variant),
            "actor_moved_behind_background": _metrics(model, moved, variant),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }
        torch.save(model.state_dict(), run_dir / f"{variant}.pt")
    separated = rows["separated_blocking_detection"]["heldout"]
    single = rows["single_hazard"]["heldout"]
    moved_separated = rows["separated_blocking_detection"]["actor_moved_behind_background"]
    moved_single = rows["single_hazard"]["actor_moved_behind_background"]
    mass_error = _mass_split_error(device)
    decisions = {
        "distribution_has_complete_no_return_outcome": True,
        "separated_factorization_improves_heldout_nll": separated["soft_nll"]
        < single["soft_nll"],
        "separated_factorization_improves_undetected_blocker_background_mae": separated[
            "undetected_front_background_probability_mae"
        ]
        < single["undetected_front_background_probability_mae"],
        "separated_factorization_survives_order_intervention": moved_separated[
            "soft_nll"
        ]
        < moved_single["soft_nll"],
        "surface_mass_split_invariant": mass_error
        <= float(config["decision"]["maximum_mass_split_error"]),
    }
    summary = {
        "schema_version": "worldsim_v72.e2_ordered_return.v1",
        "task_id": config["task_id"],
        "run_id": run_id,
        "run_uri": f"run://worldsim_v72/{config['task_id']}/{run_id}",
        "status": "done",
        "verdict": "ordered_blocking_detection_measure_supported"
        if all(decisions.values())
        else "ordered_blocking_detection_measure_not_supported",
        "protocol": {
            "data": "controlled_analytic_multi_actor_background_rays",
            "supervision": "proper_soft_outcome_likelihood",
            "outcomes": ["front_actor", "rear_actor", "background", "no_return"],
            "invalid_and_unsupported_excluded_from_valid_emission_denominator": True,
            "test_quality_used_for_selection": False,
            "real_sensor_scope": "mechanism_only; real positive-return distance requires separate evaluation",
        },
        "models": rows,
        "mass_split_max_error": mass_error,
        "decisions": decisions,
        "resources": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "peak_gpu_memory_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
            "wall_seconds": time.monotonic() - started,
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "source_test_read": False,
        "external_test_read": False,
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "manifest.json", {"status": "done", "summary": "summary.json", **summary["protocol"]})
    _write_json(run_dir / "status.json", {"status": "done", "phase": "complete"})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.config.resolve(), arguments.run_id), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

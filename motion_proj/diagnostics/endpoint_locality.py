"""F0: fixed-pair endpoint correction/locality Pareto audit.

This diagnostic deliberately keeps the existing endpoint target and LoRA scope.  Its
only semantic correction is to apply ``lambda_preserve`` *outside* the normalized
preservation loss.  Passing a scalar through ``outside_mask_preserve_v_loss(weight=)``
does not change that loss because the scalar is normalized out.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

from ..backbones import build_backbone
from ..cache.dataset import ProjectionCacheDataset
from ..config import config_fingerprint, get_paths, load_config, save_resolved_config
from ..losses import correction_v_loss, outside_mask_preserve_v_loss, teacher_relative_v_target
from ..runtime.atomic import atomic_write_json, atomic_write_text
from ..runtime.experiment import JsonlMetrics, RunManifest, utc_now
from ..runtime.fingerprint import environment_fingerprint, file_fingerprint, git_state, sha256_json
from ..train.pilot import _model_output_roundtrip_error, _to_batch
from ..train.trainer import seed_everything


def endpoint_objective(
    correction_loss: torch.Tensor,
    preserve_loss: torch.Tensor,
    lambda_preserve: float,
) -> torch.Tensor:
    """Apply the global preservation coefficient after spatial normalization."""
    if not math.isfinite(lambda_preserve) or lambda_preserve < 0:
        raise ValueError("lambda_preserve must be finite and non-negative")
    return correction_loss + float(lambda_preserve) * preserve_loss


def locality_decision(
    rows: list[dict[str, Any]],
    *,
    max_correction_fraction: float,
    max_outside_teacher_drift_ratio: float,
    max_frame0_teacher_drift: float,
) -> dict[str, Any]:
    """Classify F0 without conflating optimization and locality failures."""
    reached = [row for row in rows if float(row["correction_fraction"]) <= max_correction_fraction]
    feasible = [
        row
        for row in reached
        if float(row["outside_teacher_drift_ratio"]) <= max_outside_teacher_drift_ratio
        and abs(float(row["frame0_teacher_drift"])) <= max_frame0_teacher_drift
    ]
    if feasible:
        classification = "feasible_region_found"
    elif reached:
        classification = "locality_failure"
    else:
        classification = "optimization_failure_at_preregistered_budget"
    return {
        "classification": classification,
        "passed": bool(feasible),
        "correction_threshold_reached": bool(reached),
        "reached_count": len(reached),
        "feasible_count": len(feasible),
        "feasible_points": [
            {key: row[key] for key in (
                "lambda_preserve", "step", "correction_fraction",
                "outside_teacher_drift_ratio", "frame0_teacher_drift",
            )}
            for row in feasible
        ],
    }


def _noise_bank(item: dict[str, Any], cfg: Any, destination: Path) -> dict[str, Any]:
    base = item["base_latent"].float()
    sigma_value = float(cfg.f0.sigma)
    generator = torch.Generator(device="cpu").manual_seed(
        int(cfg.f0.noise_seed) + int(cfg.f0.dataset_index)
    )
    noise = torch.randn(base.shape, generator=generator, dtype=base.dtype)
    row = {
        "dataset_index": int(cfg.f0.dataset_index),
        "sample_id": str(item["metadata"]["sample_id"]),
        "sigma": torch.tensor([sigma_value], dtype=base.dtype),
        "noise": noise,
        "z_sigma": base + sigma_value * noise,
    }
    payload = {
        "task_id": str(cfg.f0.task_id),
        "noise_seed": int(cfg.f0.noise_seed),
        "row": row,
    }
    torch.save(payload, destination)
    return payload


def _loss_terms(backbone: Any, batch: dict[str, Any], f0_cfg: Any) -> dict[str, torch.Tensor]:
    student = backbone.predict_model_output(batch["z"], batch["sigma"], batch["condition"])
    target = teacher_relative_v_target(
        backbone,
        batch["z"],
        batch["sigma"],
        batch["condition"],
        batch["base"],
        batch["projected"],
        batch["static"],
        batch["object"],
        eta=float(f0_cfg.eta),
        trust_region_B=float(f0_cfg.trust_region_B),
    )
    correction = correction_v_loss(
        student, target["target"], batch["static"], batch["object"]
    )["loss"]
    # ``weight`` is a normalized per-element confidence, not a global coefficient.
    preserve = outside_mask_preserve_v_loss(
        student,
        target["teacher"],
        target["union_mask"],
        weight=1.0,
        dilation_radius=int(f0_cfg.dilation_radius),
    )
    return {
        "student": student,
        "teacher": target["teacher"],
        "target": target["target"],
        "outside": preserve["outside_mask"],
        "correction_loss": correction,
        "preserve_loss": preserve["loss"],
        "eta_eff": target["eta_eff"],
        "correction_rms": target["correction_rms"],
        "trust_region_clipping_fraction": target["trust_region_clipping_fraction"],
    }


def _gradient_vector(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True
    )
    return torch.cat([
        (torch.zeros_like(parameter) if gradient is None else gradient)
        .detach().float().reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ])


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) <= 1.0e-12:
        return 0.0
    return float(torch.dot(left, right) / denominator)


def _masked_rms(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.to(value).expand_as(value)
    return ((value.square() * expanded).sum() / expanded.sum().clamp_min(1.0)).sqrt()


def _evaluate(
    backbone: Any,
    batch: dict[str, Any],
    f0_cfg: Any,
    parameters: list[torch.nn.Parameter],
    *,
    lambda_preserve: float,
    step: int,
    initial_correction: float | None,
) -> dict[str, Any]:
    terms = _loss_terms(backbone, batch, f0_cfg)
    correction_vector = _gradient_vector(
        terms["correction_loss"], parameters, retain_graph=True
    )
    preserve_vector = _gradient_vector(
        terms["preserve_loss"], parameters, retain_graph=False
    )
    total_vector = correction_vector + float(lambda_preserve) * preserve_vector

    with torch.no_grad():
        student = terms["student"].detach()
        teacher = terms["teacher"].detach()
        outside = terms["outside"].detach()
        raw_drift = _masked_rms(student - teacher, outside)
        raw_teacher_rms = _masked_rms(teacher, outside)
        student_x0 = backbone.x0_from_model_output(batch["z"], batch["sigma"], student)
        teacher_x0 = backbone.x0_from_model_output(batch["z"], batch["sigma"], teacher)
        x0_drift = _masked_rms(student_x0 - teacher_x0, outside)
        x0_teacher_rms = _masked_rms(teacher_x0, outside)
        latent_delta = batch["projected"] - batch["base"]
        object_mask = batch["object"].expand_as(student_x0)
        applied_delta = (student_x0 - teacher_x0) * object_mask
        target_delta = latent_delta * object_mask
        direction_denominator = (
            torch.linalg.vector_norm(applied_delta.float())
            * torch.linalg.vector_norm(target_delta.float())
        ).clamp_min(1.0e-12)
        direction = float((applied_delta.float() * target_delta.float()).sum() / direction_denominator)
        correction = float(terms["correction_loss"].detach())
        baseline = correction if initial_correction is None else float(initial_correction)
        roundtrip = _model_output_roundtrip_error(
            backbone, batch["z"], batch["sigma"], terms["target"].detach()
        )
        row = {
            "phase": "evaluation",
            "lambda_preserve": float(lambda_preserve),
            "step": int(step),
            "sample_id": str(batch["sample_id"]),
            "sigma": float(batch["sigma"].flatten()[0]),
            "correction_loss": correction,
            "correction_fraction": correction / max(baseline, 1.0e-12),
            "preserve_loss": float(terms["preserve_loss"].detach()),
            "outside_teacher_rms": float(raw_teacher_rms),
            "outside_teacher_drift_rms": float(raw_drift),
            "outside_teacher_drift_ratio": float(raw_drift / raw_teacher_rms.clamp_min(1.0e-12)),
            "outside_x0_teacher_rms": float(x0_teacher_rms),
            "outside_x0_drift_rms": float(x0_drift),
            "outside_x0_drift_ratio": float(x0_drift / x0_teacher_rms.clamp_min(1.0e-12)),
            "frame0_teacher_drift": float((student[:, 0] - teacher[:, 0]).abs().max()),
            "frame0_x0_teacher_drift": float((student_x0[:, 0] - teacher_x0[:, 0]).abs().max()),
            "correction_direction_cosine": direction,
            "correction_grad_norm": float(torch.linalg.vector_norm(correction_vector)),
            "preserve_grad_norm": float(torch.linalg.vector_norm(preserve_vector)),
            "weighted_preserve_grad_norm": float(
                float(lambda_preserve) * torch.linalg.vector_norm(preserve_vector)
            ),
            "total_grad_norm": float(torch.linalg.vector_norm(total_vector)),
            "correction_preserve_grad_cosine": _cosine(correction_vector, preserve_vector),
            "eta_eff": [float(value) for value in terms["eta_eff"].detach().flatten()],
            "correction_rms": [float(value) for value in terms["correction_rms"].detach().flatten()],
            "trust_region_clipping_fraction": float(
                terms["trust_region_clipping_fraction"].detach()
            ),
            "target_roundtrip_max_error": float(roundtrip["absolute"]),
            "target_roundtrip_max_relative_error": float(roundtrip["relative"]),
        }
    return row


def _write_pareto_figure(rows: list[dict[str, Any]], destination: Path, f0_cfg: Any) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for value in sorted({float(row["lambda_preserve"]) for row in rows}):
        selected = [row for row in rows if float(row["lambda_preserve"]) == value]
        x = [float(row["correction_fraction"]) for row in selected]
        axes[0].plot(x, [float(row["outside_teacher_drift_ratio"]) for row in selected], marker="o", label=f"lambda={value:g}")
        axes[1].plot(x, [float(row["frame0_teacher_drift"]) for row in selected], marker="o", label=f"lambda={value:g}")
    for axis in axes:
        axis.axvline(float(f0_cfg.max_correction_fraction), color="black", linestyle="--", linewidth=1)
        axis.set_xlabel("correction loss / initial loss")
        axis.grid(alpha=0.25)
    axes[0].axhline(float(f0_cfg.max_outside_teacher_drift_ratio), color="red", linestyle="--", linewidth=1)
    axes[0].set_ylabel("outside raw-v drift / Base RMS")
    axes[1].axhline(float(f0_cfg.max_frame0_teacher_drift), color="red", linestyle="--", linewidth=1)
    axes[1].set_ylabel("frame-0 raw-v max drift")
    axes[0].legend(fontsize=8)
    fig.suptitle("F0 endpoint correction-preservation Pareto")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def _write_pareto_csv(rows: list[dict[str, Any]], destination: Path) -> None:
    fields = [
        "lambda_preserve", "step", "correction_loss", "correction_fraction",
        "outside_teacher_drift_ratio", "outside_x0_drift_ratio",
        "frame0_teacher_drift", "frame0_x0_teacher_drift",
        "correction_grad_norm", "preserve_grad_norm", "total_grad_norm",
        "correction_preserve_grad_cosine",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in rows])


def run_endpoint_locality(cfg: Any) -> dict[str, Any]:
    f0_cfg = cfg.f0
    max_steps = int(cfg.train.max_steps)
    if not 1 <= max_steps <= 200:
        raise ValueError("F0 max_steps must be in [1, 200]")
    evaluation_steps = sorted({int(value) for value in f0_cfg.evaluation_steps})
    if not evaluation_steps or evaluation_steps[0] != 0 or evaluation_steps[-1] != max_steps:
        raise ValueError("F0 evaluation_steps must include 0 and max_steps")
    lambdas = [float(value) for value in f0_cfg.preserve_lambdas]
    if not lambdas or len(lambdas) != len(set(lambdas)):
        raise ValueError("F0 preserve_lambdas must be a non-empty unique list")
    for value in lambdas:
        endpoint_objective(torch.tensor(0.0), torch.tensor(0.0), value)

    git = git_state(".")
    if git.get("dirty"):
        raise RuntimeError("formal F0 refuses to run in a dirty worktree")
    work_dir = Path(cfg.work_dir)
    if work_dir.exists():
        raise RuntimeError(f"F0 run directory already exists: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=False)
    (work_dir / "figures").mkdir()
    paths = get_paths(cfg)
    dataset = ProjectionCacheDataset(
        str(paths.cache_dir), expected_fingerprint=str(f0_cfg.cache_fingerprint)
    )
    dataset_index = int(f0_cfg.dataset_index)
    item = dataset[dataset_index]
    sample_id = str(item["metadata"]["sample_id"])
    if sample_id != str(f0_cfg.expected_sample_id):
        raise RuntimeError(
            f"F0 sample mismatch: expected={f0_cfg.expected_sample_id}, actual={sample_id}"
        )
    noise_path = work_dir / "noise_bank.pt"
    bank = _noise_bank(item, cfg, noise_path)
    noise_fingerprint = file_fingerprint(str(noise_path))
    config_fp = config_fingerprint(cfg)
    manifest = RunManifest(
        run_id=str(cfg.run_id),
        command=list(sys.argv),
        config_fingerprint=config_fp,
        cache_fingerprint=str(f0_cfg.cache_fingerprint),
        seed=int(cfg.seed),
        git=git,
        environment=environment_fingerprint(),
        data_split=str(cfg.data.split),
    )
    manifest_data = manifest.__dict__ | {
        "task_id": str(f0_cfg.task_id),
        "dataset_index": dataset_index,
        "sample_id": sample_id,
        "noise_bank_fingerprint": noise_fingerprint,
        "preregistration": {
            "preserve_lambdas": lambdas,
            "evaluation_steps": evaluation_steps,
            "max_correction_fraction": float(f0_cfg.max_correction_fraction),
            "max_outside_teacher_drift_ratio": float(f0_cfg.max_outside_teacher_drift_ratio),
            "max_frame0_teacher_drift": float(f0_cfg.max_frame0_teacher_drift),
            "classification_rule": (
                "optimization_failure if correction threshold is never reached; "
                "locality_failure if reached but no point satisfies both locality gates"
            ),
        },
    }
    atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
    save_resolved_config(cfg, str(work_dir / "resolved.yaml"))
    metrics = JsonlMetrics(str(work_dir / "metrics.jsonl"))
    rows: list[dict[str, Any]] = []
    variant_results: dict[str, Any] = {}
    try:
        device = torch.device(str(cfg.device))
        batch = _to_batch(item, bank["row"], device)
        for lambda_preserve in lambdas:
            seed_everything(int(cfg.seed), deterministic=bool(cfg.train.deterministic))
            backbone = build_backbone(cfg.model, load=True, device=str(device))
            backbone.set_train_mode(True)
            parameters = [parameter for parameter in backbone.trainable_parameters() if parameter.requires_grad]
            if not parameters:
                raise RuntimeError("F0 requires trainable LoRA parameters")
            if "model" not in manifest_data:
                adapter = backbone.adapter_metadata()
                manifest_data["model"] = {"name": str(cfg.model.name), "adapter": adapter}
                atomic_write_text(
                    str(work_dir / "selected_modules.txt"),
                    "".join(f"{name}\n" for name in adapter["selected_module_names"]),
                )
                atomic_write_json(str(work_dir / "manifest.json"), manifest_data)

            optimizer = torch.optim.AdamW(
                parameters,
                lr=float(cfg.train.lr),
                weight_decay=float(cfg.train.weight_decay),
            )
            initial = _evaluate(
                backbone, batch, f0_cfg, parameters,
                lambda_preserve=lambda_preserve, step=0, initial_correction=None,
            )
            initial_correction = float(initial["correction_loss"])
            rows.append(initial)
            metrics.append(0, initial)
            gradient_finite = True
            gradient_nonzero = False
            for step in range(1, max_steps + 1):
                optimizer.zero_grad(set_to_none=True)
                terms = _loss_terms(backbone, batch, f0_cfg)
                total = endpoint_objective(
                    terms["correction_loss"], terms["preserve_loss"], lambda_preserve
                )
                total.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, float(cfg.train.max_grad_norm)
                )
                gradient_finite = gradient_finite and bool(torch.isfinite(grad_norm))
                gradient_nonzero = gradient_nonzero or float(grad_norm) > 0.0
                optimizer.step()
                if step % int(cfg.train.log_every) == 0:
                    metrics.append(step, {
                        "phase": "training",
                        "lambda_preserve": lambda_preserve,
                        "sample_id": sample_id,
                        "sigma": float(batch["sigma"].flatten()[0]),
                        "total_loss": float(total.detach()),
                        "correction_loss": float(terms["correction_loss"].detach()),
                        "preserve_loss": float(terms["preserve_loss"].detach()),
                        "grad_norm": float(grad_norm),
                    })
                if step in evaluation_steps:
                    row = _evaluate(
                        backbone, batch, f0_cfg, parameters,
                        lambda_preserve=lambda_preserve,
                        step=step,
                        initial_correction=initial_correction,
                    )
                    rows.append(row)
                    metrics.append(step, row)
            variant_dir = work_dir / "variants" / f"lambda-{lambda_preserve:g}"
            variant_dir.mkdir(parents=True, exist_ok=False)
            checkpoint = variant_dir / "adapter.safetensors"
            backbone.save_adapter(str(checkpoint))
            variant_rows = [
                row for row in rows if float(row["lambda_preserve"]) == lambda_preserve
            ]
            variant_results[str(lambda_preserve)] = {
                "initial": variant_rows[0],
                "final": variant_rows[-1],
                "best_correction": min(variant_rows, key=lambda row: float(row["correction_fraction"])),
                "gradient_finite": gradient_finite,
                "gradient_nonzero": gradient_nonzero,
                "checkpoint": str(checkpoint),
            }
            del optimizer, backbone
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        decision = locality_decision(
            rows,
            max_correction_fraction=float(f0_cfg.max_correction_fraction),
            max_outside_teacher_drift_ratio=float(f0_cfg.max_outside_teacher_drift_ratio),
            max_frame0_teacher_drift=float(f0_cfg.max_frame0_teacher_drift),
        )
        _write_pareto_csv(rows, work_dir / "figures" / "pareto.csv")
        _write_pareto_figure(rows, work_dir / "figures" / "pareto.png", f0_cfg)
        summary = {
            "status": "completed",
            "task_id": str(f0_cfg.task_id),
            "dataset_index": dataset_index,
            "sample_id": sample_id,
            "noise_bank_fingerprint": noise_fingerprint,
            "max_steps": max_steps,
            "variants": variant_results,
            "decision": decision,
            "experiment_fingerprint": sha256_json({
                "config": config_fp,
                "noise_bank": noise_fingerprint,
                "rows": rows,
                "decision": decision,
            }),
        }
        atomic_write_json(str(work_dir / "summary.json"), summary)
        atomic_write_text(str(work_dir / "COMPLETE"), sha256_json(summary) + "\n")
        manifest_data.update({
            "status": "completed",
            "ended_at": utc_now(),
            "exit_reason": decision["classification"],
        })
        atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
        return summary
    except Exception as exc:
        failure = {"status": "failed", "error": repr(exc)}
        atomic_write_json(str(work_dir / "summary.json"), failure)
        manifest_data.update({
            "status": "failed", "ended_at": utc_now(), "exit_reason": repr(exc)
        })
        atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    result = run_endpoint_locality(load_config(args.config, list(args.overrides)))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

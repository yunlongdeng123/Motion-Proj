"""Render a fixed baseline-difficulty M7 Actor surface and literal-ray evidence panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_worldsim_v71_m0_ray_displacement as m0_runner
import run_worldsim_v71_m5_pcgrad_relocation as m5_runner
import run_worldsim_v71_m6_gt_supervised_gaussian_relocation as m6_runner
import run_worldsim_v71_m7_gt_supervised_seed_expansion as m7_runner
from motion_proj.worldsim_v7.av2_four_action_compiler import _voxel_unique
from motion_proj.worldsim_v7.completion_responsibility import FeatureStandardizer
from motion_proj.worldsim_v71.first_return_renderer import literal_first_return_partition
from motion_proj.worldsim_v71.gaussian_anchor_relocation import GaussianSeedExpansionMLP
from motion_proj.worldsim_v71.ray_displacement import RaySurfaceRelocationMLP


COLORS = {
    "target": "#999999",
    "anchor": "#111111",
    "candidate": "#d95f02",
    "m5": "#1b9e77",
    "m7": "#3b6fb6",
}


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _select_actor(rows: list[dict]) -> dict:
    hazardous = [row for row in rows if bool(row["hazardous"])]
    ordered = sorted(
        hazardous,
        key=lambda row: (
            float(row["baseline_early_count"]) / max(int(row["target_ray_count"]), 1),
            str(row["track_id"]),
        ),
    )
    return ordered[len(ordered) // 2]


def _partition(surface: np.ndarray, actor: dict, device: torch.device) -> dict[str, np.ndarray]:
    return literal_first_return_partition(
        surface,
        actor["target"],
        actor["target_sensor_origins"],
        lateral_tolerance_m=0.20,
        depth_tolerance_m=0.20,
        device=device,
        ray_chunk_size=512,
    )


def _metrics(surface: np.ndarray, partition: dict[str, np.ndarray], actor: dict) -> dict[str, float]:
    target = torch.as_tensor(actor["target"], dtype=torch.float32, device="cuda")
    points = torch.as_tensor(surface, dtype=torch.float32, device="cuda")
    with torch.inference_mode():
        distances = torch.cdist(points, target)
        chamfer = 0.5 * (distances.min(dim=1).values.mean() + distances.min(dim=0).values.mean())
    return {
        "early_rate": float(np.mean(partition["early"])),
        "hit_rate": float(np.mean(partition["hit"])),
        "chamfer_m": float(chamfer),
    }


def _scatter_projection(
    axis: plt.Axes,
    values: dict[str, np.ndarray],
    dimensions: tuple[int, int],
    title: str,
) -> None:
    first, second = dimensions
    axis.scatter(
        values["target"][:, first], values["target"][:, second],
        s=3, c=COLORS["target"], alpha=0.20, linewidths=0, label="target surface",
    )
    axis.scatter(
        values["anchor"][:, first], values["anchor"][:, second],
        s=8, c=COLORS["anchor"], alpha=0.55, linewidths=0, label="immutable anchors",
    )
    axis.scatter(
        values["candidate"][:, first], values["candidate"][:, second],
        s=26, facecolors="none", edgecolors=COLORS["candidate"], linewidths=1.0,
        label="completion seeds",
    )
    axis.scatter(
        values["m5"][:, first], values["m5"][:, second],
        s=18, marker="x", c=COLORS["m5"], linewidths=0.8, label="M5 parent",
    )
    axis.scatter(
        values["m7"][:, first], values["m7"][:, second],
        s=7, c=COLORS["m7"], alpha=0.75, linewidths=0, label="M7 children",
    )
    axis.set_title(title)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.15, linewidth=0.5)
    axis.set_xlabel("actor x (m)")
    axis.set_ylabel("actor y (m)" if second == 1 else "actor z (m)")


def run(config_path: Path, output_stem: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / config["run_id"]
    checkpoint = torch.load(run_dir / "MODEL.pt", map_location="cuda", weights_only=False)
    resolved = yaml.safe_load((run_dir / "resolved.yaml").read_text(encoding="utf-8"))
    standardizer = FeatureStandardizer.from_payload(checkpoint["standardizer"])
    model = GaussianSeedExpansionMLP(
        int(checkpoint["input_dim"]),
        int(checkpoint["hidden_dim"]),
        int(checkpoint["branch_factor"]),
        int(checkpoint["slot_dim"]),
    ).cuda()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    m5_run = Path(checkpoint["m5_run"])
    m5_checkpoint = torch.load(m5_run / "MODEL.pt", map_location="cuda", weights_only=False)
    m5_resolved = yaml.safe_load((m5_run / "resolved.yaml").read_text(encoding="utf-8"))
    base = RaySurfaceRelocationMLP(
        int(m5_checkpoint["input_dim"]), int(m5_checkpoint["hidden_dim"])
    ).cuda()
    base.load_state_dict(m5_checkpoint["state_dict"])
    base.eval()
    paths = m0_runner._paths(Path(resolved["cache_root"]), int(resolved["model"]["maximum_training_actors"]))
    actors = [
        actor
        for path in paths
        if (actor := m0_runner._prepare_actor(path, standardizer, torch.device("cuda"))) is not None
    ]
    stride = int(resolved["model"]["holdout_stride"])
    holdout = [actor for index, actor in enumerate(actors) if index % stride == 0]
    selected_row = _select_actor(_read_rows(run_dir / "TRAIN_HOLDOUT_CHILDREN.jsonl"))
    actor = next(value for value in holdout if str(value["track_id"]) == str(selected_row["track_id"]))
    with torch.inference_mode():
        _, m5_centers = m5_runner._move(base, actor, m5_resolved["model"])
        actor["m5_centers_t"] = m5_centers
        children, _, _ = m7_runner._predict(model, actor, resolved["model"])
    voxel = float(resolved["evaluation"]["output_voxel_size_m"])
    baseline = _voxel_unique(
        np.concatenate([actor["anchors"], actor["candidates"]], axis=0), voxel
    )
    m5_surface = _voxel_unique(
        torch.cat([actor["anchors_t"], m5_centers], dim=0).cpu().numpy(), voxel
    )
    m7_surface = _voxel_unique(
        torch.cat([actor["anchors_t"], children], dim=0).cpu().numpy(), voxel
    )
    partitions = {
        "baseline": _partition(baseline, actor, torch.device("cuda")),
        "m5": _partition(m5_surface, actor, torch.device("cuda")),
        "m7": _partition(m7_surface, actor, torch.device("cuda")),
    }
    metrics = {
        "baseline": _metrics(baseline, partitions["baseline"], actor),
        "m5": _metrics(m5_surface, partitions["m5"], actor),
        "m7": _metrics(m7_surface, partitions["m7"], actor),
    }
    values = {
        "target": np.asarray(actor["target"], dtype=np.float32),
        "anchor": np.asarray(actor["anchors"], dtype=np.float32),
        "candidate": np.asarray(actor["candidates"], dtype=np.float32),
        "m5": m5_centers.cpu().numpy(),
        "m7": children.cpu().numpy(),
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.55), constrained_layout=True)
    _scatter_projection(axes[0], values, (0, 1), "(a) Actor-canonical top view")
    _scatter_projection(axes[1], values, (0, 2), "(b) Actor-canonical side view")
    bins = np.linspace(-1.0, 1.0, 45)
    for name, label in (("baseline", "baseline"), ("m5", "M5"), ("m7", "M7")):
        part = partitions[name]
        residual = part["first_depth"][part["observable"]] - part["target_depth"][part["observable"]]
        axes[2].hist(
            np.clip(residual, bins[0], bins[-1]),
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            color={"baseline": COLORS["candidate"], "m5": COLORS["m5"], "m7": COLORS["m7"]}[name],
            label=(
                f"{label}: early {100*metrics[name]['early_rate']:.1f}%, "
                f"hit {100*metrics[name]['hit_rate']:.1f}%, CD {1000*metrics[name]['chamfer_m']:.1f}mm"
            ),
        )
    axes[2].axvspan(-1.0, -0.20, color="#d73027", alpha=0.08, label="literal early region")
    axes[2].axvline(0.0, color="black", linewidth=0.7)
    axes[2].set_title("(c) Literal first-return residual")
    axes[2].set_xlabel("first depth - target depth (m)")
    axes[2].set_ylabel("density")
    axes[2].grid(alpha=0.15, linewidth=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.34, -0.02))
    axes[2].legend(loc="upper right", fontsize=7, frameon=False)
    fig.suptitle(
        "Hazardous Actor selected by median baseline early-return rate (output-independent)",
        fontsize=10,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    payload = {
        "schema_version": "worldsim_v71.m7_geometry_evidence.v1",
        "selection_rule": "hazardous_holdout_median_baseline_literal_early_rate_then_track_id",
        "scene_name": str(actor["scene_name"]),
        "track_id": str(actor["track_id"]),
        "target_rays": int(len(actor["target"])),
        "anchor_count": int(len(actor["anchors"])),
        "candidate_count": int(len(actor["candidates"])),
        "m7_child_count": int(len(children)),
        "metrics": metrics,
        "output_png": str(output_stem.with_suffix(".png")),
        "output_pdf": str(output_stem.with_suffix(".pdf")),
        "external_read": False,
    }
    m6_runner._write_json(output_stem.with_suffix(".json"), payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.output_stem.resolve()), ensure_ascii=False))


if __name__ == "__main__":
    main()

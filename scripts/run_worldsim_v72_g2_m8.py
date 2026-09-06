"""以 target-free surface 生成边界重评 V7.1 M8。"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
for root in (REPO_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import run_worldsim_v71_m0_ray_displacement as m0_runner
import run_worldsim_v71_m5_pcgrad_relocation as m5_runner
import run_worldsim_v71_m7_gt_supervised_seed_expansion as m7_runner
from motion_proj.worldsim_v7.completion_responsibility import FeatureStandardizer
from motion_proj.worldsim_v71.gaussian_anchor_relocation import GaussianSeedExpansionMLP
from motion_proj.worldsim_v71.ray_displacement import RaySurfaceRelocationMLP
from motion_proj.worldsim_v72.evaluation.surface_metrics import (
    deterministic_farthest_point_sample,
    evaluate_point_surface,
)
from scripts.run_worldsim_v72_g0_raw_fusion import (
    _log_macro_summary,
    _moving,
    _stratum,
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _prepare_build_actor(
    path: Path, standardizer: FeatureStandardizer, device: torch.device
) -> dict[str, Any] | None:
    build_keys = (
        "base_features",
        "candidates",
        "size_lwh_m",
        "evidence_masses",
        "query_sensor_origin",
        "anchors",
        "scene_name",
        "track_id",
    )
    with np.load(path, allow_pickle=False) as payload:
        build = {key: payload[key] for key in build_keys}
    if len(build["candidates"]) == 0:
        return None
    features, rays, normals = m0_runner._raw_features(build, device)
    return {
        "path": str(path),
        "scene_name": str(build["scene_name"]),
        "track_id": str(build["track_id"]),
        "native_candidate_count": int(len(build["candidates"])),
        "native_anchor_count": int(len(build["anchors"])),
        "features": torch.as_tensor(
            standardizer.transform(features), dtype=torch.float32, device=device
        ),
        "candidates_t": torch.as_tensor(
            build["candidates"], dtype=torch.float32, device=device
        ),
        "anchors_t": torch.as_tensor(
            build["anchors"], dtype=torch.float32, device=device
        ),
        "ray_directions_t": torch.as_tensor(
            rays, dtype=torch.float32, device=device
        ),
        "normals_t": torch.as_tensor(normals, dtype=torch.float32, device=device),
        "size_t": torch.as_tensor(
            build["size_lwh_m"], dtype=torch.float32, device=device
        ),
    }


def _load_models(
    run_root: Path, device: torch.device
) -> tuple[
    RaySurfaceRelocationMLP,
    GaussianSeedExpansionMLP,
    FeatureStandardizer,
    Mapping[str, Any],
    Mapping[str, Any],
]:
    checkpoint = torch.load(
        run_root / "MODEL.pt", map_location=device, weights_only=False
    )
    m8_config = yaml.safe_load(
        (run_root / "resolved.yaml").read_text(encoding="utf-8")
    )
    standardizer = FeatureStandardizer.from_payload(checkpoint["standardizer"])
    m5_run = Path(checkpoint["m5_run"])
    m5_checkpoint = torch.load(
        m5_run / "MODEL.pt", map_location=device, weights_only=False
    )
    m5_config = yaml.safe_load(
        (m5_run / "resolved.yaml").read_text(encoding="utf-8")
    )
    base = RaySurfaceRelocationMLP(
        int(m5_checkpoint["input_dim"]), int(m5_checkpoint["hidden_dim"])
    ).to(device)
    base.load_state_dict(m5_checkpoint["state_dict"])
    base.eval().requires_grad_(False)
    model = GaussianSeedExpansionMLP(
        int(checkpoint["input_dim"]),
        int(checkpoint["hidden_dim"]),
        int(checkpoint["branch_factor"]),
        int(checkpoint["slot_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().requires_grad_(False)
    return base, model, standardizer, m5_config, m8_config


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config_text = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)
    if config.get("data_role") != "legacy_diagnostic":
        raise ValueError("G2 当前只允许 legacy_diagnostic")
    if bool(config.get("source_test_read")) or bool(config.get("external_test_read")):
        raise PermissionError("G2 不允许读取 source/external final")
    device = torch.device(str(config["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("G2 matched evaluator 需要 CUDA")
    run_dir = Path(config["runs_root"]) / "worldsim_v72" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "setup"})
    started = time.monotonic()
    try:
        git_commit = _git_commit()
        torch.cuda.reset_peak_memory_stats(device)
        base, model, standardizer, m5_config, m8_config = _load_models(
            Path(config["m8_run"]), device
        )
        paths = m0_runner._paths(
            Path(config["cache_root"]), int(config["maximum_source_actors"])
        )
        actors = [
            actor
            for path in paths
            if (actor := _prepare_build_actor(path, standardizer, device)) is not None
        ]
        holdout = [
            actor
            for index, actor in enumerate(actors)
            if index % int(config["holdout_stride"]) == 0
        ]
        if len(holdout) != int(config["expected_actor_count"]):
            raise RuntimeError(f"G2 holdout count {len(holdout)} 不等于冻结数量")
        expected = {
            (str(row["scene_name"]), str(row["track_id"]))
            for row in _load_rows(Path(config["cohort_rows"]))
        }
        actual = {(actor["scene_name"], actor["track_id"]) for actor in holdout}
        if expected != actual:
            raise RuntimeError("G2 identity set 与 G0/G1 不一致")
        scene_to_log = {
            str(row["name"]): str(row["log_token"])
            for row in json.loads(
                Path(config["scene_metadata"]).read_text(encoding="utf-8")
            )
        }
        surfaces: list[np.ndarray] = []
        with torch.inference_mode():
            for actor in holdout:
                _, centers = m5_runner._move(base, actor, m5_config["model"])
                actor["m5_centers_t"] = centers
                children, _, _ = m7_runner._predict(model, actor, m8_config["model"])
                surfaces.append(
                    torch.cat([actor["anchors_t"], children], dim=0).cpu().numpy()
                )
        resolved = {
            **config,
            "run_id": run_id,
            "git_commit": git_commit,
            "resolved_at_utc": datetime.now(timezone.utc).isoformat(),
            "gpu": torch.cuda.get_device_name(0),
            "torch": str(torch.__version__),
        }
        (run_dir / "resolved.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
        )
        manifest = {
            "schema_version": "worldsim_v72.run_manifest.v1",
            "task_id": config["task_id"],
            "run_id": run_id,
            "git_commit": git_commit,
            "data_role": config["data_role"],
            "training": False,
            "actor_count": len(holdout),
            "cohort_rows_sha256": _sha256(Path(config["cohort_rows"])),
            "m8_checkpoint_sha256": _sha256(Path(config["m8_run"]) / "MODEL.pt"),
            "target_free_surface_generation": True,
            "pretrained_holdout_exposure": True,
            "source_test_read": False,
            "external_test_read": False,
            "route_decision_allowed": False,
            "failure_ledger_refs": list(config["failure_ledger_refs"]),
            "failure_ledger_delta": str(config["failure_ledger_delta"]),
        }
        _write_json(run_dir / "manifest.json", manifest)
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "config": hashlib.sha256(config_text.encode()).hexdigest(),
                    "git": git_commit,
                    "cohort": manifest["cohort_rows_sha256"],
                    "checkpoint": manifest["m8_checkpoint_sha256"],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        _write_json(
            run_dir / "fingerprint.json",
            {"algorithm": "sha256", "value": fingerprint, "scope": "config+git+cohort+checkpoint"},
        )
        budgets: list[int | None] = [int(value) for value in config["density_budgets"]]
        if bool(config["include_native_density"]):
            budgets.append(None)
        evaluation = config["evaluation"]
        rows: list[dict[str, Any]] = []
        for index, (actor, prediction) in enumerate(zip(holdout, surfaces, strict=True)):
            with np.load(actor["path"], allow_pickle=False) as payload:
                target = np.asarray(payload["target"], dtype=np.float32)
                origins = np.asarray(payload["target_sensor_origins"], dtype=np.float32)
                hazardous = bool(payload["hazardous"])
                category = str(payload["category"])
                trajectory = np.asarray(payload["trajectory_xyz_m"], dtype=np.float32)
                size_lwh_m = np.asarray(payload["size_lwh_m"], dtype=np.float32)
            moving, displacement = _moving(trajectory)
            for budget in budgets:
                surface = (
                    prediction.copy()
                    if budget is None
                    else deterministic_farthest_point_sample(prediction, budget)
                )
                metrics = evaluate_point_surface(
                    surface,
                    target,
                    origins,
                    fscore_threshold_m=float(evaluation["fscore_threshold_m"]),
                    lateral_tolerance_m=float(evaluation["literal_lateral_tolerance_m"]),
                    depth_tolerance_m=float(evaluation["literal_depth_tolerance_m"]),
                    distance_chunk_size=int(evaluation["distance_chunk_size"]),
                    ray_chunk_size=int(evaluation["ray_chunk_size"]),
                    point_chunk_size=int(evaluation["point_chunk_size"]),
                )
                rows.append(
                    {
                        "scene_name": actor["scene_name"],
                        "log_id": scene_to_log[actor["scene_name"]],
                        "track_id": actor["track_id"],
                        "category": category,
                        "hazardous": hazardous,
                        "moving": moving,
                        "trajectory_max_displacement_m": displacement,
                        "size_lwh_m": size_lwh_m.tolist(),
                        "density_budget": "native" if budget is None else budget,
                        "native_input_candidate_count": actor["native_candidate_count"],
                        "native_input_anchor_count": actor["native_anchor_count"],
                        "native_output_point_count": len(prediction),
                        "surface_generation": str(config["surface_generation"]),
                        "target_used_by_surface_generation": False,
                        **metrics,
                    }
                )
            if (index + 1) % 10 == 0 or index + 1 == len(holdout):
                print(json.dumps({"stage": "g2_evaluate", "progress": f"{index + 1}/{len(holdout)}"}), flush=True)
        _write_jsonl(run_dir / "ACTORS.jsonl", rows)
        by_budget: dict[str, Any] = {}
        all_log_rows: list[dict[str, Any]] = []
        for offset, budget in enumerate(budgets):
            label = "native" if budget is None else str(budget)
            selected = [row for row in rows if str(row["density_budget"]) == label]
            log_rows, log_macro = _log_macro_summary(
                selected,
                seed=int(config["bootstrap"]["seed"]) + offset,
                samples=int(config["bootstrap"]["samples"]),
            )
            all_log_rows.extend({"density_budget": label, **row} for row in log_rows)
            by_budget[label] = {
                "all": _stratum(selected),
                "hazard": _stratum([row for row in selected if bool(row["hazardous"])]),
                "clear": _stratum([row for row in selected if not bool(row["hazardous"])]),
                "moving": _stratum([row for row in selected if bool(row["moving"])]),
                "quasi_static": _stratum([row for row in selected if not bool(row["moving"])]),
                "log_macro_bootstrap": log_macro,
            }
        _write_jsonl(run_dir / "LOGS.jsonl", all_log_rows)
        summary = {
            "schema_version": "worldsim_v72.g2_m8_matched.v1",
            **manifest,
            "status": "done",
            "verdict": "legacy_diagnostic_complete_no_route_decision",
            "baseline_id": "G2",
            "baseline_name": "M8_temporal_frame_surface",
            "density_budgets": ["native" if value is None else value for value in budgets],
            "metrics": by_budget,
            "full_return_metrics_supported": False,
            "full_return_metrics_reason": "legacy v1 cohort has positive held-out returns only",
            "resources": {
                "device": "cuda:0",
                "gpu": torch.cuda.get_device_name(0),
                "peak_gpu_memory_gib": torch.cuda.max_memory_reserved() / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "status.json",
            {"status": "done", "phase": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
        )
        return summary
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {"status": "failed", "phase": "runtime", "error": f"{type(error).__name__}: {error}"},
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.run_id), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

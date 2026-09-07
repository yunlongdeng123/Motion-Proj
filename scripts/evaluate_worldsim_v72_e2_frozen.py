"""在冻结 clean role 上评测 E2 checkpoint，不更新参数或阈值。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_worldsim_v72_e2_late_fusion as trainer
from motion_proj.worldsim_v72.eas_vggt.models import CanonicalLateFusionEvidenceAdapter
from motion_proj.worldsim_v72.evaluation.surface_metrics import evaluate_point_surface


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_model(path: Path, device: torch.device) -> tuple[CanonicalLateFusionEvidenceAdapter, str]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = CanonicalLateFusionEvidenceAdapter(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, str(checkpoint["variant"])


def _evidence_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    nll = -(target * torch.log(prediction.clamp_min(1.0e-8))).sum(dim=-1)
    brier = torch.sum((prediction - target) ** 2, dim=-1)
    return {
        "candidate_count": int(len(target)),
        "fou_nll": float(nll.mean().cpu()),
        "fou_brier": float(brier.mean().cpu()),
        "occupied_mae": float((prediction[:, 1] - target[:, 1]).abs().mean().cpu()),
    }


def _summarize_surface(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    rays = sum(row["conditional_return"]["ray_count"] for row in rows)
    return {
        "actor_count": len(rows),
        "ray_count": rays,
        "mean_symmetric_cd_l1_m": float(
            np.mean([row["geometry"]["symmetric_cd_l1_m"] for row in rows])
        ),
        "mean_fscore": float(
            np.mean([row["geometry"]["fscore_at_threshold"] for row in rows])
        ),
        "conditional_early_rate": sum(
            row["conditional_return"]["early_count"] for row in rows
        )
        / rays,
        "conditional_hit_recall": sum(
            row["conditional_return"]["hit_count"] for row in rows
        )
        / rays,
        "conditional_miss_rate": sum(
            row["conditional_return"]["miss_count"] for row in rows
        )
        / rays,
    }


def _surface_rows(
    actors: list[dict[str, Any]],
    corpus_root: Path,
    delta: torch.Tensor,
    scene_to_log: dict[str, str],
    evaluation: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    offset = 0
    delta_np = delta.detach().cpu().numpy()
    for actor in actors:
        count = len(actor["candidates"])
        source = corpus_root / actor["scene_id"] / f"{actor['track_id']}.npz"
        with np.load(source, allow_pickle=False) as payload:
            canonical = np.asarray(payload["canonical"], dtype=np.float32)
            target = np.asarray(payload["target"], dtype=np.float32)
            origins = np.asarray(payload["target_sensor_origins"], dtype=np.float32)
            hazardous = bool(payload["hazardous"])
        adjusted = actor["candidates"] + delta_np[offset : offset + count]
        surface = np.concatenate([canonical, adjusted], axis=0)
        metrics = evaluate_point_surface(
            surface,
            target,
            origins,
            **evaluation,
        )
        rows.append(
            {
                "log_id": scene_to_log[actor["scene_id"]],
                "scene_id": actor["scene_id"],
                "track_id": actor["track_id"],
                "hazardous": hazardous,
                **metrics,
            }
        )
        offset += count
    if offset != len(delta_np):
        raise RuntimeError("candidate slice accounting mismatch")
    return rows


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    role = str(config["data"]["role"])
    if role not in {"route_select", "source_test"}:
        raise PermissionError(f"frozen evaluator does not allow role {role}")
    if (role == "source_test") != bool(config["data"]["source_test_read"]):
        raise PermissionError("source_test role/read flag mismatch")
    if bool(config["data"]["external_test_read"]):
        raise PermissionError("E2 frozen evaluator does not read external test")
    run_dir = Path(config["runs_root"]) / "worldsim_v72" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    _write_json(run_dir / "status.json", {"status": "running", "phase": "load"})
    started = time.monotonic()
    device = torch.device(str(config["device"]))
    torch.cuda.reset_peak_memory_stats()
    visual_run = Path(config["data"]["visual_run"])
    corpus_root = Path(config["data"]["actor_corpus_root"])
    actors = trainer._aggregate_visual(
        visual_run, str(config["data"]["backbone"]), int(config["data"]["maximum_views"])
    )
    data, dataset_rows = trainer._prepare_dataset(
        actors, corpus_root, config["supervision_evidence"], device
    )
    metadata = json.loads(Path(config["data"]["scene_metadata"]).read_text(encoding="utf-8"))
    scene_to_log = {str(row["name"]): str(row["log_token"]) for row in metadata}
    target = data["target_fou"].float()
    zero_delta = torch.zeros_like(data["canonical_xyz"].float())
    outputs: dict[str, Any] = {}
    predictions: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
        "build_evidence_input": (data["evidence_fou"].float(), zero_delta)
    }
    checkpoint_hashes = {}
    for name, checkpoint_value in config["checkpoints"].items():
        checkpoint_path = Path(checkpoint_value)
        model, variant = _load_model(checkpoint_path, device)
        with torch.inference_mode():
            output = model(
                **trainer._model_inputs(data, variant, view_dropout_probability=0.0)
            )
        predictions[name] = (output.evidence_fou, output.surface_delta_actor_m)
        checkpoint_hashes[name] = _sha256(checkpoint_path)
        del model
    detailed_rows = {}
    for name, (prediction, delta) in predictions.items():
        _write_json(run_dir / "status.json", {"status": "running", "phase": f"evaluate_{name}"})
        surface_rows = _surface_rows(
            actors, corpus_root, delta, scene_to_log, config["surface_evaluation"]
        )
        detailed_rows[name] = surface_rows
        evidence = _evidence_metrics(prediction, target)
        per_log: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
        offset = 0
        for row in dataset_rows:
            count = int(row["candidate_count"])
            per_log[scene_to_log[row["scene_id"]]].append(
                (prediction[offset : offset + count], target[offset : offset + count])
            )
            offset += count
        log_metrics = {
            log_id: _evidence_metrics(
                torch.cat([pair[0] for pair in pairs]),
                torch.cat([pair[1] for pair in pairs]),
            )
            for log_id, pairs in sorted(per_log.items())
        }
        outputs[name] = {
            "evidence": evidence,
            "surface": _summarize_surface(surface_rows),
            "per_log_evidence": log_metrics,
        }
    primary = outputs[str(config["decision"]["primary_arm"])]["evidence"]
    comparator = outputs[str(config["decision"]["comparator_arm"])]["evidence"]
    decisions = {
        "primary_improves_frozen_role_brier": primary["fou_brier"] < comparator["fou_brier"],
        "primary_improves_frozen_role_nll": primary["fou_nll"] < comparator["fou_nll"],
        "source_test_was_not_used_for_route_selection": role != "source_test"
        or bool(config["decision"]["architecture_locked_before_source_test"]),
        "no_parameter_or_threshold_update": True,
    }
    summary = {
        "schema_version": "worldsim_v72.e2_frozen_evaluation.v1",
        "task_id": config["task_id"],
        "run_id": run_id,
        "run_uri": f"run://worldsim_v72/{config['task_id']}/{run_id}",
        "status": "done",
        "role": role,
        "verdict": "frozen_primary_supported" if all(decisions.values()) else "frozen_primary_not_supported",
        "actor_count": len(actors),
        "scene_count": len({actor["scene_id"] for actor in actors}),
        "log_count": len({scene_to_log[actor["scene_id"]] for actor in actors}),
        "results": outputs,
        "decisions": decisions,
        "checkpoint_sha256": checkpoint_hashes,
        "protocol": {
            "weights_updated": False,
            "thresholds_updated": False,
            "positive_return_only_surface_scope": True,
            "complete_no_return_scope": "separate controlled ordered-return task",
            "source_test_read": role == "source_test",
            "external_test_read": False,
        },
        "resources": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": torch.cuda.max_memory_reserved() / 1024**3,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
            "wall_seconds": time.monotonic() - started,
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
    }
    for name, rows in detailed_rows.items():
        _write_json(run_dir / f"{name}_actors.json", rows)
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "manifest.json", {"status": "done", "role": role, **summary["protocol"]})
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

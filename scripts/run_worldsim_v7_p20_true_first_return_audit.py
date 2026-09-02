"""Audit frozen V7 completion policies with literal first-return rays."""

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

from motion_proj.worldsim_v7.av2_four_action_compiler import COMPLETION_FEATURE_NAMES
from motion_proj.worldsim_v7.completion_expert_router import sparse_hazard_veto_states
from motion_proj.worldsim_v7.completion_responsibility import (
    FeatureStandardizer,
    summarize_actor_policy,
)
from motion_proj.worldsim_v7.ray_set_completion import (
    RaySetCompletionMLP,
    predict_ray_set,
)
from motion_proj.worldsim_v7.true_first_return_attribution import (
    apply_completion_policy_true_first_return,
)
from scripts.run_worldsim_v7_p19_sparse_hazard_veto import _source_test_bundles


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


def _load_model(
    fit_run: Path,
    model_config: Mapping[str, Any],
    device: torch.device,
):
    payload = torch.load(fit_run / "MODEL.pt", map_location=device, weights_only=False)
    if tuple(payload["feature_names"]) != COMPLETION_FEATURE_NAMES:
        raise RuntimeError(f"completion feature contract changed: {fit_run}")
    model = RaySetCompletionMLP(int(model_config["model"]["hidden_dim"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, FeatureStandardizer.from_payload(payload["standardizer"])


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


def run_audit(config_path: Path, repo_root: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    compiler = yaml.safe_load((repo_root / config["p2_config"]).read_text(encoding="utf-8"))
    source = yaml.safe_load((repo_root / config["p4_config"]).read_text(encoding="utf-8"))
    variant_configs = {
        name: yaml.safe_load((repo_root / row["config"]).read_text(encoding="utf-8"))
        for name, row in config["variants"].items()
        if "config" in row
    }
    attribution_config = variant_configs["p17"]["attribution"]
    run_dir = Path(config["runs_root"]) / "worldsim_v7" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "audit"})
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P20 audit requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        models = {}
        for name in ("p17", "p17r"):
            models[name] = _load_model(
                Path(config["variants"][name]["fit_run"]), variant_configs[name], device
            )
        bundles = _source_test_bundles(compiler, source, device)
        variant_rows = {name: [] for name in ("p17", "p17r", "p19")}
        for position, bundle in enumerate(bundles):
            features = np.asarray(
                bundle["diagnostics"]["completion_features"], dtype=np.float32
            )
            predictions = {}
            for name in ("p17", "p17r"):
                model, standardizer = models[name]
                threshold = float(
                    variant_configs[name]["model"]["forward_selection_threshold"]
                )
                predictions[name] = predict_ray_set(
                    model, standardizer, features, threshold, device
                )
            p17r_occupancy = predictions["p17r"][2]
            p19_threshold = float(
                variant_configs["p17r"]["model"]["forward_selection_threshold"]
            )
            p19_states, p19_probabilities, _ = sparse_hazard_veto_states(
                p17r_occupancy, bool(bundle["row"]["hazardous"]), p19_threshold
            )
            state_probability = {
                "p17": predictions["p17"][:2],
                "p17r": predictions["p17r"][:2],
                "p19": (p19_states, p19_probabilities),
            }
            baseline_attribution = None
            for name in ("p17", "p17r", "p19"):
                states, probabilities = state_probability[name]
                row = apply_completion_policy_true_first_return(
                    bundle["row"],
                    bundle["diagnostics"],
                    states,
                    probabilities,
                    compiler,
                    attribution_config,
                    device,
                    baseline_attribution=baseline_attribution,
                )
                if baseline_attribution is None:
                    baseline_attribution = row["baseline_attribution"]
                row["scene_name"] = bundle["scene"]
                row["audited_variant"] = name
                variant_rows[name].append(row)
            if (position + 1) % 25 == 0 or position + 1 == len(bundles):
                print(
                    json.dumps(
                        {
                            "stage": "true_first_return",
                            "progress": f"{position + 1}/{len(bundles)}",
                        }
                    ),
                    flush=True,
                )
        policies = {
            name: summarize_actor_policy(rows) for name, rows in variant_rows.items()
        }
        decisions = {name: _decisions(policy) for name, policy in policies.items()}
        pareto_variants = [name for name, row in decisions.items() if all(row.values())]
        for name, rows in variant_rows.items():
            _write_jsonl(run_dir / f"SOURCE_TEST_{name.upper()}_ACTORS.jsonl", rows)
        summary = {
            "schema_version": "worldsim_v7.p20_true_first_return_audit.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": (
                "true_first_return_audit_identifies_pareto_variant"
                if pareto_variants
                else "true_first_return_audit_all_variants_rejected"
            ),
            "source_evidence_status": "consumed_diagnostic_only",
            "actor_count": len(bundles),
            "ray_operator": config["attribution"]["ray_operator"],
            "policies": policies,
            "decisions": decisions,
            "pareto_variants": pareto_variants,
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
                "phase": "audit",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return {
            "run_dir": str(run_dir),
            "verdict": summary["verdict"],
            "pareto_variants": pareto_variants,
        }
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "phase": "audit",
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    result = run_audit(args.config.resolve(), args.repo_root.resolve(), args.run_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

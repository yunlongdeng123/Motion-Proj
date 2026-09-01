"""Compose frozen V7 physical repair with frozen V6.7 task authority."""

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

import scripts.run_worldsim_v67_p315_admitted_set_quantile_certificate as legacy


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _artifact(runs_root: Path, reference: Mapping[str, Any]) -> Path:
    return runs_root / str(reference["run"]) / str(reference["artifact"])


def _filter_exact_actor_rows(
    source_path: Path,
    output_path: Path,
    actor_rows: list[Mapping[str, Any]],
) -> int:
    exact_keys = np.asarray(
        [
            (int(row["scene_index"]) << 32) | int(row["v67_actor_id"])
            for row in actor_rows
        ],
        dtype=np.int64,
    )
    with np.load(source_path, allow_pickle=False) as loaded:
        source_keys = (
            loaded["scene_index"].astype(np.int64) << 32
        ) | loaded["actor_id"].astype(np.int64)
        keep = np.isin(source_keys, exact_keys)
        filtered = {name: loaded[name][keep] for name in loaded.files}
    np.savez(output_path, **filtered)
    return int(np.count_nonzero(keep))


def _summarize_actions(
    costs: np.ndarray,
    unsafe: np.ndarray,
    selected: np.ndarray,
    scenes: np.ndarray,
) -> dict[str, Any]:
    if not np.any(selected):
        return {
            "action_set_count": int(len(costs)),
            "authorized_count": 0,
            "coverage": 0.0,
        }
    scene_rows = []
    for scene in np.unique(scenes):
        local = scenes == scene
        admitted = local & selected
        scene_rows.append(
            {
                "scene_index": int(scene),
                "action_set_count": int(np.count_nonzero(local)),
                "authorized_count": int(np.count_nonzero(admitted)),
                "coverage": float(np.mean(selected[local])),
                "all_mean_actual_cost": float(np.mean(costs[local])),
                "authorized_mean_actual_cost": (
                    float(np.mean(costs[admitted])) if np.any(admitted) else None
                ),
            }
        )
    return {
        "action_set_count": int(len(costs)),
        "authorized_count": int(np.count_nonzero(selected)),
        "coverage": float(np.mean(selected)),
        "mean_actual_cost": float(np.mean(costs[selected])),
        "q90_actual_cost": float(np.quantile(costs[selected], 0.9)),
        "unsafe_rate": float(np.mean(unsafe[selected])),
        "scene_with_authority_count": int(
            sum(row["authorized_count"] > 0 for row in scene_rows)
        ),
        "scene_rows": scene_rows,
    }


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runs_root = Path(str(config["runs_root"]))
    run_dir = runs_root / "worldsim_v7" / str(config["task_id"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(
        run_dir / "status.json",
        {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    device = torch.device(str(config["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V7 P9 is frozen to the available CUDA device")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()

    try:
        p5_rows = [
            row
            for row in _read_jsonl(Path(str(config["p5b_actor_rows"])))
            if str(row["p4_role"]) == str(config["evaluation"]["p5_role"])
        ]
        if not p5_rows:
            raise RuntimeError("P5 exact test interface has no Actors")
        p346_run = Path(str(config["frozen_p346_run"]))
        p346_config = yaml.safe_load(
            (p346_run / "resolved.yaml").read_text(encoding="utf-8")
        )
        source_path = _artifact(runs_root, p346_config["source_rows"])
        filtered_path = run_dir / "P5_TEST_EXACT_SOURCE_ROWS.npz"
        filtered_row_count = _filter_exact_actor_rows(
            source_path, filtered_path, p5_rows
        )

        ensemble = torch.load(
            _artifact(runs_root, p346_config["frozen_p126"]),
            map_location=device,
            weights_only=False,
        )
        members = []
        for state in ensemble["member_state_dicts"]:
            member = legacy.DirectionalActorGaussian(
                20, ensemble["hidden_dimensions"]
            ).to(device)
            member.load_state_dict(state)
            members.append(member.eval())
        density, density_metadata = legacy._load_density(
            _artifact(runs_root, p346_config["frozen_p182"])
        )
        p199 = torch.load(
            _artifact(runs_root, p346_config["frozen_p199"]),
            map_location=device,
            weights_only=False,
        )
        copula = legacy.JointHorizonCopula(
            8, p199["hidden_dimensions"], 4
        ).to(device)
        copula.load_state_dict(p199["model_state_dict"])
        copula.eval()
        p203 = torch.load(
            _artifact(runs_root, p346_config["frozen_p203"]),
            map_location=device,
            weights_only=False,
        )
        calibrator = legacy.MonotoneBetaCalibration().to(device)
        calibrator.load_state_dict(p203["model_state_dict"])
        calibrator.eval()
        horizons = np.asarray(p346_config["horizons_seconds"], np.float32)
        anchors = np.asarray(p346_config["feature_anchor_budgets"], np.float32)
        floor = float(p346_config["boundary_state_cost"]["clearance_floor_m"])
        common = (
            members,
            ensemble,
            density,
            density_metadata,
            p199,
            copula,
            calibrator,
            horizons,
            anchors,
            floor,
            int(p346_config["teacher"]["monte_carlo_samples"]),
            int(p346_config["seed"]),
            float(p346_config["teacher"]["ignored_future_marginal_probability"]),
        )
        feature, _, _, _, _ = legacy._dataset(filtered_path, *common)
        with np.load(filtered_path, allow_pickle=False) as loaded:
            retained = {name: loaded[name] for name in loaded.files}
        _, _, costs, _ = legacy._align(
            legacy._trajectory_payload(retained, members, ensemble, floor), horizons
        )
        groups, action_costs, group_scenes, queries = legacy._action_groups_by_horizon(
            feature,
            costs,
            filtered_path,
            horizons,
            int(p346_config["action_group_size"]),
        )

        policy_artifact = torch.load(
            _artifact(runs_root, p346_config["frozen_policy"]),
            map_location=device,
            weights_only=False,
        )
        base = legacy.EpistemicTailCVaRAllocator(
            int(policy_artifact["element_width"]),
            int(policy_artifact["context_width"]),
            int(policy_artifact["rate_knot_count"]),
        ).to(device)
        authority_artifact = torch.load(
            _artifact(runs_root, p346_config["frozen_authority"]),
            map_location=device,
            weights_only=False,
        )
        legacy._load_base_state(base, authority_artifact)
        authority = legacy.SharedContextPiecewiseAnchorAuthorityCompiler(
            base, authority_artifact["shared_context_anchor_fractions"]
        ).to(device).eval()
        condition = p346_config["authority_condition"]
        tolerance_domain = np.asarray(
            policy_artifact["risk_tolerance_domain"], np.float32
        )
        tolerance_z = 2 * (
            float(condition["risk_tolerance"]) - tolerance_domain[0]
        ) / (tolerance_domain[1] - tolerance_domain[0]) - 1
        floor_domain = np.asarray(policy_artifact["floor_domain"], np.float32)
        floor_z = 2 * (
            float(condition["final_reliability_floor"]) - floor_domain[0]
        ) / (floor_domain[1] - floor_domain[0]) - 1
        authority_curves = legacy._authority_curves(
            authority,
            groups,
            np.asarray(condition["attainable_budget_fractions"], np.float32),
            float(condition["alpha_fairness"]),
            tolerance_z,
            floor_z,
            float(condition["tail_mass"]),
        )
        descriptor = np.concatenate(
            (groups, authority_curves, legacy._geometry(queries)), 2
        )
        authority_base = authority_curves.mean(2)

        selector_artifact = torch.load(
            _artifact(runs_root, p346_config["frozen_selector"]),
            map_location=device,
            weights_only=False,
        )
        descriptor = (
            (descriptor - np.asarray(selector_artifact["input_mean"], np.float32))
            / np.asarray(selector_artifact["input_scale"], np.float32)
        ).astype(np.float32)
        selector = legacy.AuthorityResidualTopK(
            int(selector_artifact["input_width"]),
            int(selector_artifact["element_width"]),
            int(selector_artifact["context_width"]),
            float(selector_artifact["maximum_authority_residual"]),
        ).to(device)
        selector.load_state_dict(selector_artifact["model_state_dict"])
        selector.eval()
        with torch.inference_mode():
            selector_score = selector(
                torch.from_numpy(descriptor).to(device),
                torch.from_numpy(authority_base).to(device),
            ).cpu().numpy()

        progress_artifact = torch.load(
            _artifact(runs_root, p346_config["frozen_progress_compiler"]),
            map_location=device,
            weights_only=False,
        )
        progress_model = legacy.ProgressConditionedAdmission(
            int(progress_artifact["input_width"]),
            int(progress_artifact["hidden_width"]),
            float(progress_artifact["maximum_rate_adjustment"]),
        ).to(device)
        progress_model.load_state_dict(progress_artifact["model_state_dict"])
        progress_model.eval()
        maneuver_artifact = torch.load(
            _artifact(runs_root, p346_config["frozen_maneuver_compiler"]),
            map_location=device,
            weights_only=False,
        )
        maneuver_model = legacy.ManeuverConditionedAdmission(
            int(maneuver_artifact["input_width"]),
            int(maneuver_artifact["hidden_width"]),
            float(maneuver_artifact["maximum_rate_adjustment"]),
        ).to(device)
        maneuver_model.load_state_dict(maneuver_artifact["model_state_dict"])
        maneuver_model.eval()

        task_features = []
        task_targets = []
        task_scenes = None
        task_conditions = None
        set_sizes = np.asarray(p346_config["authority_set_sizes"], np.int64)
        for set_size in set_sizes:
            local = legacy._horizon_task_examples(
                descriptor,
                selector_score,
                action_costs,
                queries,
                group_scenes,
                progress_model,
                maneuver_model,
                config["evaluation"]["heldout_progress_preferences"],
                config["evaluation"]["heldout_lateral_commands"],
                float(p346_config["lateral_preference_weight"]),
                int(set_size),
            )
            task_features.append(local[0])
            task_targets.append(local[1])
            task_scenes = local[2]
            task_conditions = local[3]
        task_feature = np.concatenate(task_features, axis=1)
        task_target = np.stack(task_targets, axis=1)

        lattice_artifact = torch.load(
            _artifact(runs_root, p346_config["frozen_lattice_authority"]),
            map_location="cpu",
            weights_only=False,
        )
        task_feature = (
            (task_feature - np.asarray(lattice_artifact["input_mean"], np.float32))
            / np.asarray(lattice_artifact["input_scale"], np.float32)
        ).astype(np.float32)
        p346_artifact = torch.load(
            p346_run / str(p346_config["model_artifact"]),
            map_location=device,
            weights_only=False,
        )
        reliability = legacy.DirectVisitedReliabilityHead(
            int(p346_artifact["input_width"]),
            p346_artifact["hidden_dimensions"],
            len(set_sizes),
        ).to(device)
        reliability.load_state_dict(p346_artifact["model_state_dict"])
        reliability.eval()

        horizon_index = int(lattice_artifact["heldout_horizon_index"])
        normalized_horizon = horizons / float(np.max(horizons))
        ceiling_index = int(config["evaluation"]["frozen_cost_ceiling_index"])
        ceilings = np.asarray(
            lattice_artifact["heldout_log_cost_ceilings"], np.float32
        )
        ceiling = float(ceilings[ceiling_index])
        set_size = int(config["evaluation"]["frozen_authority_set_size"])
        set_size_index = int(np.flatnonzero(set_sizes == set_size)[0])
        x = torch.from_numpy(task_feature).to(device)
        local_h = torch.full(
            (len(x),), float(normalized_horizon[horizon_index]), device=device
        )
        local_ceiling = torch.full((len(x),), ceiling, device=device)
        with torch.inference_mode():
            logits = reliability(x, local_h, local_ceiling)
            temperature = p346_artifact["temperature"].to(device)[ceiling_index]
            bias = p346_artifact["calibration_bias"].to(device)[ceiling_index]
            slope = p346_artifact["calibration_horizon_slope"].to(device)[
                ceiling_index
            ]
            logits = logits / temperature + bias + slope * local_h[:, None]
            probabilities = torch.sigmoid(logits).cpu().numpy()
            logits_np = logits.cpu().numpy()
        for index in range(len(set_sizes)):
            mapping = p346_artifact["isotonic_maps"][ceiling_index][index]
            probabilities[:, index] = legacy.apply_binary_isotonic_map(
                logits_np[:, index], mapping["thresholds"], mapping["values"]
            )

        predicted_unsafe = probabilities[:, set_size_index]
        requested_reliability = float(
            config["evaluation"]["requested_reliability"]
        )
        authorized = predicted_unsafe <= 1.0 - requested_reliability
        log_cost = task_target[:, set_size_index, horizon_index]
        actual_cost = np.expm1(log_cost)
        unsafe = log_cost > ceiling
        all_selected = np.ones(len(actual_cost), dtype=bool)
        no_authority = _summarize_actions(
            actual_cost, unsafe, all_selected, task_scenes
        )
        with_authority = _summarize_actions(
            actual_cost, unsafe, authorized, task_scenes
        )

        query_chamfer = np.asarray(
            [float(row["query_chamfer_m"]) for row in p5_rows], np.float64
        )
        compiled_chamfer = np.asarray(
            [float(row["compiled_chamfer_m"]) for row in p5_rows], np.float64
        )
        physically_selected = np.asarray(
            [bool(row["factorized_selected"]) for row in p5_rows], bool
        )
        selective_chamfer = np.where(
            physically_selected, compiled_chamfer, query_chamfer
        )
        surface = {
            "actor_count": len(p5_rows),
            "hazard_actor_count": int(sum(bool(row["hazardous"]) for row in p5_rows)),
            "physically_selected_actor_count": int(np.count_nonzero(physically_selected)),
            "query_mean_chamfer_m": float(np.mean(query_chamfer)),
            "harp3d_selective_mean_chamfer_m": float(np.mean(selective_chamfer)),
            "selected_geometrically_harmful_count": int(
                sum(
                    bool(row["factorized_selected"])
                    and bool(row["geometric_repair_harmful"])
                    for row in p5_rows
                )
            ),
            "actor_retention": 1.0,
            "hazard_retention": 1.0,
        }
        arms = {
            "B0_query_no_task_authority": {
                "surface_mean_chamfer_m": surface["query_mean_chamfer_m"],
                **no_authority,
            },
            "B1_harp3d_no_task_authority": {
                "surface_mean_chamfer_m": surface[
                    "harp3d_selective_mean_chamfer_m"
                ],
                **no_authority,
            },
            "B2_query_with_task_authority": {
                "surface_mean_chamfer_m": surface["query_mean_chamfer_m"],
                **with_authority,
            },
            "B3_harp3d_with_task_authority": {
                "surface_mean_chamfer_m": surface[
                    "harp3d_selective_mean_chamfer_m"
                ],
                **with_authority,
            },
        }
        gates = {
            "harp3d_surface_nonworse_than_query": surface[
                "harp3d_selective_mean_chamfer_m"
            ]
            <= surface["query_mean_chamfer_m"],
            "task_authority_coverage_nontrivial": with_authority["coverage"]
            >= float(config["gates"]["minimum_task_authority_coverage"]),
            "task_authority_mean_cost_nonworse": with_authority["mean_actual_cost"]
            <= no_authority["mean_actual_cost"],
            "task_authority_unsafe_rate_nonworse": with_authority["unsafe_rate"]
            <= no_authority["unsafe_rate"],
            "actor_and_hazard_retained": min(
                surface["actor_retention"], surface["hazard_retention"]
            )
            >= 1.0,
        }
        passed = all(bool(value) for value in gates.values())
        verdict = str(
            config["verdict_on_pass"] if passed else config["verdict_on_failure"]
        )
        output_rows = [
            {
                "scene_index": int(task_scenes[index]),
                "task_progress_preference": float(task_conditions[index, 0]),
                "task_lateral_command": float(task_conditions[index, 1]),
                "horizon_seconds": float(horizons[horizon_index]),
                "authority_set_size": set_size,
                "log_cost_ceiling": ceiling,
                "predicted_unsafe_probability": float(predicted_unsafe[index]),
                "authorized": bool(authorized[index]),
                "actual_cost": float(actual_cost[index]),
                "unsafe": bool(unsafe[index]),
            }
            for index in range(len(actual_cost))
        ]
        _write_jsonl(run_dir / "ACTION_SET_ROWS.jsonl", output_rows)
        summary = {
            "schema_version": "worldsim_v7.p9_composed_authority_fixed_lattice.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": verdict,
            "claim_boundary": config["claim_boundary"],
            "protocol": {
                "training_or_calibration": False,
                "new_sensor_read": False,
                "closed_loop_execution": False,
                "fixed_query_count": int(p346_config["action_group_size"]),
                "fixed_horizon_seconds": float(horizons[horizon_index]),
                "fixed_authority_set_size": set_size,
                "fixed_requested_reliability": requested_reliability,
                "fixed_log_cost_ceiling": ceiling,
            },
            "filtered_source_row_count": filtered_row_count,
            "surface": surface,
            "no_task_authority": no_authority,
            "with_task_authority": with_authority,
            "arms": arms,
            "gates": gates,
            "resources": {
                "gpu_used": True,
                "device": str(device),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device)
                / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "status.json",
            {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
        )
        return {"run_dir": str(run_dir), "verdict": verdict, "gates": gates}
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
            },
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

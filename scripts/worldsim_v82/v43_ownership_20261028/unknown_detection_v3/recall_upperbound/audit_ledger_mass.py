"""台账独立质量/阈值审计；fixed broad内机械贡献，不是真车影GT。"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[name] = "2"

import argparse
import json
from pathlib import Path

import numpy as np

from rule_miss_ledger import WEIGHT_COLUMNS, dense_weights, discover, tri_and, tri_or

SCOPE = "fixed broad待查域的机械贡献，包含道路/背景；不是真车影GT，不是独立语义recall；各gate集合重叠，不能相加"


def mass_row(mask, weights):
    return {"rows": int(mask.sum()), "broad_alpha_mass": float(weights[mask, 1].sum(dtype=np.float64)),
            "old_domain_alpha_mass": float(weights[mask, 0].sum(dtype=np.float64)),
            "body_alpha_mass": float(weights[mask, 2].sum(dtype=np.float64)),
            "effect_alpha_mass": float(weights[mask, 3].sum(dtype=np.float64))}


def visibility_split(initial, after, epsilon):
    result = {}
    for col, name in enumerate(WEIGHT_COLUMNS):
        present = after[:, col] > epsilon
        newly = present & (initial[:, col] <= epsilon)
        exact = newly & (initial[:, col] == 0)
        crossing = newly & (initial[:, col] > 0)
        assert int(newly.sum()) == int(exact.sum() + crossing.sum())
        result[name] = {"newly_above_epsilon": int(newly.sum()),
                        "initial_mass_numerically_exact_zero": int(exact.sum()),
                        "initial_positive_mass_at_or_below_epsilon": int(crossing.sum()),
                        "newly_after_alpha_mass": float(after[newly, col].sum(dtype=np.float64)),
                        "interpretation": "numerical zero is not proof of physical invisibility; threshold crossing is not completely newly exposed geometry"}
    return result


def audit(ledger, campaign, exact):
    contract = json.loads((ledger / "contract.json").read_text())
    summary = json.loads((ledger / "summary.json").read_text())
    n, epsilon = contract["static_rows"], contract["epsilon"]
    assert contract["background_veto"] is False
    mass_reports, visibility_reports, checks, env_reports = [], [], [], []
    queries, _ = discover(campaign, [q["raw"] for q in summary["queries"]])
    byraw = {q["raw"]: q for q in summary["queries"]}
    for query in queries:
        raw = query["raw"]
        old_report = byraw[raw]
        assert old_report["formula_membership_audit"]["known_formula_disagrees_original_membership"] == 0
        assert old_report["formula_membership_audit"]["formula_unknown_missing_fields"] == 0
        initial, _ = dense_weights(query["initial"], n)
        with np.load(ledger / ("%03d" % raw) / "rule_relaxations.npz") as f:
            variants = {k: f[k] for k in f.files if k not in ["ids", "original_formula_state"]}
        original = variants["original_selected"]
        for variant in variants.values():
            assert np.all(variant[original])
        stage_mass, stage_visibility = [], []
        for stage in query["stages"]:
            after, removed = dense_weights(stage["path"], n)
            remaining = after[:, 1] > epsilon
            ids = np.flatnonzero(remaining)
            with np.load(ledger / ("%03d" % raw) / (stage["name"] + "_misses.npz")) as saved:
                assert np.array_equal(ids, saved["ids"])
                assert np.array_equal(initial[ids], saved["initial_weights"])
                assert np.array_equal(after[ids], saved["after_weights"])
                gate_names = list(saved["gate_names"])
                gate_states = saved["gate_states"]
            assert not (removed & (after[:, -1] > 1e-8)).any()
            cohort = remaining & (initial[:, 0] > epsilon) & ~original
            failed = []
            local_cohort = cohort[ids]
            for j, gate in enumerate(gate_names):
                local = local_cohort & (gate_states[:, j] == 0)
                failed.append({"gate": str(gate), **mass_row(local, after[ids])})
            marginal = []
            for name, variant in variants.items():
                if not name.startswith("drop_"):
                    continue
                gained = variant & ~original
                marginal.append({"gate": name[5:], "global_marginal_added_rows": int(gained.sum()),
                                 **mass_row(remaining & gained, after)})
            stage_mass.append({"stage": stage["name"], "remaining": mass_row(remaining, after),
                               "failed_gate_multilabel_sorted_by_broad_mass": sorted(failed, key=lambda x: -x["broad_alpha_mass"]),
                               "single_gate_marginal_sorted_by_broad_mass": sorted(marginal, key=lambda x: -x["broad_alpha_mass"]),
                               "note": "failed condition is not necessarily the sole cause; marginal means only this gate was relaxed while the OR/AND formula stayed fixed"})
            split = visibility_split(initial, after, epsilon)
            stage_visibility.append({"stage": stage["name"], "regions": split,
                "scope_missed_at_initial_selection": int((remaining & (initial[:, 0] <= epsilon)).sum()),
                "scope_missed_at_this_stage": int((remaining & (after[:, 0] <= epsilon)).sum()),
                "newly_visible_in_any_target_region": int((remaining & (((initial[:, 0] <= epsilon) & (after[:, 0] > epsilon)) |
                    ((initial[:, 2] <= epsilon) & (after[:, 2] > epsilon)) | ((initial[:, 3] <= epsilon) & (after[:, 3] > epsilon)))).sum())})
        mass_reports.append({"raw": raw, "scope": SCOPE, "stages": stage_mass})
        visibility_reports.append({"raw": raw, "stages": stage_visibility})
        env_path = exact / ("%03d_environment_texels.npz" % raw)
        if env_path.exists():
            with np.load(env_path) as env:
                integrals = env["weights"].astype(np.float64).T @ env["rgb"].astype(np.float64)
                actual = env["actual_rgb_integral"].astype(np.float64)
                error = float(np.max(np.abs(integrals - actual) / np.maximum(1, np.abs(actual))))
                assert error < .003
                env_reports.append({"raw": raw, "row_namespace": "environment_texels; not original_static_rows",
                    "source_file": str(env_path), "texels": len(env["flat_ids"]), "negative_rgb_texels": int((env["rgb"] < 0).any(1).sum()),
                    "integral_relative_error": error, "weighted_original_negative_channel_integral": (env["weights"].astype(np.float64).T @ np.minimum(env["rgb"], 0).astype(np.float64)).tolist(),
                    "environment_modified": False, "labels_for_training": False})
        checks.append({"raw": raw, "static_rows": n, "old_formula_exact_replay_mismatch": 0,
                       "stage_weight_and_ids_exact_replay": True, "all_relaxations_preserve_old_selection": True,
                       "all_invariants_passed": True})
    (ledger / "gate_relaxation_mass_audit.json").write_text(json.dumps({"schema": "recall-gate-marginal-mass-audit-v1", "scope": SCOPE,
                    "alpha_mass_units": "sum of native alpha*T over pixels in the named domain", "queries": mass_reports}, indent=2), encoding="utf-8")
    (ledger / "visibility_by_roi.json").write_text(json.dumps({"schema": "post-suppression-threshold-and-visibility-audit-v1",
                    "scope": SCOPE, "epsilon": epsilon, "queries": visibility_reports}, indent=2), encoding="utf-8")
    (ledger / "environment_texel_audit.json").write_text(json.dumps({"schema": "original-environment-texel-source-audit-v1",
                    "queries": env_reports, "scope": "independent mechanical environment source; never car/shadow semantic GT"}, indent=2), encoding="utf-8")
    verification = {"schema": "recall-rule-ledger-verification-v1", "all_invariants_passed": True,
                    "queries": checks, "background_veto": False, "labels_for_training": False, "scope": SCOPE}
    (ledger / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--exact-gates", type=Path, required=True)
    a = p.parse_args()
    audit(a.ledger, a.campaign, a.exact_gates)

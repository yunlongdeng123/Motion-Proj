"""Independent read-only standard-library review of the new long-window files.

Does not import Torch, the generator, training, a launcher or a GPU API. It writes
only this reviewer directory; source/config files are evidence, never edited.
"""
import argparse
import ast
import datetime
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def expand_harness(text, variant=False):
    text = text.replace("30300", "32000")
    text = text.replace("NODES = (30001, 30010, 30050, 30100, 30200, 32000)", "NODES = (30001, 30300, 31000, 32000)")
    text = text.replace("PLAN = BASE / 'campaign/vgpu_bugfix_20261012/component_diagnostic/continuation_plan'", "COMPONENT_PLAN = BASE / 'campaign/vgpu_bugfix_20261012/component_diagnostic/continuation_plan'\nPLAN = BASE / 'campaign/vgpu_bugfix_20261012/loss_audit/long_pair_30k_32000/materialized_plan'")
    for old, new in (
        ("PLAN.parent / 'cpu_audit.json'", "COMPONENT_PLAN.parent / 'cpu_audit.json'"),
        ("sha(PLAN / 'runtime_safety_parameterized.py')", "sha(COMPONENT_PLAN / 'runtime_safety_parameterized.py')"),
        ("sys.path.insert(0, str(PLAN))", "sys.path.insert(0, str(COMPONENT_PLAN))"),
        ("PLAN.parent / 'scripts'", "COMPONENT_PLAN.parent / 'scripts'"),
    ):
        text = text.replace(old, new)
    if variant:
        text = text.replace("all300", "all2000").replace("every one of300", "every one of2000")
    return text


def expand_launcher(text, variant=False):
    text = text.replace("30300", "32000")
    text = text.replace("campaign/vgpu_bugfix_20261012/component_diagnostic/continuation_plan", "campaign/vgpu_bugfix_20261012/loss_audit/long_pair_30k_32000/materialized_plan")
    if variant:
        text = text.replace("campaign/vgpu_bugfix_20261012/loss_audit/controlled_opacity_outside/experiment_binding.json", "campaign/vgpu_bugfix_20261012/loss_audit/long_pair_30k_32000/materialized_plan/experiment_binding.json")
    return text


def function_text(text, name):
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return "\n".join(text.splitlines()[node.lineno - 1:node.end_lineno])
    raise ValueError("Function absent: " + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long-dir", type=Path)
    parser.add_argument("--prepared-b1", type=Path)
    parser.add_argument("--prepared-b2", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    audit = here.parents[1]
    long = args.long_dir or audit / "loss_audit/long_pair_30k_32000"
    materialized = long / "materialized_plan"
    component = audit / "component_diagnostic/continuation_plan"
    loss = audit / "loss_audit"
    manifest = load(materialized / "materialization_receipt.json")
    draft = load(long / "long_pair_plan_draft.json")
    short_binding = load(loss / "controlled_opacity_outside/experiment_binding.json")
    checks = []

    def check(label, condition):
        checks.append({"check": label, "PASS": bool(condition)})
        if not condition:
            raise AssertionError(label)

    protected = {
        component / "continuation_harness.py": "ac9745d5c0cbab7701ae8be5beec9587d57ea79288a3167259d0af1d0e164458",
        component / "continuation_variant_harness.py": "3a2f9add3cab0763c4b0ef612fd5b735afd7f2d337b151306da1a4ceffcb6154",
        component / "launch_controlled_continuation.py": "9e25b61f09f2c59c5d58bd3753792b501b9bfb12b8846ef2f1b6782cb19372b9",
        component / "launch_opacity_outside_continuation.py": "426f99b3b1a9c75ffcf8e225933065c5fa1ca898dfa566aa7b5e42b6715a94dc",
        component / "runtime_safety_parameterized.py": "f60faca00ec0d5b10b80d14d3e8ca07125ada5bff06d1f0fb8c43c5fec6d42d1",
        component.parent / "scripts/component_diagnostic.py": "0170283337ce3e2a1ea7a016e6f8a8e2b534101eebfe7b3ba4ecdcbcc1903baa",
        component / "prepared_plan.json": "f12d6c0801997d0a09020617c7629f5f41abfab4083ebbf0400f1e715be6d205",
        component / "opacity_outside_prepared_plan.json": "be9e385b37805457d2d9986985c9b7bd65af3fcde66445d3f15d156db5c90a93",
        loss / "outside_experiment_guard.py": "f3a86b6fb128b83c6a189b95787ed001da2ac84a4444ecc68bba283fae5477f9",
        loss / "controlled_opacity_outside/experiment_binding.json": "b96ec92c0694a251e39da85a3ae4abab131366a75ad52cd47098ec0df953b214",
        loss / "candidates/opacity_outside/train.py": "ff82c0d86b385bcdf28feff550163cd3ec9c7bec92e939306330ba4af335e99a",
        loss / "candidates/opacity_outside/utils/paper_losses.py": "3832a7ced1387e1ff33f5e3aee0b56bc533303f80f5be840269236139fc866ab",
    }
    check("All12 protected short-window source/config/helper bytes remain frozen", all(sha(path) == expected for path, expected in protected.items()))
    check("All17 actually materialized proposed files match receipt SHA/size", len(manifest["files"]) == 17 and all(sha(materialized / name) == row["sha256"] and (materialized / name).stat().st_size == row["bytes"] for name, row in manifest["files"].items()))
    check("No new algorithm repository clone or GPU run asserted by materializer", manifest["repository_clones_created"] == 0 and manifest["GPU_used"] is False and manifest["allow_training"] is False)
    before_b = (component / "continuation_harness.py").read_text(encoding="utf-8")
    before_v = (component / "continuation_variant_harness.py").read_text(encoding="utf-8")
    expected_b = expand_harness(before_b)
    expected_v = expand_harness(before_v, True)
    long_path = "campaign/vgpu_bugfix_20261012/loss_audit/long_pair_30k_32000/materialized_plan"
    expected = {
        "continuation_harness.py": expected_b,
        "continuation_variant_harness.py": expected_v,
        "launch_controlled_continuation.py": expand_launcher((component / "launch_controlled_continuation.py").read_text(encoding="utf-8")),
        "launch_opacity_outside_continuation.py": expand_launcher((component / "launch_opacity_outside_continuation.py").read_text(encoding="utf-8"), True),
        "outside_experiment_guard.py": (loss / "outside_experiment_guard.py").read_text(encoding="utf-8").replace('(\"stop_iteration\", 30300)', '(\"stop_iteration\", 32000)'),
    }
    for name in ("continuation_harness.py", "launch_controlled_continuation.py"):
        expected["baseline_noise_reference/" + name] = expected[name].replace(long_path, long_path + "/baseline_noise_reference").replace("original30k_to32000", "original30k_to32000_noise_reference")
    check("All7 actual harness/launcher/guard sources equal only independently allowed expansion/routing edits", all((materialized / name).read_text(encoding="utf-8") == text for name, text in expected.items()))
    actual_py = {str(path.relative_to(materialized)).replace("\\", "/") for path in materialized.rglob("*.py")}
    check("Exactly9 small source files, no hidden third algorithm override", actual_py == set(expected) | {"algorithm_overrides/train.py", "algorithm_overrides/utils/paper_losses.py"})
    check("Candidate two algorithm overrides exactly equal reviewed short-window bytes", all((materialized / "algorithm_overrides" / name).read_bytes() == (loss / "candidates/opacity_outside" / name).read_bytes() for name in ("train.py", "utils/paper_losses.py")))
    check("Baseline and candidate evaluation functions retain exact short-window source", function_text(expected_b, "diagnostic_eval") == function_text(before_b, "diagnostic_eval") and function_text(expected_v, "diagnostic_eval") == function_text(before_v, "diagnostic_eval"))
    check("Candidate exact-before-real-backward gate retains identical function source", function_text(expected_v, "before_first_backward") == function_text(before_v, "before_first_backward"))
    configurations = {
        "B1": load(materialized / "baseline_effective_config.yaml"),
        "B2": load(materialized / "baseline_noise_reference/effective_config.yaml"),
        "V": load(materialized / "outside_effective_config.yaml"),
    }
    parent = short_binding["parent_training_config"]
    extras = {"diagnostic_background_domain_complement": True, "diagnostic_lambda_dynamic_background_opacity": .05}
    for role, config in configurations.items():
        excluded = {"model_path", "start_checkpoint"} | (set(extras) if role == "V" else set())
        check(role + " complete original config values/types are unchanged except routing and the2 V fields", set(config) == set(parent) | (set(extras) if role == "V" else set()) and all(type(config[key]) is type(value) and config[key] == value for key, value in parent.items() if key not in excluded) and config["start_checkpoint"] == short_binding["parent_static_checkpoint"] and (role != "V" or all(type(config[key]) is type(value) and config[key] == value for key, value in extras.items())))
    check("B2 config differs from B1 only in independent output path", {k: v for k, v in configurations["B2"].items() if k != "model_path"} == {k: v for k, v in configurations["B1"].items() if k != "model_path"})
    check("All arms fresh30k, 2000steps, horizon50k; no reset/densify/SGCM before32000", draft["parent_iteration"] == 30000 and draft["first_iteration"] == 30001 and draft["stop_iteration"] == 32000 and draft["steps"] == 2000 and draft["reuse30300_terminal_selected"] is False and all(c["iterations"] == c["position_lr_max_steps"] == 50000 and c["opacity_reset_interval"] == 3000 and c["densify_until_iter"] == c["dynamic_densify_until_iter"] == c["sgcm_end"] == 15000 for c in configurations.values()) and not any(i % 3000 == 0 for i in range(30001, 32001)))
    protocol = load(materialized / "evaluation_protocol_locked.json")
    check("Locked full49 three-trajectory protocol and four node schedule are unchanged", sha(materialized / "evaluation_protocol_locked.json") == "c725cd2153d1104ddaf81078ac753f4e7f474e0be325ca9fbb5b1fccb342f658" and protocol["nodes"] == [30001, 30300, 31000, 32000] and set(protocol["three_trajectories_if_window_is_authorized"]) == {"B1", "B2", "V"})
    check("V draft explicitly leaves real long baseline/binding/pair references missing", load(materialized / "outside_cpu_plan_draft.json")["first_step_reference"] is None and load(materialized / "outside_cpu_plan_draft.json")["all_iteration_pair_reference"] is None and load(materialized / "outside_cpu_plan_draft.json")["experiment_binding_sha256"] is None)
    prepared = {}
    for role, path in (("B1", args.prepared_b1), ("B2", args.prepared_b2)):
        if path:
            plan = load(path)
            filename = "continuation_harness.py" if role == "B1" else "baseline_noise_reference/continuation_harness.py"
            check(role + " actual CPU prepared plan binds actual harness/config and all49 endpoint", plan["status"] == "prepared_not_started" and plan["harness_sha256"] == sha(materialized / filename) and plan["first_iteration"] == 30001 and plan["stop_iteration"] == 32000 and plan["steps"] == 2000 and plan["horizon"] == plan["position_lr_max_steps"] == 50000 and plan["seed"] == 0 and plan["training_config"] == configurations[role] and plan["training_config_changed_fields"] == ["model_path", "start_checkpoint"] and plan["extra_eval_nodes"] == [30001, 30300, 31000, 32000] and plan["terminal_eval_camera_count"] == 49 and plan["allow_training"] is False and plan["algorithm_changes"] == [])
            prepared[role] = {"path": str(path), "sha256": sha(path), "harness_sha256": plan["harness_sha256"], "protected_input_count": len(plan["original_file_sha256"])}
    receipt = {
        "status": "PASS_CPU_SOURCE_CONFIG_REVIEW_NO_GPU_AUTHORIZATION",
        "reviewed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "checks": checks,
        "source_materialization_receipt_sha256": sha(materialized / "materialization_receipt.json"),
        "long_pair_plan_draft_sha256": sha(long / "long_pair_plan_draft.json"),
        "loss_agent_locked_protocol_sha256": sha(materialized / "evaluation_protocol_locked.json"),
        "independent_interpretation_sha256": sha(here / "interpretation_preregistered.json"),
        "materialized_file_sha256": {name: sha(materialized / name) for name in manifest["files"]},
        "actual_prepared_original_arms": prepared,
        "runtime_model_Adam_CUDA_RNG_restore_checked": False,
        "V_launchability": "Not launchable; actual long B1/B2 trajectories, common49 and independent B1/B2 noise receipt must exist and enter a new binding/CPU plan/review/root receipt.",
        "B1_B2_run_authorization": "Only exact separate root receipts and exclusive GPU handoff can authorize a run; this reviewer uses CPU only.",
        "GPU_used": False,
        "current_short_candidate_source_changed": False,
        "reviewer_source_sha256": sha(__file__),
    }
    output = args.output or here / "materialized_review_cpu.json"
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "checks": len(checks), "prepared_original_arms": list(prepared), "receipt": str(output), "receipt_sha256": sha(output)}))


if __name__ == "__main__":
    main()

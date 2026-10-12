"""CPU-only replica/config/guard contract preparation; deliberately no launch."""
import argparse
import copy
import difflib
import json
from pathlib import Path
import shutil

from outside_experiment_guard import ALGORITHM_FILES, EXTRA_CONFIG, OutsideExperimentGuard, sha, source_inventory


BASE = Path("/root/autodl-tmp/drivingeditor_repro")
AUDIT = BASE / "campaign/vgpu_bugfix_20261012/loss_audit"
COMPONENT = BASE / "campaign/vgpu_bugfix_20261012/component_diagnostic/continuation_plan"
ORIGINAL = BASE / "campaign/resume_vgpu_baseline_20261011/frozen_repository"
CONTROL = AUDIT / "controlled_opacity_outside"
EXPECTED_BASELINE_HARNESS = "ac9745d5c0cbab7701ae8be5beec9587d57ea79288a3167259d0af1d0e164458"
EXPECTED_BASELINE_PLAN = "f12d6c0801997d0a09020617c7629f5f41abfab4083ebbf0400f1e715be6d205"
EXPECTED_BASELINE_OBSERVER = "f60faca00ec0d5b10b80d14d3e8ca07125ada5bff06d1f0fb8c43c5fec6d42d1"
EXPECTED_COMPONENT_HELPER = "0170283337ce3e2a1ea7a016e6f8a8e2b534101eebfe7b3ba4ecdcbcc1903baa"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-protected-files", action="store_true")
    args = parser.parse_args()
    from omegaconf import OmegaConf
    for name, expected in (("continuation_harness.py", EXPECTED_BASELINE_HARNESS), ("prepared_plan.json", EXPECTED_BASELINE_PLAN), ("runtime_safety_parameterized.py", EXPECTED_BASELINE_OBSERVER)):
        if sha(COMPONENT / name) != expected:
            raise ValueError("Final reviewed baseline harness contract changed: " + name)
    helper = COMPONENT.parent / "scripts/component_diagnostic.py"
    if sha(helper) != EXPECTED_COMPONENT_HELPER:
        raise ValueError("Shared RNG/render/region diagnostic helper changed")
    protocol = BASE / "campaign/vgpu_bugfix_20261012/opacity_pair_protocol.json"
    protocol_json = json.loads(protocol.read_text())
    if protocol_json["coefficient"] != 0.05 or protocol_json["steps"] != [30001, 30300] or protocol_json["lr_horizon"] != 50000:
        raise ValueError("Root preregistered paired protocol invariants differ")
    baseline_plan = json.loads((COMPONENT / "prepared_plan.json").read_text())
    if baseline_plan["algorithm_changes"] or baseline_plan["repository"] != str(ORIGINAL):
        raise ValueError("Expected unchanged original baseline plan")
    parent = Path(baseline_plan["parent"])
    parent_config = OmegaConf.to_container(OmegaConf.load(parent / "effective_config.yaml"), resolve=True)
    baseline_output = Path(baseline_plan["output"])
    baseline_report_path = baseline_output / "continuation_diagnostic.json"
    baseline_status_path = baseline_output / "status.json"
    baseline_snapshot_path = baseline_output / "terminal_30300/diagnostic_snapshot_manifest.json"
    baseline_steps_path = baseline_output / "step_losses.jsonl"
    baseline_report = json.loads(baseline_report_path.read_text())
    baseline_status = json.loads(baseline_status_path.read_text())
    baseline_snapshot = json.loads(baseline_snapshot_path.read_text())
    baseline_steps = [json.loads(line) for line in baseline_steps_path.read_text().splitlines() if line]
    if baseline_status["status"] != "complete" or baseline_status["actual_exit_code"] != 0 or baseline_status["gpu_lock_held"]:
        raise ValueError("Original baseline must finish exit0 and release GPU lock")
    if not baseline_report["all_node_eval_rng_preserved"] or baseline_report["restore"]["status"] != "actual_restore_pass" or not baseline_report["terminal_full49_evaluation"] or not baseline_report["genuine_full_loss_backward_Adam"]:
        raise ValueError("Baseline restore/real train/RNG/terminal49 evidence failed")
    if baseline_snapshot["status"] != "complete" or baseline_snapshot["iteration"] != 30300 or set(baseline_snapshot["checkpoints"]) != {"static", "dynamic", "environment", "training_state"} or not all(row["cpu_load_iteration_schema_validated"] for row in baseline_snapshot["checkpoints"].values()):
        raise ValueError("Baseline terminal complete four-checkpoint snapshot required")
    if [row["iteration"] for row in baseline_steps] != list(range(30001, 30301)):
        raise ValueError("Baseline needs exactly300 consecutive true original steps")
    reference_keys = sorted(baseline_plan["first_step_reference"]["values"])
    if len(reference_keys) != 10:
        raise ValueError("Exactly original10 first-step scalar reference required")
    first_values = {key: baseline_steps[0]["losses"][key] for key in reference_keys}
    baseline_auxiliary = {str(path): sha(path) for path in (baseline_report_path, baseline_status_path, baseline_snapshot_path, baseline_steps_path)}
    source = source_inventory(ORIGINAL)
    replica = CONTROL / "frozen_repository"
    if not replica.exists():
        replica.mkdir(parents=True, exist_ok=False)
        for relative in source:
            target = replica / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ORIGINAL / relative, target)
        for relative in sorted(ALGORITHM_FILES):
            shutil.copyfile(AUDIT / "candidates/opacity_outside" / relative, replica / relative)
    inventory = source_inventory(replica)
    if set(source) != set(inventory) or {k for k in source if source[k] != inventory[k]} != ALGORITHM_FILES:
        raise ValueError("Replica must have exactly reviewed two-file algorithm diff")
    for relative in ALGORITHM_FILES:
        if sha(replica / relative) != sha(AUDIT / "candidates/opacity_outside" / relative):
            raise ValueError("Replica differs from CPU-tested candidate: " + relative)
    config = copy.deepcopy(baseline_plan["training_config"])
    output = BASE / "campaign/vgpu_bugfix_20261012/continuation_diagnostic/opacity_outside30k_to30300"
    if output.exists():
        raise ValueError("Hypothesis output unexpectedly exists before launch")
    config["model_path"] = str(output)
    config.update(EXTRA_CONFIG)
    binding = {
        "schema_version": 1,
        "status": "source_config_guard_prepared_pending_shared_harness_binding",
        "training_started": False,
        "allow_training": False,
        "experiment_kind": "hypothesis_complement_background_domain",
        "scientific_status": "Complement missing background domain as explicit hypothesis; NOT certified author formula or strict original config",
        "parent_iteration": 30000, "first_iteration": 30001, "stop_iteration": 30300,
        "steps": 300, "horizon": 50000, "seed": 0,
        "output": str(output), "repository": str(replica), "original_repository": str(ORIGINAL),
        "parent_static_checkpoint": str(parent / "chkpnt30000.pth"),
        "parent_training_config": parent_config, "training_config": config,
        "added_training_fields": EXTRA_CONFIG,
        "training_config_changed_original_fields": ["model_path", "start_checkpoint"],
        "training_config_added_fields": sorted(EXTRA_CONFIG), "strict_original_config": False,
        "algorithm_changed_files": sorted(ALGORITHM_FILES),
        "algorithm_patch": str(AUDIT / "candidates/opacity_outside.patch"),
        "algorithm_patch_sha256": sha(AUDIT / "candidates/opacity_outside.patch"),
        "original_source_sha256": source, "replica_source_sha256": inventory,
        "protected_original_file_sha256": baseline_plan["original_file_sha256"],
        "baseline_binding": {"plan_path": str(COMPONENT / "prepared_plan.json"), "plan_sha256": EXPECTED_BASELINE_PLAN, "harness_path": str(COMPONENT / "continuation_harness.py"), "harness_sha256": EXPECTED_BASELINE_HARNESS, "observer_path": str(COMPONENT / "runtime_safety_parameterized.py"), "observer_sha256": EXPECTED_BASELINE_OBSERVER, "output": baseline_plan["output"]},
        "guard_path": str(AUDIT / "outside_experiment_guard.py"), "guard_sha256": sha(AUDIT / "outside_experiment_guard.py"),
        "auxiliary_file_sha256": {str(helper): EXPECTED_COMPONENT_HELPER, str(protocol): sha(protocol), str(AUDIT / "outside_experiment_guard.py"): sha(AUDIT / "outside_experiment_guard.py"), str(COMPONENT / "continuation_harness.py"): EXPECTED_BASELINE_HARNESS, str(COMPONENT / "runtime_safety_parameterized.py"): EXPECTED_BASELINE_OBSERVER, str(COMPONENT / "prepared_plan.json"): EXPECTED_BASELINE_PLAN},
        "paired_protocol": {"path": str(protocol), "sha256": sha(protocol), "preregistered_before_candidate_outcome": True, "original_protocol": protocol_json},
        "runtime_helper_import_contract": {"component_diagnostic": {"path": str(helper), "sha256": EXPECTED_COMPONENT_HELPER, "functions": ["stats", "rgba_package", "region_stats", "rng_capture", "rng_restore", "rng_pins"]}, "runtime_safety_parameterized": {"path": str(COMPONENT / "runtime_safety_parameterized.py"), "sha256": EXPECTED_BASELINE_OBSERVER}},
        "evaluation": {"nodes": baseline_plan["extra_eval_nodes"], "early_fixed_camera_ids": baseline_plan["early_fixed_camera_ids"], "terminal_eval_camera_count": baseline_plan["terminal_eval_camera_count"], "all_extra_eval_rng_preserved": True},
        "restore_policy": baseline_plan["restore_policy"],
        "restore_verification_status": "CPU contract validation only; no model loaded, no CUDA RNG restore performed; actual shared GPU restore observer remains required",
        "baseline_completion": {"status": "complete", "actual_exit_code": 0, "restore_status": "actual_restore_pass", "terminal49_and_four_validated_checkpoints": True, "metadata_file_sha256": baseline_auxiliary, "baseline_report_path": str(baseline_report_path)},
        "first_step_pair_reference": {"iteration": 30001, "values": first_values, "original_objective_log_key": "loss_original_objective_before_diagnostic", "comparison_phase": "Before actual first Tensor.backward; after rendering/original compose and explicit original-objective capture, before any optimizer update", "policy": "All10 original loss/PSNR scalars must equal actual original baseline30001 exactly; compare original captured total, never recover via subtraction; later losses may differ", "source_path": str(baseline_steps_path), "source_sha256": sha(baseline_steps_path)},
        "all_iteration_pair_reference": {"rows": [{key: row[key] for key in ("iteration", "camera_id", "time_shift", "learning_rates")} for row in baseline_steps], "source_path": str(baseline_steps_path), "source_sha256": sha(baseline_steps_path), "policy": "Exact all300 iteration/camera/time_shift/LR equality; all extra evaluation restores RNG"},
        "extra_eval_rng_policy": baseline_plan["extra_eval_rng_policy"],
        "harness_integration_status": "Reuse final reviewed harness after minimum explicit parameterization; no copied legacy harness and no launchable variant generated yet",
        "launch_gates": ["Original baseline completed with terminal49frames and four validated checkpoints", "Root reviewed exact variant plan/source/guard/harness pins", "Explicit GPU ownership handoff and shared gpu_slot.lock"],
        "unchanged_algorithms": ["foreground Eq13", "all original paper loss coefficients", "original backward/Adam", "camera/time_shift RNG and sampler", "50k LR horizon", "depth projection and normalization", "SGCM", "static opacity reset"],
    }
    binding["auxiliary_file_sha256"].update(baseline_auxiliary)
    guard = OutsideExperimentGuard(binding)
    checks = guard.verify_sources()
    checks.update(guard.verify_auxiliary_files())
    # Execute the actual unchanged formal validator on the stripped original
    # fields without importing project/GPU code: only its three function ASTs.
    import ast
    tree = ast.parse((ORIGINAL / "utils/paper_checkpoint.py").read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_config_dict", "_validate_resume_configuration", "_same_config_value")]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "unchanged_formal_config_validator", "exec"), namespace)
    checks["configuration"] = guard.validate_config(parent_config, config, namespace["_validate_resume_configuration"])
    if args.verify_protected_files:
        checks.update(guard.verify_protected_files())
    else:
        checks["protected_original_sha_reverification"] = "not repeated at preparation; exact baseline pins retained; runtime guard must reverify"
    CONTROL.mkdir(parents=True, exist_ok=True)
    (CONTROL / "experiment_binding.json").write_text(json.dumps(binding, ensure_ascii=False, indent=2, allow_nan=False))
    OmegaConf.save(OmegaConf.create(config), CONTROL / "effective_config.yaml")
    diff = []
    for relative in sorted(ALGORITHM_FILES):
        a = (ORIGINAL / relative).read_bytes().decode().splitlines(keepends=True)
        b = (replica / relative).read_bytes().decode().splitlines(keepends=True)
        diff.extend(difflib.unified_diff(a, b, fromfile="a/" + relative, tofile="b/" + relative))
    (CONTROL / "actual_algorithm_diff.patch").write_bytes("".join(diff).encode())
    checks.update(status=binding["status"], training_started=False, allow_training=False, CUDA_imported=False, experiment_binding_sha256=sha(CONTROL / "experiment_binding.json"), config_sha256=sha(CONTROL / "effective_config.yaml"), actual_algorithm_diff_sha256=sha(CONTROL / "actual_algorithm_diff.patch"))
    (CONTROL / "cpu_preparation.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2))
    print(json.dumps(checks, ensure_ascii=False))


if __name__ == "__main__":
    main()

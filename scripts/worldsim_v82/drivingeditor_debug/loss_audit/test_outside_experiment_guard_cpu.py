"""Meaningful reject/accept tests for isolated hypothesis contract; no CUDA."""
import argparse
import ast
import copy
import datetime
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from outside_experiment_guard import OutsideExperimentGuard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    binding = json.loads(args.binding.read_text())
    tests = []

    def check(name, run):
        result = run()
        assert result is not False, name
        tests.append({"name": name, "passed": True})

    def reject(run):
        try:
            run()
        except (ValueError, RuntimeError):
            return True
        raise AssertionError("Invalid configuration/source was accepted")

    original = Path(binding["original_repository"])
    tree = ast.parse((original / "utils/paper_checkpoint.py").read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ("_config_dict", "_same_config_value", "_validate_resume_configuration")]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "exact_unchanged_formal_config_functions", "exec"), namespace)
    validator = namespace["_validate_resume_configuration"]
    guard = OutsideExperimentGuard(binding)
    saved = binding["parent_training_config"]
    current = binding["training_config"]
    check("real_original_and_replica_inventory_sha", guard.verify_sources)
    check("real_root_protocol_and_helper_auxiliary_sha", guard.verify_auxiliary_files)
    check("accept_parent30k_plus_only_exact_two_experiment_fields", lambda: guard.validate_config(saved, current, validator))
    check("accept_saved_same_bound_variant_same_coefficient", lambda: guard.validate_config(current, current, validator))
    check("formal_validator_still_rejects_unstripped_experiment_fields", lambda: reject(lambda: validator(saved, current)))
    mutations = {
        "reject_unlisted_new_field": ("extra_anything", 1),
        "reject_zero_background_weight": ("diagnostic_lambda_dynamic_background_opacity", 0.0),
        "reject_changed_background_weight": ("diagnostic_lambda_dynamic_background_opacity", 0.1),
        "reject_weight_bool_type": ("diagnostic_lambda_dynamic_background_opacity", True),
        "reject_hypothesis_flag_integer_type": ("diagnostic_background_domain_complement", 1),
        "reject_hypothesis_flag_false": ("diagnostic_background_domain_complement", False),
        "reject_changed_horizon": ("iterations", 30300),
        "reject_float_horizon_type": ("iterations", 50000.0),
        "reject_changed_velocity_lr": ("velocity_lr", 0.002),
        "reject_changed_beta_lr": ("scaling_t_lr", 0.004),
        "reject_changed_reset_schedule": ("opacity_reset_interval", 6000),
        "reject_changed_depth_policy": ("depth_blend_mode", 1),
        "reject_changed_sgcm": ("paper_sgcm_enabled", False),
        "reject_changed_source_data": ("source_path", "/tmp/not_original_scene"),
        "reject_original_foreground_coefficient_change": ("paper_lambda_dynamic_opacity", 0.1),
        "reject_output_directory_change": ("model_path", "/tmp/not_bound_output"),
        "reject_parent_checkpoint_change": ("start_checkpoint", "/tmp/not_saved30k.pth"),
    }
    for name, (key, value) in mutations.items():
        changed = copy.deepcopy(current)
        changed[key] = value
        check(name, lambda changed=changed: reject(lambda: guard.validate_config(saved, changed, validator)))
    changed = copy.deepcopy(current)
    del changed["diagnostic_background_domain_complement"]
    check("reject_missing_explicit_hypothesis_flag", lambda: reject(lambda: guard.validate_config(saved, changed, validator)))
    changed_saved = copy.deepcopy(current)
    changed_saved["diagnostic_lambda_dynamic_background_opacity"] = .1
    check("reject_saved_variant_changed_coefficient", lambda: reject(lambda: guard.validate_config(changed_saved, current, validator)))
    changed_saved = copy.deepcopy(saved)
    changed_saved["iterations"] = 49999
    check("reject_parent_saved_protocol_change", lambda: reject(lambda: guard.validate_config(changed_saved, current, validator)))
    changed_binding = copy.deepcopy(binding)
    changed_binding["stop_iteration"] = 40000
    check("reject_unapproved_iteration_extension_in_binding", lambda: reject(lambda: OutsideExperimentGuard(changed_binding)))

    # Create only a private scratch replica for negative source tests. No source
    # under original or the reviewable prepared candidate is modified.
    test_root = args.out.parent / ("guard_cpu_source_test_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
    test_replica = test_root / "frozen_repository"
    test_replica.mkdir(parents=True, exist_ok=False)
    for relative in binding["replica_source_sha256"]:
        target = test_replica / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(binding["repository"]) / relative, target)
    test_binding = copy.deepcopy(binding)
    test_binding["repository"] = str(test_replica)
    test_guard = OutsideExperimentGuard(test_binding)
    file = test_replica / "utils/camera_utils.py"
    data = file.read_bytes()
    try:
        file.write_bytes(data + b"\n# forbidden depth edit\n")
        check("reject_third_algorithm_file_change", lambda: reject(test_guard.verify_sources))
    finally:
        file.write_bytes(data)
    extra = test_replica / "extra_loss.py"
    extra.write_text("unapproved = True\n")
    check("reject_extra_algorithm_file", lambda: reject(test_guard.verify_sources))
    extra.unlink()
    fake_module = SimpleNamespace(__file__=str(original / "utils/paper_checkpoint.py"), _config_dict=namespace["_config_dict"], _validate_resume_configuration=validator)
    check("cannot_install_adapter_on_actual_formal_guard_module", lambda: reject(lambda: test_guard.install_on_isolated_module(fake_module)))
    fake_module.__file__ = str(test_replica / "utils/paper_checkpoint.py")
    returned = test_guard.install_on_isolated_module(fake_module)
    check("isolated_adapter_returns_original_unchanged_validator", lambda: returned is validator)
    check("isolated_adapter_calls_original_validator_parent30k", lambda: fake_module._validate_resume_configuration(saved, current))
    check("isolated_adapter_calls_original_validator_same_saved_variant", lambda: fake_module._validate_resume_configuration(current, current))
    check("reject_double_install", lambda: reject(lambda: test_guard.install_on_isolated_module(fake_module)))
    report = {"scope": "CPU AST exact config validator and private scratch source invariants; no Torch/CUDA/model/checkpoint restore/training", "binding": str(args.binding), "tests": tests, "all_passed": all(row["passed"] for row in tests), "test_count": len(tests), "limitations": "Actual parameter/Adam/RNG/SGCM restoration remains required in shared GPU observer. Config support for same saved hypothesis does not authorize future iterations or coefficient changes."}
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: report[key] for key in ("all_passed", "test_count", "scope")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

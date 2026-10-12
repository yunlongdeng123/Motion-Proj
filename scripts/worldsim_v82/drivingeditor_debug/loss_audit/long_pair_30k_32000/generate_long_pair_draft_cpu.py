"""CPU-only reproducible expansion draft; never imports training or launches.

Reads the pinned short-window engineering sources, transforms them in memory,
and writes review diffs/metadata only. No repository, checkpoint or executable
harness copies are created. New exact plans, bindings and approvals are deferred.
"""
import argparse
import ast
import copy
import datetime
import difflib
import hashlib
import json
from pathlib import Path


BASE = "/root/autodl-tmp/drivingeditor_repro"
REMOTE_DRAFT = BASE + "/campaign/vgpu_bugfix_20261012/loss_audit/long_pair_30k_32000"
REMOTE_PLAN = REMOTE_DRAFT + "/materialized_plan"
ORIGINAL_COMPONENT_PLAN = "campaign/vgpu_bugfix_20261012/component_diagnostic/continuation_plan"
NEW_COMPONENT_PLAN = "campaign/vgpu_bugfix_20261012/loss_audit/long_pair_30k_32000/materialized_plan"
OLD_NODES = "NODES = (30001, 30010, 30050, 30100, 30200, 32000)"
NEW_NODES = "NODES = (30001, 30300, 31000, 32000)"
FIRST, STOP, NODES = 30001, 32000, [30001, 30300, 31000, 32000]

# Source v2 corrects the checked PSNR import to its actual loss_utils owner.
# Its current approval status is observed separately, never inferred from a hash.
PINS = {
    "baseline_harness": "ac9745d5c0cbab7701ae8be5beec9587d57ea79288a3167259d0af1d0e164458",
    "variant_harness": "3a2f9add3cab0763c4b0ef612fd5b735afd7f2d337b151306da1a4ceffcb6154",
    "baseline_launcher": "9e25b61f09f2c59c5d58bd3753792b501b9bfb12b8846ef2f1b6782cb19372b9",
    "variant_launcher": "426f99b3b1a9c75ffcf8e225933065c5fa1ca898dfa566aa7b5e42b6715a94dc",
    "outside_guard": "f3a86b6fb128b83c6a189b95787ed001da2ac84a4444ecc68bba283fae5477f9",
    "observer": "f60faca00ec0d5b10b80d14d3e8ca07125ada5bff06d1f0fb8c43c5fec6d42d1",
    "metric_helper": "0170283337ce3e2a1ea7a016e6f8a8e2b534101eebfe7b3ba4ecdcbcc1903baa",
    "short_binding": "b96ec92c0694a251e39da85a3ae4abab131366a75ad52cd47098ec0df953b214",
    "short_variant_plan_v2": "be9e385b37805457d2d9986985c9b7bd65af3fcde66445d3f15d156db5c90a93",
    "short_baseline_plan": "f12d6c0801997d0a09020617c7629f5f41abfab4083ebbf0400f1e715be6d205",
    "paired_protocol": "3f7d11ba7aa5fb81bb959fd3c530f8b253f8b87a9c612d3d7d644474d4c79760",
    "candidate_train": "ff82c0d86b385bcdf28feff550163cd3ec9c7bec92e939306330ba4af335e99a",
    "candidate_paper_losses": "3832a7ced1387e1ff33f5e3aee0b56bc533303f80f5be840269236139fc866ab",
    "algorithm_diff": "21388ca6b00caf94a0e869a2f22c5c506b17d846859d3b8a31623672bf9a5b49",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_bytes(data, expected):
    if sha(data) != expected:
        raise ValueError("Source/metadata pin mismatch")


def json_write(path, data):
    path.write_bytes((json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def replace(text, old, new, edits, count=None):
    actual = text.count(old)
    if not actual or (count is not None and actual != count):
        raise ValueError("Exact expansion anchor missing or ambiguous: " + repr(old))
    edits.append({"old": old, "new": new, "occurrences": actual})
    return text.replace(old, new)


def expand_harness(text, variant):
    edits = []
    text = replace(text, "30300", "32000", edits)
    text = replace(text, OLD_NODES, NEW_NODES, edits, 1)
    old = "PLAN = BASE / '" + ORIGINAL_COMPONENT_PLAN + "'"
    new = "COMPONENT_PLAN = BASE / '" + ORIGINAL_COMPONENT_PLAN + "'\nPLAN = BASE / '" + NEW_COMPONENT_PLAN + "'"
    text = replace(text, old, new, edits, 1)
    for old, new in (
        ("PLAN.parent / 'cpu_audit.json'", "COMPONENT_PLAN.parent / 'cpu_audit.json'"),
        ("sha(PLAN / 'runtime_safety_parameterized.py')", "sha(COMPONENT_PLAN / 'runtime_safety_parameterized.py')"),
        ("sys.path.insert(0, str(PLAN))", "sys.path.insert(0, str(COMPONENT_PLAN))"),
        ("PLAN.parent / 'scripts'", "COMPONENT_PLAN.parent / 'scripts'"),
    ):
        text = replace(text, old, new, edits, 1)
    if variant:
        text = replace(text, "all300", "all2000", edits)
        text = replace(text, "every one of300", "every one of2000", edits, 1)
    return text, edits


def expand_launcher(text, variant):
    edits = []
    text = replace(text, "30300", "32000", edits)
    text = replace(text, ORIGINAL_COMPONENT_PLAN, NEW_COMPONENT_PLAN, edits, 1)
    if variant:
        text = replace(text, "campaign/vgpu_bugfix_20261012/loss_audit/controlled_opacity_outside/experiment_binding.json", NEW_COMPONENT_PLAN + "/experiment_binding.json", edits, 1)
    return text, edits


def named_function(text, name):
    matching = [node for node in ast.walk(ast.parse(text)) if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matching) != 1:
        raise ValueError("Ambiguous function: " + name)
    return ast.get_source_segment(text, matching[0])


def assigned_literal(text, name):
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Assignment not found: " + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--short-receipt", type=Path)
    args = parser.parse_args()
    audit = args.audit_root.resolve()
    out = Path(__file__).resolve().parent
    component = audit.parent / "component_diagnostic/continuation_plan"
    paths = {
        "baseline_harness": component / "continuation_harness.py",
        "variant_harness": component / "continuation_variant_harness.py",
        "baseline_launcher": component / "launch_controlled_continuation.py",
        "variant_launcher": component / "launch_opacity_outside_continuation.py",
        "outside_guard": audit / "outside_experiment_guard.py",
        "observer": component / "runtime_safety_parameterized.py",
        "metric_helper": component.parent / "scripts/component_diagnostic.py",
        "short_binding": audit / "controlled_opacity_outside/experiment_binding.json",
        "short_variant_plan_v2": component / "opacity_outside_prepared_plan.json",
        "short_baseline_plan": component / "prepared_plan.json",
        "paired_protocol": audit.parent / "opacity_pair_protocol.json",
        "candidate_train": audit / "candidates/opacity_outside/train.py",
        "candidate_paper_losses": audit / "candidates/opacity_outside/utils/paper_losses.py",
        "algorithm_diff": audit / "controlled_opacity_outside/actual_algorithm_diff.patch",
    }
    if set(paths) != set(PINS):
        raise ValueError("Unexpected input inventory")
    blobs = {name: path.read_bytes() for name, path in paths.items()}
    for name in paths:
        verify_bytes(blobs[name], PINS[name])
    binding = json.loads(blobs["short_binding"])
    protocol = json.loads(blobs["paired_protocol"])
    config = binding["parent_training_config"]
    if config["iterations"] != 50000 or config["position_lr_max_steps"] != 50000 or config["seed"] != 0 or protocol["coefficient"] != .05:
        raise ValueError("Original parent/horizon/seed/coefficient invariant failed")
    preview, changes = {}, {}
    for name in ("baseline_harness", "variant_harness"):
        preview[name], changes[name] = expand_harness(blobs[name].decode(), name.startswith("variant"))
    for name in ("baseline_launcher", "variant_launcher"):
        preview[name], changes[name] = expand_launcher(blobs[name].decode(), name.startswith("variant"))
    for original, noise in (("baseline_harness", "baseline_noise_harness"), ("baseline_launcher", "baseline_noise_launcher")):
        changes[noise] = copy.deepcopy(changes[original])
        preview[noise] = replace(preview[original], NEW_COMPONENT_PLAN, NEW_COMPONENT_PLAN + "/baseline_noise_reference", changes[noise], 1)
        preview[noise] = replace(preview[noise], "original30k_to32000", "original30k_to32000_noise_reference", changes[noise])
    changes["outside_guard"] = []
    preview["outside_guard"] = replace(blobs["outside_guard"].decode(), '("stop_iteration", 30300)', '("stop_iteration", 32000)', changes["outside_guard"], 1)
    checks = []

    def check(label, value):
        if not value:
            raise AssertionError(label)
        checks.append({"check": label, "PASS": True})

    for name, text in preview.items():
        compile(text, name + "_32k_preview", "exec")
        check(name + " syntax", True)
    for name in ("baseline_harness", "variant_harness", "baseline_noise_harness"):
        original_name = "baseline_harness" if name == "baseline_noise_harness" else name
        text = preview[name]
        check(name + " exact 2000 step bound", assigned_literal(text, "FIRST") == FIRST and assigned_literal(text, "STOP") == STOP)
        check(name + " preregistered four nodes", assigned_literal(text, "NODES") == tuple(NODES))
        check(name + " original diagnostic_eval exact bytes", named_function(text, "diagnostic_eval") == named_function(blobs[original_name].decode(), "diagnostic_eval"))
        check(name + " original30k restore remains", "expected_iteration=30000" in text and "training_state30000.pth" in text and "args.iterations remains50000" in text)
        check(name + " new approval cannot accept short receipt", "TO30300'" not in text and "TO_30300'" not in text and "32000'" in text)
        check(name + " shared observation helper directory retained", "sys.path.insert(0, str(COMPONENT_PLAN))" in text and "COMPONENT_PLAN.parent / 'scripts'" in text)
    guard_before = ast.parse(blobs["outside_guard"].decode())
    guard_after = ast.parse(preview["outside_guard"])
    for node in ast.walk(guard_after):
        if isinstance(node, ast.Constant) and type(node.value) is int and node.value == 32000:
            node.value = 30300
    check("new isolated guard only stop constant differs", ast.dump(guard_before) == ast.dump(guard_after))
    check("variant original10 before-backward gate retained", named_function(preview["variant_harness"], "before_first_backward") == named_function(blobs["variant_harness"].decode(), "before_first_backward"))
    check("lambda remains exact .05 and original mask/render/loss untouched", binding["added_training_fields"] == {"diagnostic_background_domain_complement": True, "diagnostic_lambda_dynamic_background_opacity": .05} and PINS["candidate_train"] == binding["replica_source_sha256"]["train.py"] and PINS["candidate_paper_losses"] == binding["replica_source_sha256"]["utils/paper_losses.py"])
    try:
        verify_bytes(blobs["variant_harness"] + b"# drift\n", PINS["variant_harness"])
    except ValueError:
        check("changed source rejected using memory-only negative check", True)
    else:
        raise AssertionError("Drift not rejected")
    receipt_path = args.short_receipt.resolve() if args.short_receipt else component / "opacity_outside_root_coordination_receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt_matches = receipt.get("harness_sha256") == PINS["variant_harness"] and receipt.get("prepared_plan_sha256") == PINS["short_variant_plan_v2"]
    check("actual short-v2 receipt binds read-only source/config/guard", receipt_matches and receipt.get("decision") == "APPROVED_SINGLE_VARIABLE_OPACITY_OUTSIDE_30K_TO30300" and receipt.get("experiment_binding_sha256") == PINS["short_binding"] and receipt.get("guard_sha256") == PINS["outside_guard"] and receipt.get("observer_sha256") == PINS["observer"])
    filename_map = {"baseline_harness": "continuation_harness.py", "variant_harness": "continuation_variant_harness.py", "baseline_launcher": "launch_controlled_continuation.py", "variant_launcher": "launch_opacity_outside_continuation.py", "outside_guard": "outside_experiment_guard.py", "baseline_noise_harness": "baseline_noise_reference/continuation_harness.py", "baseline_noise_launcher": "baseline_noise_reference/launch_controlled_continuation.py"}
    input_name_map = {"baseline_noise_harness": "baseline_harness", "baseline_noise_launcher": "baseline_launcher"}
    source_manifest = {}
    for name, text in preview.items():
        input_name = input_name_map.get(name, name)
        patch = "".join(difflib.unified_diff(blobs[input_name].decode().splitlines(True), text.splitlines(True), fromfile="approved300/" + filename_map[input_name], tofile="draft32000/" + filename_map[name], n=2))
        patch_path = out / (name + "_extension.patch")
        patch_path.write_bytes(patch.encode())
        source_manifest[name] = {"input_path": str(paths[input_name]), "input_sha256": PINS[input_name], "proposed_source_sha256": sha(text.encode()), "patch_file": patch_path.name, "patch_sha256": sha(patch.encode()), "exact_text_replacements": changes[name], "source_materialized": False}
    proposed = copy.deepcopy(config)
    proposed["model_path"] = BASE + "/campaign/vgpu_bugfix_20261012/continuation_diagnostic/opacity_outside30k_to32000"
    proposed["start_checkpoint"] = binding["parent_static_checkpoint"]
    proposed.update(binding["added_training_fields"])
    original_proposed = {key: value for key, value in proposed.items() if key not in binding["added_training_fields"]}
    original_proposed["model_path"] = BASE + "/campaign/vgpu_bugfix_20261012/continuation_diagnostic/original30k_to32000"
    noise_proposed = copy.deepcopy(original_proposed)
    noise_proposed["model_path"] += "_noise_reference"
    locked_path = out / "evaluation_protocol_locked.json"
    locked = json.loads(locked_path.read_bytes())
    check("long evaluation protocol locked with full-window original repeat", locked["window"] == [FIRST, STOP] and locked["coefficient"] == .05 and set(locked["three_trajectories_if_window_is_authorized"]) == {"B1", "B2", "V"} and locked["reuse_30300_terminal"] is False)
    check("noise arm only output routing differs from original-math baseline", {k: v for k, v in noise_proposed.items() if k != "model_path"} == {k: v for k, v in original_proposed.items() if k != "model_path"})
    bundle = {Path(path).name: {"path": path, "sha256": value} for path, value in binding["protected_original_file_sha256"].items() if Path(path).name in ("chkpnt30000.pth", "dynamic_chkpnt30000.pth", "env_light_chkpnt30000.pth", "training_state30000.pth")}
    check("both groups use original four30k files, never30300 state", len(bundle) == 4 and all("30300" not in row["path"] for row in bundle.values()))
    check("source window has no later reset", config["opacity_reset_interval"] == 3000 and not any(step % 3000 == 0 for step in range(FIRST, STOP + 1)))
    check("densification and SGCM end precede window", config["densify_until_iter"] == 15000 and config["dynamic_densify_until_iter"] == 15000 and config["sgcm_end"] == 15000)
    draft = {
        "schema_version": 1, "status": "DRAFT_NOT_LAUNCHABLE", "allow_training": False,
        "source_materialized": False, "repository_clones_created": 0, "GPU_used": False,
        "scope": "Conditional longer paired causal diagnostic; root decides only after short300 results. No claim of final paper reproduction.",
        "parent_iteration": 30000, "first_iteration": FIRST, "stop_iteration": STOP, "steps": STOP - FIRST + 1,
        "lr_horizon": 50000, "position_lr_max_steps": 50000, "seed": 0,
        "coefficient": .05, "coefficient_provenance": "Unchanged prior preregistered value; no outcome tuning.",
        "original30k_four_file_pins": bundle,
        "baseline_training_config_draft": original_proposed, "outside_training_config_draft": proposed,
        "baseline_noise_training_config_draft": noise_proposed,
        "original_repeat_noise_reference": "Independent B2 original-math trajectory from original30k through32000, all2000 camera/time_shift/LR compared to B1 and terminal all49. One observed repeat difference is descriptive, not a statistical bound.",
        "same_window_noise_reference_required_before_small_effect_claim": True,
        "single_existing_short_trajectory_or_first_step_exact_is_not_noise_reference": True,
        "reuse30300_terminal_selected": False,
        "reuse30300_fairness_and_gate": "Selected approach is fresh original30k for all three. A30300 shortcut could only be fair after exact branch-specific full terminal Adam/RNG/sampler/EMA/runtime/config/source lineage validation plus equal30001..30300 histories, new30301 restore/first-step guards and RNG-safe eval proof; it is not implemented or authorized here.",
        "evaluation_protocol_locked_path": str(locked_path), "evaluation_protocol_locked_sha256": sha(locked_path.read_bytes()),
        "baseline_repository": binding["original_repository"],
        "outside_repository_proposed_not_created": REMOTE_DRAFT + "/variant_frozen_repository",
        "outside_algorithm_source_pins_same_as_short": {"train.py": PINS["candidate_train"], "utils/paper_losses.py": PINS["candidate_paper_losses"]},
        "outside_algorithm_diff_sha256_same_as_short": PINS["algorithm_diff"],
        "added_training_fields": binding["added_training_fields"], "strict_original_config_for_outside": False,
        "all_other_original_config_fields_unchanged": True,
        "evaluation": {"nodes": NODES, "early_fixed_camera_ids": binding["evaluation"]["early_fixed_camera_ids"], "terminal_eval_camera_count": 49, "all_extra_eval_rng_preserved": True},
        "schedule": {"next_static_reset": 33000, "stop_before_next_static_reset": True, "densify_and_SGCM_ended": 15000, "original_SH_hook_at32000_retained": True, "other_original_hooks_not_disabled": True},
        "restore": "Both restore original saved30k static/dynamic/environment/complete sidecar, all Adam/RNG/sampler/runtime/EMA/SGCM state; same exact observer expected_iteration30000; no seed after restore.",
        "paired_checks": ["First30001 original10 scalars exactly equal long baseline before real backward; pre-add loss capture, no subtraction.", "All2000 iteration/camera/time_shift/LR records match actual long baseline exactly; later losses may differ.", "Every extra eval restores all RNG exactly; common49 input/source/kernel/helper hashes preserved.", "Terminal32000 all49 original frames and static/dynamic/environment/complete state four-file snapshot, hashed and CPU-load checked."],
        "reference_not_yet_available": {"long_baseline_2000_step_rows": None, "long_baseline_terminal49": None, "long_baseline_repeat_2000_step_rows": None, "long_baseline_repeat_terminal49": None, "full_window_noise_comparison_SHA": None, "new_exact_baseline_plan_hash": None, "new_exact_noise_baseline_plan_hash": None, "new_exact_outside_binding_hash": None, "new_exact_outside_plan_hash": None, "new_independent_review_hashes": None, "new_root_receipts": None},
        "short_protocol_path": str(paths["paired_protocol"]), "short_protocol_sha256": PINS["paired_protocol"],
        "measurements_same_as_short": protocol["primary_measurements"],
        "directional_hypothesis_same_as_short": protocol["directional_hypothesis"],
        "interpretation": "2000-step continuation from an already degenerated30k model tests recovery/time-window adequacy. Success supports the background-domain hypothesis, not author formula; failure does not rule out from-zero training. Report direction, magnitude and reconstruction tradeoffs without new outcome thresholds.",
        "forbidden_joint_changes": protocol["forbidden_joint_changes"],
        "source_manifest": source_manifest,
        "shared_observer_sha256": PINS["observer"], "shared_metric_helper_sha256": PINS["metric_helper"],
        "new_materialized_plan_directory_proposed": REMOTE_PLAN,
        "new_approval_decisions_required": ["APPROVED_CONTROLLED_30K_TO_32000", "APPROVED_SINGLE_VARIABLE_OPACITY_OUTSIDE_30K_TO32000"],
        "short_approval_never_authorizes_long_window": True,
        "deferred_work": ["Root assesses short300 outcome and may decline this longer window; no commitment to run32000.", "After instruction, materialize reviewed small source previews into independent directories; create at most one isolated algorithm replica; regenerate full CPU B1/B2 plans and source/config/input pins with new independent reviews and exact separate root receipts.", "Execute original-math B1 and independent B2, each fresh30k only with exclusive root-assigned GPU ownership; require each complete2000 records, all49 terminal eval, four-state snapshot and exit0. CPU-verify all2000 sampling/time_shift/LR equality and save complete empirical noise comparison.", "Build new outside binding using actual completed B1 references and SHA-bound B2 full-window noise comparison, isolated32000 guard and same .05 algorithm; regenerate CPU preflight/independent review and exact separate root receipt before any candidate launch."],
    }
    json_write(out / "long_pair_plan_draft.json", draft)
    json_write(out / "source_transforms.json", source_manifest)
    after = {name: sha(path.read_bytes()) for name, path in paths.items()}
    check("all14 protected short inputs byte-identical after generation", after == PINS)
    check("no executable source/model copies created", not (out / "materialized_plan").exists() and not (out / "variant_frozen_repository").exists())
    report = {"status": "PASS_DRAFT_ONLY", "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "checks": checks, "check_count": len(checks), "input_file_pins": {name: {"path": str(paths[name]), "sha256": PINS[name]} for name in paths}, "short_current_receipt_path": str(receipt_path), "short_current_receipt_sha256": sha(receipt_path.read_bytes()), "short_current_receipt_matches_v2_source": receipt_matches, "short_receipt_not_reused_for_long": True, "training_imported": False, "torch_imported": False, "actual_model_optimizer_CUDA_restore_performed": False, "allow_training": False, "GPU_used": False, "source_copies_created": 0, "repository_clones_created": 0, "only_diffs_and_draft_plan_emitted": True, "generator_sha256": sha(Path(__file__).read_bytes())}
    json_write(out / "generator_cpu_validation.json", report)
    print(json.dumps({"status": report["status"], "checks": len(checks), "draft_plan": str(out / "long_pair_plan_draft.json"), "short_receipt_matches_v2": receipt_matches, "source_copies": 0, "GPU_used": False}))


if __name__ == "__main__":
    main()

"""Stage the reviewed small long-window sources/configs/CPU plan drafts only.

Default is a read-only check. --materialize writes only this independent new
materialized_plan directory, refuses overwrite, creates no repository clone,
imports no Torch/training, and has no launch or run method. Variant runtime plan
cannot be finalized until genuine B1/B2 long-window evidence is available.
"""
import argparse
import copy
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    audit = args.audit_root.resolve()
    component = audit.parent / "component_diagnostic/continuation_plan"
    destination = here / "materialized_plan"
    validation = json.loads((here / "generator_cpu_validation.json").read_bytes())
    draft = json.loads((here / "long_pair_plan_draft.json").read_bytes())
    manifest = json.loads((here / "source_transforms.json").read_bytes())
    generator_path = here / "generate_long_pair_draft_cpu.py"
    if validation["status"] != "PASS_DRAFT_ONLY" or validation["generator_sha256"] != sha(generator_path.read_bytes()) or draft["status"] != "DRAFT_NOT_LAUNCHABLE" or draft["allow_training"]:
        raise ValueError("Reviewed CPU draft/generator pin changed")
    if sha((here / "evaluation_protocol_locked.json").read_bytes()) != draft["evaluation_protocol_locked_sha256"]:
        raise ValueError("Locked evaluation/noise protocol changed")
    spec = importlib.util.spec_from_file_location("readonly_long_draft_source_generator", generator_path)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    paths = {
        "baseline_harness": component / "continuation_harness.py",
        "variant_harness": component / "continuation_variant_harness.py",
        "baseline_launcher": component / "launch_controlled_continuation.py",
        "variant_launcher": component / "launch_opacity_outside_continuation.py",
        "outside_guard": audit / "outside_experiment_guard.py",
        "candidate_train": audit / "candidates/opacity_outside/train.py",
        "candidate_paper_losses": audit / "candidates/opacity_outside/utils/paper_losses.py",
        "short_binding": audit / "controlled_opacity_outside/experiment_binding.json",
        "short_baseline_plan": component / "prepared_plan.json",
        "short_variant_plan_v2": component / "opacity_outside_prepared_plan.json",
    }
    blobs = {name: path.read_bytes() for name, path in paths.items()}
    for name, data in blobs.items():
        generator.verify_bytes(data, generator.PINS[name])
    preview = {}
    for name in ("baseline_harness", "variant_harness"):
        preview[name], _ = generator.expand_harness(blobs[name].decode(), name.startswith("variant"))
    for name in ("baseline_launcher", "variant_launcher"):
        preview[name], _ = generator.expand_launcher(blobs[name].decode(), name.startswith("variant"))
    preview["outside_guard"] = blobs["outside_guard"].decode().replace('(\"stop_iteration\", 30300)', '(\"stop_iteration\", 32000)')
    for name in ("harness", "launcher"):
        preview["baseline_noise_" + name] = preview["baseline_" + name].replace(generator.NEW_COMPONENT_PLAN, generator.NEW_COMPONENT_PLAN + "/baseline_noise_reference").replace("original30k_to32000", "original30k_to32000_noise_reference")
    names = {"baseline_harness": "continuation_harness.py", "variant_harness": "continuation_variant_harness.py", "baseline_launcher": "launch_controlled_continuation.py", "variant_launcher": "launch_opacity_outside_continuation.py", "outside_guard": "outside_experiment_guard.py", "baseline_noise_harness": "baseline_noise_reference/continuation_harness.py", "baseline_noise_launcher": "baseline_noise_reference/launch_controlled_continuation.py"}
    if set(preview) != set(manifest) or set(preview) != set(names):
        raise ValueError("Unexpected materialized source inventory")
    staged = {}
    for name, text in preview.items():
        if sha(text.encode()) != manifest[name]["proposed_source_sha256"]:
            raise ValueError("Source differs from independently reviewable expansion diff: " + name)
        compile(text, name, "exec")
        staged[names[name]] = text.encode()
    staged["algorithm_overrides/train.py"] = blobs["candidate_train"]
    staged["algorithm_overrides/utils/paper_losses.py"] = blobs["candidate_paper_losses"]
    roles = [
        ("B1", "baseline_training_config_draft", "baseline_effective_config.yaml", "baseline_cpu_plan_draft.json", "continuation_harness.py"),
        ("B2", "baseline_noise_training_config_draft", "baseline_noise_reference/effective_config.yaml", "baseline_noise_reference/cpu_plan_draft.json", "baseline_noise_reference/continuation_harness.py"),
        ("V", "outside_training_config_draft", "outside_effective_config.yaml", "outside_cpu_plan_draft.json", "continuation_variant_harness.py"),
    ]
    short_baseline = json.loads(blobs["short_baseline_plan"])
    short_variant = json.loads(blobs["short_variant_plan_v2"])
    for role, config_key, config_filename, plan_filename, harness_filename in roles:
        plan = copy.deepcopy(short_variant if role == "V" else short_baseline)
        config = copy.deepcopy(draft[config_key])
        if role == "V":
            plan["repository"] = draft["outside_repository_proposed_not_created"]
            old_repo = short_variant["repository"]
            plan["runtime_source_sha256"] = {key.replace(old_repo, plan["repository"], 1): value for key, value in plan["runtime_source_sha256"].items()}
            plan["first_step_reference"] = None
            plan["all_iteration_pair_reference"] = None
            plan["baseline_terminal_evaluation"] = None
            plan["experiment_binding_path"] = generator.REMOTE_PLAN + "/experiment_binding.json"
            plan["experiment_binding_sha256"] = None
            plan["guard_path"] = generator.REMOTE_PLAN + "/outside_experiment_guard.py"
            plan["guard_sha256"] = sha(staged["outside_experiment_guard.py"])
            plan["launcher_path"] = generator.REMOTE_PLAN + "/launch_opacity_outside_continuation.py"
            plan["launcher_sha256"] = sha(staged["launch_opacity_outside_continuation.py"])
            plan["preparation_checks"] = None
            plan["all467_original_inputs_hash_verified"] = False
            plan["original_file_sha256"] = short_baseline["original_file_sha256"]
        for field in ("training_math", "diagnostics", "stop_policy", "paired_trajectory_policy"):
            if isinstance(plan.get(field), str):
                plan[field] = plan[field].replace("30300", "32000").replace("all300", "all2000").replace("of300", "of2000")
        plan.update(status="CPU_SOURCE_CONFIG_DRAFT_NOT_RUNTIME_PREFLIGHTED", allow_training=False, role=role, first_iteration=30001, stop_iteration=32000, steps=2000, output=config["model_path"], training_config=config, harness_sha256=sha(staged[harness_filename]), extra_eval_nodes=draft["evaluation"]["nodes"], evaluation_protocol_sha256=draft["evaluation_protocol_locked_sha256"], exact_50k_schedule_preserved=True, actual_current_source_input_checkpoint_rehash_pending=True, actual_model_optimizer_CUDA_restore_performed=False, new_exact_root_approval=None, full_window_B1_B2_noise_reference_pending=True, new_plan_requires_actual_harness_CPU_prepare=True, baseline_original_training_math_unchanged=(role != "V"), only_algorithm_delta=[".05 outside dynamic opacity domain complement"] if role == "V" else [])
        staged[config_filename] = encode(config)  # JSON is also valid YAML for OmegaConf.
        staged[plan_filename] = encode(plan)
    staged["evaluation_protocol_locked.json"] = (here / "evaluation_protocol_locked.json").read_bytes()
    staged["long_pair_plan_draft.json"] = (here / "long_pair_plan_draft.json").read_bytes()
    inputs_before = {name: sha(data) for name, data in blobs.items()}
    if args.materialize:
        if destination.exists() or destination.is_symlink() or destination.parent.resolve() != here:
            raise ValueError("Only a new isolated materialized_plan directory permitted; no overwrite")
        destination.mkdir()
        for relative, data in staged.items():
            target = destination / relative
            if target.resolve().parent != destination.resolve() and not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError("Target escaped independent directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(data)
    inputs_after = {name: sha(path.read_bytes()) for name, path in paths.items()}
    if inputs_before != inputs_after:
        raise ValueError("Protected current inputs changed during materializer")
    if args.materialize and any((destination / name).read_bytes() != data for name, data in staged.items()):
        raise ValueError("Materialized output differs from pinned proposed bytes")
    receipt = {"status": "CPU_MATERIALIZED_DRAFT_NOT_LAUNCHABLE" if args.materialize else "CHECK_ONLY_PASS", "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "allow_training": False, "destination": str(destination), "files": {name: {"sha256": sha(data), "bytes": len(data)} for name, data in staged.items()}, "source_files": 9, "configuration_files": 3, "CPU_plan_drafts": 3, "repository_clones_created": 0, "original_current_input_bytes_unchanged": True, "inputs": {name: {"path": str(paths[name]), "sha256": inputs_after[name]} for name in paths}, "torch_imported": False, "training_imported": False, "GPU_used": False, "actual_model_optimizer_CUDA_restore_performed": False, "long_variant_runtime_references": "Pending genuine B1/B2 2000-step/all49 evidence; original300 approval/binding never reused", "materializer_sha256": sha(Path(__file__).read_bytes())}
    if args.materialize:
        (destination / "materialization_receipt.json").write_bytes(encode(receipt))
    else:
        (here / "materializer_check_only.json").write_bytes(encode(receipt))
    print(json.dumps({key: receipt[key] for key in ("status", "allow_training", "destination", "source_files", "configuration_files", "CPU_plan_drafts", "repository_clones_created", "GPU_used")}))


if __name__ == "__main__":
    main()

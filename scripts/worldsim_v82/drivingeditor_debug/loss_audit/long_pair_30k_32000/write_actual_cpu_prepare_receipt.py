"""Bind actual default CPU prepare outputs; never load Torch or start training."""
import datetime
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    materialized = here / "materialized_plan"
    draft = json.loads((here / "long_pair_plan_draft.json").read_bytes())
    roles = [
        ("B1", materialized / "prepared_plan.json", materialized / "continuation_harness.py", "baseline_training_config_draft", "359b9d62ee9aec28951532cedbaed45401838dd6ea2b961d5704bec2d303a2d5", 97139),
        ("B2", materialized / "baseline_noise_reference/prepared_plan.json", materialized / "baseline_noise_reference/continuation_harness.py", "baseline_noise_training_config_draft", "0b07086c6d425b7dc8526266b8520d5bd8d8d4736697833574029b2c22681329", 58403),
    ]
    checks, results = [], {}
    for role, path, source, config_key, expected_plan_sha, session in roles:
        plan = json.loads(path.read_bytes())
        if sha(path) != expected_plan_sha or plan["harness_sha256"] != sha(source) or plan["training_config"] != draft[config_key]:
            raise ValueError("Actual CPU plan/source/config differs from frozen draft: " + role)
        if (plan["parent_iteration"], plan["first_iteration"], plan["stop_iteration"], plan["steps"], plan["horizon"], plan["position_lr_max_steps"], plan["seed"]) != (30000, 30001, 32000, 2000, 50000, 50000, 0):
            raise ValueError("Actual prepared schedule or initial state differs")
        if plan["allow_training"] or plan["status"] != "prepared_not_started" or plan["extra_eval_nodes"] != [30001, 30300, 31000, 32000] or plan["terminal_eval_camera_count"] != 49 or plan["algorithm_changes"] or plan["training_config_changed_fields"] != ["model_path", "start_checkpoint"]:
            raise ValueError("Prepared baseline changed protocol or mathematics")
        if plan["observer_sha256"] != "f60faca00ec0d5b10b80d14d3e8ca07125ada5bff06d1f0fb8c43c5fec6d42d1" or len(plan["first_step_reference"]["values"]) != 10:
            raise ValueError("Original restore observer/first reference changed")
        checks.append({"role": role, "source_config_schedule_nodes_original_math": "PASS"})
        results[role] = {"path": str(path), "prepared_plan_sha256": sha(path), "harness_sha256": sha(source), "actual_remote_default_CPU_prepare_exit_code": 0, "tool_session_id": session, "environment": {"PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}, "original_file_pin_count_actual_harness_checked": len(plan["original_file_sha256"]), "parent_complete30k_bundle_config_CPU_metadata_checked": True, "original_file_sha256": plan["original_file_sha256"], "training_started": False, "GPU_stage_invoked": False, "actual_model_optimizer_CUDA_restore_performed": False}
    p1 = json.loads(roles[0][1].read_bytes())
    p2 = json.loads(roles[1][1].read_bytes())
    if p1["original_file_sha256"] != p2["original_file_sha256"] or p1["first_step_reference"] != p2["first_step_reference"]:
        raise ValueError("Original repeat changed parent/kernel/data/first reference")
    receipt = {"status": "PASS_ACTUAL_CPU_PREPARE_B1_B2_ONLY", "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "results": results, "checks": checks, "all_original_pin_maps_and_first_step_reference_equal": True, "evaluation_protocol_locked_sha256": sha(here / "evaluation_protocol_locked.json"), "outside_status": "SOURCE_CONFIG_CPU_DRAFT_PASS; runtime binding/plan pending genuine B1/B2 all2000 trajectories and49 endpoints", "new_root_receipts": None, "allow_training": False, "GPU_used": False, "full_original_optimizer_RNG_sampler_restore_requires_future_GPU_observer": True}
    (here / "actual_baseline_cpu_prepare_receipt.json").write_bytes((json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode())
    print(json.dumps({"status": receipt["status"], "B1_plan": results["B1"]["prepared_plan_sha256"], "B2_plan": results["B2"]["prepared_plan_sha256"], "pin_count_each": results["B1"]["original_file_pin_count_actual_harness_checked"], "training_started": False}))


if __name__ == "__main__":
    main()

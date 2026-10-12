"""Independent, standard-library-only review of the actual long V preparation.

Reads frozen source and saved evidence. Writes only this reviewer directory.
Never imports training, Torch, the candidate generator, a launcher, or a GPU API.
Large checkpoint/NPZ/input rehashes are not repeated: their actual CPU receipts
are pinned and their precise evidence limits are preserved.
"""
import argparse
import ast
import datetime
import hashlib
import json
import math
from pathlib import Path


REMOTE_ROOT = "/root/autodl-tmp/drivingeditor_repro/campaign/vgpu_bugfix_20261012"
LONG_REL = "loss_audit/long_pair_30k_32000"
EXPECTED = {
    LONG_REL + "/materialized_plan/experiment_binding.json": "5701482a50b078e980a39a4aec9b6747cdc1628882b2b48a2dd985d588314260",
    LONG_REL + "/materialized_plan/opacity_outside_prepared_plan.json": "1afe9a9b5f05dfcff1d565e064f276ef8161250060eecc0bb2b31a35be031065",
    LONG_REL + "/materialized_plan/continuation_variant_harness.py": "93df22a4b8db482af596657ba1d3e86e61a0ad0893dac3af35f85adf872ba1d4",
    LONG_REL + "/materialized_plan/outside_experiment_guard.py": "8356201320c757079a5dfaf4cb4968e22bc537fbf8f1dd7e0e6d1b6242f0d42c",
    LONG_REL + "/materialized_plan/launch_opacity_outside_continuation.py": "633019f97b800ea3a942a1b44dd41d7972b2e332ee714caddb21a0e69e443898",
    LONG_REL + "/materialized_plan/outside_effective_config.yaml": "97fa2db569466c9a903ad5d83ac788a1b2afeb5fff8804875ffd39c37851ad3c",
    LONG_REL + "/materialized_plan/prepared_plan.json": "359b9d62ee9aec28951532cedbaed45401838dd6ea2b961d5704bec2d303a2d5",
    LONG_REL + "/materialized_plan/baseline_noise_reference/prepared_plan.json": "0b07086c6d425b7dc8526266b8520d5bd8d8d4736697833574029b2c22681329",
    LONG_REL + "/evaluation_protocol_locked.json": "c725cd2153d1104ddaf81078ac753f4e7f474e0be325ca9fbb5b1fccb342f658",
    LONG_REL + "/v_binding_cpu/approved_original_source_sha256.json": "459e41773f8200f35db6812e72ea13b673d0a576abe588422f7be7b3d3e8c0c9",
    LONG_REL + "/v_binding_cpu/actual_algorithm_diff.patch": "4ac897365605f30f91ac0a21d4440cbc79b86d60be20de274cfee50f5e6628de",
    LONG_REL + "/v_binding_cpu/prepare_long_variant_binding_cpu.py": "ec2adbf6b83ed3372fcd09b6007fcb2c91404a6f9068a0827dc601d243c58832",
    LONG_REL + "/v_binding_cpu/pair_reference_validation.json": "d9555323b446f20ca425a0f7448793628ca30e3559ed24e551610eb7c0996afc",
    LONG_REL + "/v_binding_cpu/cpu_prepare_receipt.json": "0d93907f924ad730415dd9ad8597bd034a222548a33c9dffd3a93dc9b6df497f",
    "component_diagnostic/continuation_plan/runtime_safety_parameterized.py": "f60faca00ec0d5b10b80d14d3e8ca07125ada5bff06d1f0fb8c43c5fec6d42d1",
    "component_diagnostic/scripts/component_diagnostic.py": "0170283337ce3e2a1ea7a016e6f8a8e2b534101eebfe7b3ba4ecdcbcc1903baa",
    "component_diagnostic/scripts/quantify_long_original_repeat_cpu.py": "d2a5d00ff35f352a186b2f6fa5bfcc7d3036b92eecd0cf7a889122f88e51a705",
    "component_diagnostic/scripts/verify_long_repeat_common_inputs_cpu.py": "38361760284ebc04b49829e274d69fa3c6a6d21319c5f51db957a0b1a05f1ec8",
    "sgcm_data_audit/long_pair_review/interpretation_preregistered.json": "af9726931762fde14c00684ed1004ce2a6464a46299e832f969cdce3c916a75d",
    "continuation_diagnostic/original30k_to32000_noise_reference/receipts/B1_B2_observed_noise.json": "e4ae32c07b63ed7be0171a727f8a12ec153f66e63d6ac89f4594fa4985145f08",
    "continuation_diagnostic/original30k_to32000_noise_reference/receipts/common49_inputs_after_B2_cpu.json": "dfd33893200805b4aebf147121b2be72270163a2ec61674e7704d2053de7f381",
    "continuation_diagnostic/original30k_to32000_noise_reference/receipts/full49_summary.json": "2ded7dda0952e0bf8605154a642576049ce45b41e8c1df1ddda3bec62085c92e",
    "continuation_diagnostic/original30k_to32000_noise_reference/receipts/gpu_release_handoff.json": "3486faf0bfe3bcc85949094a0d538a829766504b1eddd63414c10008144df4eb",
}
EXTRAS = {"diagnostic_background_domain_complement": True, "diagnostic_lambda_dynamic_background_opacity": 0.05}
ALGORITHMS = {"train.py": "ff82c0d86b385bcdf28feff550163cd3ec9c7bec92e939306330ba4af335e99a", "utils/paper_losses.py": "3832a7ced1387e1ff33f5e3aee0b56bc533303f80f5be840269236139fc866ab"}
FRAMES = [f"image_0/0017{i:03}" for i in range(49)]
NODES = [30001, 30300, 31000, 32000]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def same(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def inventory(root):
    rows = {}
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if any(x in (".git", "__pycache__") for x in rel.parts) or path.suffix in (".pyc", ".pyo"):
            continue
        if path.is_symlink():
            raise AssertionError("Source symlink: " + str(rel))
        if path.is_file():
            rows[rel.as_posix()] = sha(path)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-source-only", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    audit = here.parents[1]
    long = audit / LONG_REL
    materialized = long / "materialized_plan"
    binding = load(materialized / "experiment_binding.json")
    plan = load(materialized / "opacity_outside_prepared_plan.json")
    checks = []

    def check(label, condition):
        checks.append({"check": label, "PASS": bool(condition)})
        if not condition:
            raise AssertionError(label)

    def local(remote):
        assert remote.startswith(REMOTE_ROOT + "/"), remote
        return audit / remote[len(REMOTE_ROOT) + 1:]

    pins = {rel: sha(audit / rel) for rel in EXPECTED}
    check("Actual23 source/config/binding/plan/protocol/reference SHA pins match frozen final files", pins == EXPECTED)
    original = load(long / "v_binding_cpu/approved_original_source_sha256.json")
    replica = inventory(long / "variant_frozen_repository")
    check("Actual replica inventory148 matches binding; approved original148 differs in exactly2 reviewed algorithm files", len(replica) == len(original) == 148 and replica == binding["replica_source_sha256"] and original == binding["original_source_sha256"] and {k for k in original if original[k] != replica[k]} == set(ALGORITHMS) and all(replica[k] == v for k, v in ALGORITHMS.items()))
    if args.remote_source_only:
        check("Actual remote original148 bytes match approved inventory; formal checkpoint/SGCM helper unchanged", inventory(Path(binding["original_repository"])) == original and original["utils/paper_checkpoint.py"] == replica["utils/paper_checkpoint.py"] and original["scene/paper_sgcm.py"] == replica["scene/paper_sgcm.py"])
        result = {"status": "PASS", "scope": "Actual remote small source/config/plan/reference and148+148 source SHA only; no CUDA, training or large checkpoint/NPZ/input rehash", "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "checks": checks, "critical_file_sha256": pins, "source_file_count_each": 148, "reviewer_script_sha256": sha(__file__), "GPU_used": False}
        path = here / "remote_actual_V_source_plan_cpu.json"
        write(path, result)
        print(json.dumps({"status": result["status"], "path": str(path), "sha256": sha(path), "checks": len(checks)}))
        return

    remote = load(here / "remote_actual_V_source_plan_cpu.json")
    check("Independent actual remote review binds same final files and reviewer source", remote["status"] == "PASS" and remote["critical_file_sha256"] == pins and remote["reviewer_script_sha256"] == sha(__file__) and remote["GPU_used"] is False)
    b1 = load(materialized / "prepared_plan.json")
    b2 = load(materialized / "baseline_noise_reference/prepared_plan.json")
    config = load(materialized / "outside_effective_config.yaml")
    expected_config = dict(b1["training_config"], model_path=binding["output"], **EXTRAS)
    parent = binding["parent_training_config"]
    check("Actual complete130-field V config is original128 plus exact2 added fields and output/30k selector routing only", len(parent) == 128 and same(config, expected_config) and same(config, binding["training_config"]) and same(config, plan["training_config"]) and {k for k in parent if not same(parent[k], config[k])} == {"model_path", "start_checkpoint"} and set(config) - set(parent) == set(EXTRAS) and config["start_checkpoint"] == binding["parent_static_checkpoint"] and binding["parent_static_checkpoint"].endswith("/chkpnt30000.pth"))
    check("Actual window fresh30k through32000 retains50k LR horizon, fixed4 nodes; no33000 reset or new SGCM/densification", all(same(plan[k], v) and same(binding[k], v) for k, v in {"parent_iteration": 30000, "first_iteration": 30001, "stop_iteration": 32000, "steps": 2000, "horizon": 50000, "seed": 0}.items()) and plan["position_lr_max_steps"] == config["position_lr_max_steps"] == config["iterations"] == 50000 and plan["extra_eval_nodes"] == NODES and config["opacity_reset_interval"] == 3000 and not any(i % 3000 == 0 for i in range(30001, 32001)) and config["sgcm_end"] == config["densify_until_iter"] == config["dynamic_densify_until_iter"] == 15000)
    check("Original B1/B2 full initial source/kernel/data/checkpoint pins equal; V retains all originals and all protected/auxiliary pins", b1["original_file_sha256"] == b2["original_file_sha256"] and all(plan["original_file_sha256"].get(k) == v for d in (b1["original_file_sha256"], binding["protected_original_file_sha256"], binding["auxiliary_file_sha256"]) for k, v in d.items()) and {str(Path(binding["repository"]) / k).replace("\\", "/"): v for k, v in replica.items()} == plan["runtime_source_sha256"])
    initial4 = {k: v for k, v in binding["protected_original_file_sha256"].items() if k.endswith(".pth")}
    check("Only same4 saved original30k initial states selected; no30300 or32000 terminal used for V initialization", len(initial4) == 4 and all("/runs/" in k and k.endswith("30000.pth") and plan["original_file_sha256"].get(k) == v for k, v in initial4.items()) and plan["parent"] == b1["parent"] == b2["parent"])
    pair = load(long / "v_binding_cpu/pair_reference_validation.json")
    arms = []
    for baseline in (b1, b2):
        output = local(baseline["output"])
        status = load(output / "status.json")
        report = load(output / "continuation_diagnostic.json")
        summary = load(output / "receipts/full49_summary.json")
        snap = load(output / "terminal_32000/diagnostic_snapshot_manifest.json")
        final = load(output / "extra_eval/32000/diagnostic_eval.json")
        rows = [json.loads(s) for s in (output / "step_losses.jsonl").read_text().splitlines()]
        check(output.name + " actual exit0/2000 real steps/restore/RNG-safe49/four-state CPU-schema receipts complete", status["status"] == "complete" and status["actual_exit_code"] == int((output / "actual_exit_code.txt").read_text()) == 0 and status["gpu_lock_held"] is False and status["iteration"] == 32000 and report["step_count"] == 2000 and report["original_algorithm_changed"] is False and report["genuine_full_loss_backward_Adam"] is True and report["restore"]["status"] == "actual_restore_pass" and all(report["restore"][k] is True for k in ("all_rng_restored_exactly", "camera_sampler_scale_ema_restored_exactly", "model_runtime_sgcm_restored_exactly", "conditional_adam_restored_exactly", "saved_optimizer_tensor_placement_preserved", "optimizer_bound_to_saved_parameters")) and report["restore"]["seed_after_restore"] is False and report["all_node_eval_rng_preserved"] is True and report["terminal_full49_evaluation"] is True and summary["all_protected_SHA_unchanged"] is True and set(snap["checkpoints"]) == {"static", "dynamic", "environment", "training_state"} and all(r["cpu_load_iteration_schema_validated"] is True and r["sha256"] == summary["checkpoint_validation"][k]["sha256"] and r["bytes"] == summary["checkpoint_validation"][k]["bytes"] for k, r in snap["checkpoints"].items()) and [f["frame_id"] for f in final["frames"]] == FRAMES and final["all_rng_unchanged"] is True)
        arms.append({"rows": rows, "final": final, "report": report, "snap": snap})
    keys = ("iteration", "camera_id", "time_shift", "learning_rates")
    trajectories = [[{k: r[k] for k in keys} for r in arm["rows"]] for arm in arms]
    first = binding["first_step_pair_reference"]["values"]
    check("All2000 actual B1/B2 camera/time_shift/LR exact; V plan/binding reuse those actual rows and first10 original scalars", trajectories[0] == trajectories[1] == binding["all_iteration_pair_reference"]["rows"] == plan["all_iteration_pair_reference"]["rows"] and [r["iteration"] for r in trajectories[0]] == list(range(30001, 32001)) and len(first) == 10 and all({k: arm["rows"][0]["losses"][k] for k in first} == first for arm in arms) and plan["first_step_reference"] == binding["first_step_pair_reference"] and pair["all2000_sampling_time_shift_LR_exact"] is True)
    check("Actual full49 region counts identical; pinned CPU array-pair receipt proves original GT float32 equality without claiming independent runtime-mask/pose snapshots", len(pair["all49_domains"]) == 49 and all(a["frame_id"] == b["frame_id"] == p["frame_id"] and {k: v["pixel_count"] for k, v in a["regions"].items()} == {k: v["pixel_count"] for k, v in b["regions"].items()} == p["region_pixel_counts"] and p["GT_float32_exact"] is True for a, b, p in zip(arms[0]["final"]["frames"], arms[1]["final"]["frames"], pair["all49_domains"])) and plan["independent_actual_Scene_per_frame_pose_receipt_available"] is False)
    noise = load(audit / "continuation_diagnostic/original30k_to32000_noise_reference/receipts/B1_B2_observed_noise.json")
    protocol = load(long / "evaluation_protocol_locked.json")
    primary = [(p["domain"], p["array"], metric) for p in protocol["primary_metrics"] for metric in p["metrics"]]
    all_noise = noise["primary_metrics"] + noise["context_metrics_all_six_domains"]
    numeric_ok = True
    for row in all_noise:
        values = [[f["regions"][row["region"]]["metrics"][row["array"]][row["metric"]] for f in arm["final"]["frames"] if f["regions"][row["region"]]["pixel_count"] > 0] for arm in arms]
        means = [sum(v) / len(v) for v in values]
        numeric_ok &= all(math.isclose(x, y, rel_tol=0, abs_tol=1e-12) for x, y in ((means[0], row["equal_frame_mean_B1"]), (means[1], row["equal_frame_mean_B2"]), (abs(means[1] - means[0]), row["N_abs_difference_of_equal_frame_means"])))
    check("Actual canonical noise retains frozen11 primary+126 context metrics and equal-frame means from real49; empirical repeat not CI/noise bound", primary == [(r["region"], r["array"], r["metric"]) for r in noise["primary_metrics"]] and len(primary) == 11 and len(noise["context_metrics_all_six_domains"]) == 126 and numeric_ok and binding["original_repeat_noise_reference"]["is_confidence_interval_or_CUDA_noise_bound"] is False and noise["normal_algorithm_success_claimed"] is False)
    common = load(audit / "continuation_diagnostic/original30k_to32000_noise_reference/receipts/common49_inputs_after_B2_cpu.json")
    original_common = load(audit / "sgcm_data_audit/outside_opacity_review/common49_input_manifest_review_cpu.json")
    check("Actual postB2 input467/required245/decoded98mask receipt matches original immutable inputs, no shadow and explicit snapshot limitation", common["actual467_file_SHA_matches_original"] is True and common["actual245_required_common49_SHA_match"] is True and common["actual_input_SHA"] == original_common["file_sha256"] and len(common["actual_input_SHA"]) == 467 and len(common["common49_required_relative_paths"]) == 245 and len(common["common49_original_decoded_mask_receipts"]) == 98 and {(r["frame_id"], r["kind"]) for r in common["common49_original_decoded_mask_receipts"]} == {(f, k) for f in FRAMES for k in ("sky", "actor")} and common["new_shadow_mask_absent"] is True and common["common_independent_receipt_sha256"] == sha(audit / "sgcm_data_audit/outside_opacity_review/common49_input_manifest_review_cpu.json"))
    train_ast = ast.parse((long / "variant_frozen_repository/train.py").read_text(encoding="utf-8"))
    real_import = [n for n in ast.walk(train_ast) if isinstance(n, ast.ImportFrom) and n.module == "utils.loss_utils"]
    harness = (materialized / "continuation_variant_harness.py").read_text(encoding="utf-8")
    check("Frozen runtime actual dependency inventory exists, PSNR/SSIM bind loss_utils; first-before-backward/paired2000/RNG-finally/full4 stop guards retained", any({x.name for x in n.names} >= {"psnr", "ssim"} for n in real_import) and "utils.image_utils" not in harness and all(k in replica for k in ("utils/paper_losses.py", "utils/paper_masks.py", "utils/loss_utils.py", "utils/general_utils.py", "utils/sh_utils.py", "scene/paper_sgcm.py")) and all(s in harness for s in ("training.psnr is not loss_utils.psnr", "phase='before_actual_first_Tensor_backward'", "original_loss = float(log['loss_original_objective_before_diagnostic'])", "paired_actual != paired_baseline_rows[iteration]", "torch.Tensor.backward = original_tensor_backward", "rng_restore(rng_before)", "checkpoint_events", "cpu_load_iteration_schema_validated")))
    preparation = load(long / "v_binding_cpu/cpu_prepare_receipt.json")
    check("Actual CPU prepare exit0 binds actual new long binding/plan; V GPU restore/training unperformed and only root exact approval can start", preparation["actual_CPU_prepare_exit_code"] == 0 and preparation["CUDA_initialized"] is False and preparation["actual_GPU_restore"] is False and preparation["training_started"] is False and plan["allow_training"] is False and binding["allow_training"] is False and binding["training_started"] is False and preparation["binding_sha256"] == plan["experiment_binding_sha256"] == sha(materialized / "experiment_binding.json") and preparation["prepared_plan_sha256"] == sha(materialized / "opacity_outside_prepared_plan.json"))
    receipt = {"status": "PASS", "scope": "Independent CPU readiness review of actual long V source/config/binding/plan and pinned completed B1/B2 evidence; no GPU authorization or effect/normal/paper-reproduction acceptance", "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "checks": checks, "critical_file_sha256": pins, "reviewer_script_sha256": sha(__file__), "remote_source_plan_receipt_sha256": sha(here / "remote_actual_V_source_plan_cpu.json"), "initial_four30k_sha256": initial4, "GPU_used": False, "source_or_config_edited": False, "large_checkpoint_NPZ_input_rehash_repeated_by_reviewer": False}
    write(here / "actual_V_review_cpu.json", receipt)
    final = {"status": "PASS", "scope": receipt["scope"], "reviewed_utc": receipt["checked_utc"], "arm": "V", "harness_path": REMOTE_ROOT + "/" + LONG_REL + "/materialized_plan/continuation_variant_harness.py", "harness_sha256": plan["harness_sha256"], "prepared_plan_path": REMOTE_ROOT + "/" + LONG_REL + "/materialized_plan/opacity_outside_prepared_plan.json", "prepared_plan_sha256": sha(materialized / "opacity_outside_prepared_plan.json"), "experiment_binding_path": REMOTE_ROOT + "/" + LONG_REL + "/materialized_plan/experiment_binding.json", "experiment_binding_sha256": plan["experiment_binding_sha256"], "guard_sha256": plan["guard_sha256"], "launcher_sha256": plan["launcher_sha256"], "observer_sha256": plan["observer_sha256"], "metric_helper_sha256": EXPECTED["component_diagnostic/scripts/component_diagnostic.py"], "algorithm_changed_files": ALGORITHMS, "extra_config": EXTRAS, "first_iteration": 30001, "stop_iteration": 32000, "steps": 2000, "horizon": 50000, "seed": 0, "independent_CPU_review_sha256": sha(here / "actual_V_review_cpu.json"), "independent_remote_source_plan_review_sha256": receipt["remote_source_plan_receipt_sha256"], "locked_protocol_sha256": EXPECTED[LONG_REL + "/evaluation_protocol_locked.json"], "interpretation_preregistered_sha256": EXPECTED["sgcm_data_audit/long_pair_review/interpretation_preregistered.json"], "actual_B1_B2_pair_sha256": EXPECTED[LONG_REL + "/v_binding_cpu/pair_reference_validation.json"], "actual_original_repeat_noise_sha256": EXPECTED["continuation_diagnostic/original30k_to32000_noise_reference/receipts/B1_B2_observed_noise.json"], "actual_common49_inputs_after_B2_sha256": EXPECTED["continuation_diagnostic/original30k_to32000_noise_reference/receipts/common49_inputs_after_B2_cpu.json"], "architecture": "Same saved30k4state + unchanged467 inputs -> unchanged restore observer -> B1 original / B2 original repeat / V original+.05 outside-opacity -> same2000 sample-time-LR + RNG-safe4 eval nodes -> full49 paired domains +4 terminal states -> pre-locked per-frame/joint interpretation", "findings": ["Actual original/replica148 inventories and frozen execution/source/config hashes match; exactly train.py and paper_losses.py differ, with only the already-reviewed .05 complement term and log-only original loss capture. Formal guard and SGCM bytes unchanged.", "Same original30k complete initial state,50k horizon,2000 steps,4 fixed nodes and terminal49; no33000 reset. Actual B1/B2 initial restore, complete real execution and four validated states exist. All2000 camera/time_shift/LR match; first10 scalars match exactly. Later CUDA loss trajectories are not bitwise identical.", "Actual long B1/B2 noise and common467/245/98-mask receipts are newly bound, not the short30300 run. Canonical equal-frame means were independently recomputed from saved49 statistics; pixel weighting remains separately labelled. One repeat is empirical variation only.", "V readiness only. First original10 BEFORE actual backward, exact saved model/Adam/allRNG/sampler/SGCM restore,2000 paired inputs, RNG-preserved49 GT/domain checks and complete4-state terminal evidence remain mandatory runtime gates."], "limitations": ["No reviewer GPU execution or V model/optimizer/RNG restore occurred. Large saved checkpoint and NPZ/input rehashes were not repeated; their actual CPU receipts and unchanged source are pinned.", "No independent runtime mask pixels or actualScene camera-pose snapshots were saved; original encoded/decoded inputs, common loader source, actual paired GT arrays and domain counts bound the comparison to the documented extent.", "The .05 complementary domain is an unverified author-formula hypothesis; this is recovery from a degenerated30k state, not from-zero training or author paper acceptance. Preserve actor fidelity/composite costs/locality and real-image evidence in any effect conclusion.", "Do not infer normal separation from composite PSNR. B1/B2 static road remains poor and dynamic support widespread; no thresholds, source fields or formula weights were tuned after long results."], "root_action_required": "Root may separately approve only this exact V plan and give the existing GPU agent exclusive ownership; this CPU review launches nothing.", "GPU_used_by_reviewer": False, "formal_or_current_candidate_changed": False}
    path = here / "final_review_V.json"
    write(path, final)
    print(json.dumps({"status": final["status"], "path": str(path), "sha256": sha(path), "checks": len(checks), "harness_sha256": final["harness_sha256"], "prepared_plan_sha256": final["prepared_plan_sha256"], "experiment_binding_sha256": final["experiment_binding_sha256"]}))


if __name__ == "__main__":
    main()

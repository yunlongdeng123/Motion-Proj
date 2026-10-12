"""Write review receipts from already completed independent CPU evidence only."""
import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
LONG = AUDIT / "loss_audit/long_pair_30k_32000/materialized_plan"
REMOTE = "/root/autodl-tmp/drivingeditor_repro/campaign/vgpu_bugfix_20261012"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


review = load(HERE / "materialized_and_prepared_review_cpu.json")
actual = load(HERE / "remote_source_plan_receipt_cpu.json")
if review["status"] != "PASS_CPU_SOURCE_CONFIG_REVIEW_NO_GPU_AUTHORIZATION" or actual["status"] != "PASS" or not all(check["PASS"] for check in review["checks"]):
    raise RuntimeError("Independent CPU review not PASS")
receipts = {}
for role, subdir in (("B1", ""), ("B2", "baseline_noise_reference/")):
    directory = LONG / subdir
    plan = load(directory / "prepared_plan.json")
    if sha(directory / "prepared_plan.json") != review["actual_prepared_original_arms"][role]["sha256"]:
        raise RuntimeError("Actual prepared plan changed")
    payload = {
        "status": "PASS",
        "scope": "Independent CPU source/config/prepared-plan review only; not GPU authorization, runtime restore PASS, science PASS or normal-algorithm acceptance.",
        "reviewed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "arm": role,
        "harness_path": REMOTE + "/loss_audit/long_pair_30k_32000/materialized_plan/" + subdir + "continuation_harness.py",
        "harness_sha256": sha(directory / "continuation_harness.py"),
        "prepared_plan_path": REMOTE + "/loss_audit/long_pair_30k_32000/materialized_plan/" + subdir + "prepared_plan.json",
        "prepared_plan_sha256": sha(directory / "prepared_plan.json"),
        "launcher_sha256": sha(directory / "launch_controlled_continuation.py"),
        "observer_sha256": plan["observer_sha256"],
        "metric_helper_sha256": "0170283337ce3e2a1ea7a016e6f8a8e2b534101eebfe7b3ba4ecdcbcc1903baa",
        "original_short_binding_unchanged_sha256": "b96ec92c0694a251e39da85a3ae4abab131366a75ad52cd47098ec0df953b214",
        "original_short_harness_unchanged_sha256": "3a2f9add3cab0763c4b0ef612fd5b735afd7f2d337b151306da1a4ceffcb6154",
        "first_iteration": 30001,
        "stop_iteration": 32000,
        "steps": 2000,
        "horizon": 50000,
        "seed": 0,
        "training_config_changed_fields": ["model_path", "start_checkpoint"],
        "algorithm_changes": [],
        "loss_agent_locked_protocol_sha256": review["loss_agent_locked_protocol_sha256"],
        "independent_interpretation_sha256": review["independent_interpretation_sha256"],
        "independent_materialized_prepared_review_sha256": sha(HERE / "materialized_and_prepared_review_cpu.json"),
        "independent_actual_remote_source_plan_receipt_sha256": sha(HERE / "remote_source_plan_receipt_cpu.json"),
        "materializer_sha256": "b4d8fa110682acf7d45ef38fbe3faec65b5624f956b9a9c18fbfdf8be60be543",
        "findings": [
            "Actual17 staged source/config/draft files match their hashes; 7 executable/guard expansions contain only independent stop/node/path/routing changes. Two candidate algorithm overrides match short-window bytes exactly.",
            "Complete original128-field configuration retained with identical types/values except output/start-checkpoint routing; original arm adds no candidate diagnostic fields. B2 differs from B1 only in its independent output and plan paths.",
            "Actual CPU prepared plan rehashed61 protected original source/kernel/data/checkpoint inputs and binds the real new harness. Independent remote CPU review confirmed21 actual small source/config/plan/helper hashes.",
            "All arms restore original30k complete state; 30300 terminal is not reused. Window has no33000 static reset or densification/SGCM event; saved static/dynamic SH degree3 and original max-degree guard make retained32000 SH hook saturated.",
            "Existing diagnostic_eval math, restore observer and shared metrics bytes unchanged: four fixed nodes, first three fixed5 cameras, terminal all49, RNG capture/restore in finally and complete four-state terminal snapshot after actual32000 post-step.",
            "No new absolute quality margin: effects must be reported against B1 and B2 with empirical repeated-original differences, actor preservation, composite costs and real-image/removal-locality evidence. The two repeats do not establish statistical significance."
        ],
        "runtime_requirements_not_yet_proved": [
            "Actual GPU restore of parameters/Adam/allRNG/sampler/runtime/EMA/SGCM, complete finite real2000-step execution and four-state32000 snapshot.",
            "Exact all2000 camera/time_shift/LR pairing and identical all49 GT/dimensions/mask-region counts between completed B1/B2/V.",
            "Independent B1/B2 full-window noise receipt before small-effect interpretation; long V binding/plan/review must be newly generated from actual completed original references."
        ],
        "root_action_required": "Root may authorize only this exact arm with a separate exact receipt and exclusive GPU handoff. This review launches nothing.",
        "V_status": "NOT_LAUNCHABLE; real B1/B2 long references and new isolated binding/CPU plan/review remain pending.",
        "GPU_used_by_reviewer": False,
        "formal_or_current_candidate_changed": False
    }
    out = HERE / ("final_review_" + role + ".json")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    receipts[role] = {"path": str(out), "sha256": sha(out), "harness_sha256": payload["harness_sha256"], "prepared_plan_sha256": payload["prepared_plan_sha256"]}
print(json.dumps(receipts))

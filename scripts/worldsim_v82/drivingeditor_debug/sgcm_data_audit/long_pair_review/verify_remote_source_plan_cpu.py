"""Verify actual remote small source/plan bytes against independent CPU review."""
import datetime
import hashlib
import json
from pathlib import Path

BASE = Path("/root/autodl-tmp/drivingeditor_repro")
HERE = BASE / "campaign/vgpu_bugfix_20261012/sgcm_data_audit/long_pair_review"
LONG = BASE / "campaign/vgpu_bugfix_20261012/loss_audit/long_pair_30k_32000/materialized_plan"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


review = json.loads((HERE / "materialized_and_prepared_review_cpu.json").read_bytes())
pins = {str(LONG / relative): expected for relative, expected in review["materialized_file_sha256"].items()}
pins.update({
    str(LONG / "prepared_plan.json"): review["actual_prepared_original_arms"]["B1"]["sha256"],
    str(LONG / "baseline_noise_reference/prepared_plan.json"): review["actual_prepared_original_arms"]["B2"]["sha256"],
    str(BASE / "campaign/vgpu_bugfix_20261012/component_diagnostic/continuation_plan/runtime_safety_parameterized.py"): "f60faca00ec0d5b10b80d14d3e8ca07125ada5bff06d1f0fb8c43c5fec6d42d1",
    str(BASE / "campaign/vgpu_bugfix_20261012/component_diagnostic/scripts/component_diagnostic.py"): "0170283337ce3e2a1ea7a016e6f8a8e2b534101eebfe7b3ba4ecdcbcc1903baa",
})
actual = {path: sha(path) for path in pins}
if actual != pins:
    raise RuntimeError("Actual remote source/plan/helper SHA differs from independent CPU review")
receipt = {
    "status": "PASS",
    "verified_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "checked_small_file_count": len(pins),
    "actual_remote_file_sha256": actual,
    "independent_source_config_review_sha256": sha(HERE / "materialized_and_prepared_review_cpu.json"),
    "GPU_used": False,
    "torch_or_training_imported": False,
    "actual_saved_parameter_Adam_CUDA_RNG_restore_performed": False,
    "full_parent_checkpoint_bytes_rehash_in_this_check": False,
    "parent_protected_hash_evidence": "Each actual harness default CPU preparation already rehashed all61 protected inputs; this independent final check binds its real prepared-plan bytes and new source/helper bytes without repeating old large-file checks.",
    "review_script_sha256": sha(__file__),
}
out = HERE / "remote_source_plan_receipt_cpu.json"
out.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps({"status": receipt["status"], "checked_small_file_count": len(pins), "receipt_sha256": sha(out), "GPU_used": False}))

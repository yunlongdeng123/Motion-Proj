"""只用已有C00素材验证交互；明确fixture，禁止被作为真实四臂报告。"""
import argparse
import copy
import datetime
import json
from pathlib import Path
import shutil

from export_four_arm_assets import ARMS, comparison, read, require_terminal, write
from build_four_arm_viewer import build, validate_manifest


def reject(function, label):
    try:
        function()
    except ValueError:
        return label + ":PASS_REJECTED"
    raise AssertionError("Expected fail-closed check: " + label)


def fixture(c00_manifest, gate_plan, output):
    source = read(c00_manifest)
    gate = read(gate_plan)
    if output.exists():
        raise FileExistsError("Refuse to overwrite fixture")
    output.mkdir(parents=True)
    shutil.copytree(c00_manifest.parent / "assets", output / "assets/fixture_C00")
    records = {}
    for arm in ARMS:
        record = {"source": str(c00_manifest), "status": {"status": "fixture", "fixture_source_arm": "C00", "wall_seconds": None},
                  "memory": {"fixture": True, "complete_memory_evidence_available": False}, "node_contracts": [],
                  "diagnostics": {"fixture": True, "real_four_arm_diagnostics_available": False},
                  "frames": copy.deepcopy(source["frames"]), "static_time_frames": copy.deepcopy(source["static_time_frames"]),
                  "static_time_summary": source["static_time_summary"]}
        for frame in record["frames"] + record["static_time_frames"]:
            frame["images"] = {key: value.replace("assets/", "assets/fixture_C00/", 1) for key, value in frame["images"].items()}
        records[arm] = record
    data = {"schema_version": 1, "mode": "fixture_ui_only", "status": "FIXTURE_NOT_A_REAL_FOUR_ARM_RESULT", "title": "FIXTURE · C00素材四槽交互检查",
            "scene": source["scene"], "checkpoint": 30000, "frame_count": 49, "width": 480, "height": 320,
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "arm_order": list(ARMS), "arms": records,
            "metric_paths": gate["metrics"]["key_paths"], "comparison": comparison(records, gate["metrics"]["key_paths"]),
            "pairing": {"status": "FIXTURE_ONLY_NO_REAL_SAMPLER_VERIFICATION"}, "source_gate_plan": str(gate_plan),
            "GPU_used_by_export": False, "checkpoint_loaded": False, "restoration_verified": False,
            "gate_evaluation_performed": False, "human_verdict": None, "display": {"fixture": "Allfour UI slots repeat C00 assets; noG10/N01/GN11 result implied"},
            "scope": "Controlled UI fixture only; no scientific four-arm comparison or gate/human result"}
    checks = [reject(lambda: require_terminal({"status": "running", "arm": "GN11", "iteration": 29999, "iterations": 30000}, "GN11"), "nonterminal"),
              reject(lambda: validate_manifest(data), "fixture_requires_explicit_flag")]
    wrong = copy.deepcopy(data)
    wrong["arms"]["GN11"]["frames"][4]["timestamp"] += .01
    checks.append(reject(lambda: validate_manifest(wrong, allow_fixture=True), "wrong_paired_timestamp"))
    wrong_id = copy.deepcopy(data)
    wrong_id["arms"]["N01"]["frames"][10]["id"] = "image_0/0017048"
    checks.append(reject(lambda: validate_manifest(wrong_id, allow_fixture=True), "wrong_paired_frame"))
    validate_manifest(data, allow_fixture=True)
    write(output / "manifest.json", data)
    build(output / "manifest.json", output, allow_fixture=True)
    for arm in ARMS:
        for frame in records[arm]["frames"] + records[arm]["static_time_frames"]:
            if not all((output / path).is_file() for path in frame["images"].values()):
                raise AssertionError("Fixture references missing assets")
    report = {"status": "PASS_CONTROLLED_FIXTURE_SMOKE", "mode": "fixture_ui_only", "real_four_arm_result": False,
              "checks": checks, "frame_count": 49, "static_time_count": 49, "source_assets": "ExistingC00only",
              "GPU_used": False, "checkpoint_loaded": False, "gate_evaluation_performed": False, "human_verdict": None}
    write(output / "fixture_smoke.json", report)
    print(json.dumps(report, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c00-manifest", type=Path, required=True)
    parser.add_argument("--gate-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture(args.c00_manifest, args.gate_plan, args.output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import json
from pathlib import Path

base = Path("/root/autodl-tmp/drivingeditor_repro")
scenes = []
for name in ["0017085_demo_000_020", "0017085", "0145050", "0147030", "0158150"]:
    folder = base / "data/waymo_scenes" / name
    manifest = json.loads((folder / "preparation_manifest.json").read_text())
    validation = json.loads((folder / "validation.json").read_text())
    assert validation["status"] == "PASS"
    scenes.append({"scene": name, "path": str(folder), "frames": len(manifest["frames"]),
                   "source_frame_start": manifest["frames"][0]["source_index"],
                   "source_frame_end_inclusive": manifest["frames"][-1]["source_index"],
                   "camera": "FRONT", "initialization_points": validation["initialization_points"],
                   "sky_masks": manifest["sky_masks_present"], "validation": "PASS",
                   "raw_sha256": manifest["raw_sha256"], "ply_sha256": validation["ply_sha256"],
                   "raw_interval_seconds_median": manifest["raw_interval_seconds_median"],
                   "model_frame_interval": manifest["model_frame_interval"],
                   "exact_github_demo_match": "source_demo" in manifest})
summary = {"prepared_static_inputs": scenes, "total_frames": sum(s["frames"] for s in scenes),
           "total_initialization_points_across_input_copies": sum(s["initialization_points"] for s in scenes),
           "status": "all_static_inputs_validated", "training_status": "awaiting_environment_ready",
           "dynamic_inputs_status": "separate_real_actor_preparation_pending"}
diagnostics = []
for run, iteration in [("0017085_demo_000_020_static_smoke_100", 100),
                       ("0017085_demo_000_020_static_resume_101", 101),
                       ("0017085_demo_000_020_dynamic_smoke_100_retry", 100),
                       ("0017085_demo_000_020_dynamic_resume_101", 101)]:
    root = base / "runs" / run
    if (root / "smoke_telemetry.json").exists():
        telemetry = json.loads((root / "smoke_telemetry.json").read_text())
        metrics = json.loads((root / "eval" / f"train_{iteration}_render" / "metrics.json").read_text())
        diagnostics.append({"run": run, "telemetry": telemetry, "training_view_metrics": metrics})
if len(diagnostics) >= 2 and all(d["telemetry"]["success"] for d in diagnostics[:2]):
    summary["training_status"] = "static_training_and_resume_verified"
if len(diagnostics) == 4 and all(d["telemetry"]["success"] for d in diagnostics):
    summary["training_status"] = "static_and_dynamic_training_and_resume_verified"
actor_validation_path = base / "data/waymo_scenes/0017085_demo_000_020/dynamic_actor_validation.json"
if actor_validation_path.exists():
    actor_validation = json.loads(actor_validation_path.read_text())
    summary["dynamic_inputs_status"] = "validated_real_selected_actor" if actor_validation["status"] == "passed" else "validation_failed"
    summary["dynamic_actor"] = {key: value for key, value in actor_validation.items() if key != "frames_detail"}
summary["diagnostics"] = diagnostics
destination = base / "logs/data_preparation_summary.json"
destination.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))

#!/usr/bin/env python3
"""Audit approved DriveEditor cases against the paper's data-domain proxy.

The paper selects unobstructed objects within 20 m over consecutive frames.
Here we can verify camera distance and image projection, but not unobstructed
visibility; that component remains explicitly unverified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prepare_driveeditor_worldsim_v75_inputs import _counterfactual, _track
from motion_proj.cfbench.geometry import pose_at, project_box


def inspect_pose(scene: Path, frame: int, camera: int,
                 pose: np.ndarray | None, size: list[float] | None) -> dict:
    if pose is None or size is None:
        return {"track_present": False, "projected_in_frame": False,
                "camera_distance_m": None, "within_20m": False,
                "projected_area_px2": None}
    extrinsic = np.loadtxt(scene / "extrinsics" / f"{frame:03d}_{camera}.txt")
    distance = float(np.linalg.norm(pose[:3, 3] - extrinsic[:3, 3]))
    projection = project_box(scene, frame, camera, pose, size)
    return {"track_present": True, "projected_in_frame": projection is not None,
            "camera_distance_m": round(distance, 3), "within_20m": distance <= 20.0,
            "projected_area_px2": round(float(projection["area_px2"]), 1) if projection else None}


def summarize_window(frames: list[dict]) -> dict:
    return {
        "source_projected_frames": sum(row["source"]["projected_in_frame"] for row in frames),
        "desired_projected_frames": sum(row["desired"]["projected_in_frame"] for row in frames),
        "source_within_20m_frames": sum(row["source"]["within_20m"] for row in frames),
        "desired_within_20m_frames": sum(row["desired"]["within_20m"] for row in frames),
        "paper_training_domain_proxy_all_frames": all(row["domain_proxy_valid"] for row in frames),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = []
    for approved in manifest["cases"]:
        case = approved["case"]
        family = case["intervention"]["family"]
        if case["target"]["role"] == "ego":
            cases.append({"case_id": approved["case_id"], "status": "unsupported_ego"})
            continue
        scene = Path(case["dataset"]["root"]) / case["dataset"]["scene_id"]
        target = case["target"]
        key = str(target.get("actor_key") or target.get("source_asset_actor_key"))
        poses, sizes, _ = _track(scene, key)
        camera = int(approved["camera_index"])
        start, end = map(int, approved["source_frame_range"])
        event = int(case["anchor"]["event_frame"])
        frames = []
        for frame in range(start, end + 1):
            source_pose = pose_at(poses, float(frame))
            source_size = sizes.get(frame)
            desired_pose, desired_size = ((source_pose, source_size) if frame < event else
                                          _counterfactual(case, poses, sizes, frame))
            source = inspect_pose(scene, frame, camera, source_pose, source_size)
            desired = inspect_pose(scene, frame, camera, desired_pose, desired_size)
            if family == "actor_insertion":
                valid = desired["projected_in_frame"] and desired["within_20m"]
            elif family == "actor_removal":
                valid = source["projected_in_frame"] and source["within_20m"]
            else:
                valid = (source["projected_in_frame"] and source["within_20m"] and
                         desired["projected_in_frame"] and desired["within_20m"])
            frames.append({"clip_frame": frame - start, "source_frame": frame,
                           "source": source, "desired": desired, "domain_proxy_valid": bool(valid)})
        if len(frames) != 100:
            raise RuntimeError(f"{approved['case_id']}: expected exactly 100 frames")
        native_windows = [{"window": w, "clip_frames": [w * 10, w * 10 + 9],
                           **summarize_window(frames[w * 10:w * 10 + 10])}
                          for w in range(10)]
        iterative_windows = [{"window": w, "clip_frames": [w * 9, w * 9 + 9],
                              **summarize_window(frames[w * 9:w * 9 + 10])}
                             for w in range(11)]
        cases.append({"case_id": approved["case_id"], "status": "audited",
                      "operation_family": family,
                      "full_100_frame_domain_proxy": all(f["domain_proxy_valid"] for f in frames),
                      "valid_frames_of_100": sum(f["domain_proxy_valid"] for f in frames),
                      "native_10_frame_eligible_windows": [w["window"] for w in native_windows
                                                            if w["paper_training_domain_proxy_all_frames"]],
                      "native_windows": native_windows, "iterative_windows": iterative_windows,
                      "frames": frames})
    output = {"schema_version": "driveeditor_r9_paper_training_domain_audit_v1",
              "approved_manifest": str(args.manifest), "paper_condition":
              "training data selects unobstructed objects within a 20-meter radius of the camera across consecutive frames",
              "audited_proxy": "source/desired 3D center camera distance <=20m and projected in the approved camera; no occlusion/segmentation validation",
              "not_a_paper_benchmark_claim": True,
              "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"object_cases": sum(c["status"] == "audited" for c in cases),
                      "full_100_frame_proxy": sum(c.get("full_100_frame_domain_proxy", False) for c in cases),
                      "native_eligible_windows": sum(len(c.get("native_10_frame_eligible_windows", [])) for c in cases),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()

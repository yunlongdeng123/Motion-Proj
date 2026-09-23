#!/usr/bin/env python3
"""Export approved input geometry for auditable output-side metric masks.

These are *planned* boxes, never detections of a generated video.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_driveeditor_worldsim_v75_inputs import _counterfactual, _track  # noqa: E402
from motion_proj.cfbench.geometry import pose_at, project_box  # noqa: E402


def _projection(scene: Path, frame: int, camera: int, pose, size):
    if pose is None or size is None:
        return None
    p = project_box(scene, frame, camera, pose, size)
    return p["bbox_xyxy"] if p else None


def build_case(row: dict) -> dict:
    case = row["case"]
    result = {
        "schema_version": "cfbench_r9_planned_geometry_v1",
        "case_id": row["case_id"],
        "source_kind": "approved_nuScenes_3d_state_projected_to_original_camera",
        "warning": "planned condition only; not an observation of generated video",
        "original_image_size_wh": [1600, 900],
        "fps": 10,
        "event_frame_in_clip": int(case["anchor"]["event_frame"]) - int(row["source_frame_range"][0]),
        "target_role": case["target"]["role"],
        "family": case["intervention"]["family"],
        "frames": [],
    }
    if case["target"]["role"] == "ego":
        result["geometry_available"] = False
        result["reason"] = "ego intervention changes camera viewpoint; actor-box preservation mask is invalid"
        return result
    scene = Path(case["dataset"]["root"]) / case["dataset"]["scene_id"]
    target = case["target"]
    key = str(target.get("actor_key") or target.get("source_asset_actor_key"))
    poses, sizes, _ = _track(scene, key)
    camera = int(row["camera_index"])
    family = case["intervention"]["family"]
    event = int(case["anchor"]["event_frame"])
    source_start, source_end = row["source_frame_range"]
    assert source_end - source_start + 1 == 100
    for index, frame in enumerate(range(source_start, source_end + 1)):
        factual_pose = pose_at(poses, float(frame))
        factual_size = sizes.get(frame)
        factual_box = _projection(scene, frame, camera, factual_pose, factual_size)
        if frame < event:
            desired_pose, desired_size = factual_pose, factual_size
        else:
            desired_pose, desired_size = _counterfactual(case, poses, sizes, frame)
        desired_box = _projection(scene, frame, camera, desired_pose, desired_size)
        if family == "actor_insertion":
            allowed = [desired_box] if frame >= event and desired_box else []
        elif family == "actor_removal":
            allowed = [factual_box] if frame >= event and factual_box else []
        else:
            allowed = [box for box in (factual_box, desired_box) if box is not None] if frame >= event else []
        result["frames"].append({
            "index": index,
            "source_frame": frame,
            "factual_bbox_xyxy": factual_box,
            "desired_bbox_xyxy": desired_box if frame >= event and family != "actor_removal" else None,
            "allowed_edit_bbox_xyxy": allowed,
            "desired_world_center_m": (desired_pose[:3, 3].tolist() if desired_pose is not None
                                       and frame >= event and family != "actor_removal" else None),
        })
    result["geometry_available"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    assert manifest["case_count"] == 24 and all(r["approval"]["status"] == "approved" for r in manifest["cases"])
    args.output_root.mkdir(parents=True, exist_ok=True)
    counts = {"available": 0, "ego_not_applicable": 0}
    for row in manifest["cases"]:
        result = build_case(row)
        assert len(result["frames"]) == (100 if result["geometry_available"] else 0)
        (args.output_root / f"{row['case_id']}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        counts["available" if result["geometry_available"] else "ego_not_applicable"] += 1
    print(json.dumps(counts))


if __name__ == "__main__":
    main()

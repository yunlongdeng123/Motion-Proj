#!/usr/bin/env python3
"""Compile the user-approved 10 s cases into DriveEditor's native 1 s windows.

This is an adapter, not a claim that DriveEditor natively generates 10 s videos.
It never changes the approved scene, camera, actor, or intervention parameters.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_driveeditor_worldsim_v75_inputs import (  # noqa: E402
    _box,
    _counterfactual,
    _intrinsic,
    _reference_crop,
    _track,
)
from motion_proj.cfbench.geometry import pose_at, project_box  # noqa: E402


def compile_window(row: dict, window: int) -> tuple[dict, dict]:
    case = row["case"]
    if case["target"]["role"] != "non_ego":
        raise ValueError("DriveEditor has no native ego-editing operation")
    if not 0 <= window < 10:
        raise ValueError("window must be in [0, 9]")
    scene = Path(case["dataset"]["root"]) / case["dataset"]["scene_id"]
    target = case["target"]
    key = str(target.get("actor_key") or target.get("source_asset_actor_key"))
    poses, sizes, actor = _track(scene, key)
    camera = int(row["camera_index"])
    start = int(row["source_frame_range"][0]) + window * 10
    frames = list(range(start, start + 10))
    cls = str(target["class_name"])
    token = str(target.get("source_asset_actor_id") or target["entity_id"])
    visible = []
    for frame in range(int(row["source_frame_range"][0]), int(row["source_frame_range"][1]) + 1):
        p = pose_at(poses, float(frame))
        size = sizes.get(frame)
        if p is not None and size is not None:
            projected = project_box(scene, frame, camera, p, size)
            if projected is not None:
                visible.append((float(projected["area_px2"]), frame))
    if not visible:
        raise RuntimeError(f"{row['case_id']}: approved camera never shows source actor")
    ref_frame = max(visible)[1]
    ref, mask, ref_metrics = _reference_crop(
        scene, ref_frame, camera, poses[ref_frame], sizes[ref_frame], cls
    )
    operation = {
        "actor_speed_change": "Repositioning",
        "actor_lateral_relocation": "Repositioning",
        "actor_removal": "Deletion",
        "actor_insertion": "Insertion",
    }[case["intervention"]["family"]]
    payload_frames = []
    missing = []
    for frame in frames:
        p = pose_at(poses, float(frame))
        size = sizes.get(frame)
        # The legacy adapter accelerates timestamps even before the event;
        # the approved r9 pair instead fixes the entire factual prefix.
        if frame < int(case["anchor"]["event_frame"]):
            cf_p, cf_size = p, size
        else:
            cf_p, cf_size = _counterfactual(case, poses, sizes, frame)
        if p is None or size is None or cf_p is None or cf_size is None:
            missing.append(frame)
            continue
        image_path = scene / "images" / f"{frame:03d}_{camera}.jpg"
        image = np.asarray(Image.open(image_path).convert("RGB"))
        factual_box = _box(scene, frame, camera, p, size, cls, token)
        edited_box = _box(scene, frame, camera, cf_p, cf_size, cls, token)
        payload_frames.append({
            "im": image,
            "box": factual_box,
            "box_Repositioning": edited_box,
            "box_Insertion": edited_box,
        })
    if missing:
        raise RuntimeError(f"{row['case_id']} window {window}: missing actor states {missing}")
    name = f"{row['case_id']}__w{window:02d}"
    item = {
        "name": name,
        "camera_intrinsic": _intrinsic(scene, camera),
        "category_name": cls,
        "data": payload_frames,
        "im": ref,
        "mask": mask,
        "im_Insertion": [ref],
        "mask_Insertion": [mask],
        "im_Replacement": [ref],
        "mask_Replacement": [mask],
    }
    record = {
        "case_id": name,
        "approved_case_id": row["case_id"],
        "window_index": window,
        "window_source_frames": frames,
        "approved_camera_index": camera,
        "approved_scene_id": case["dataset"]["scene_id"],
        "operation": operation,
        "frame_count": 10,
        "reference_mask": ref_metrics,
        "native_duration_s": 1.0,
        "ten_second_output_requires_stitching": True,
        "ego_native_support": False,
    }
    return item, record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--window", type=int, action="append", default=[])
    args = parser.parse_args()
    if int(np.__version__.split(".", 1)[0]) != 1:
        raise RuntimeError("DriveEditor pickle input must be serialized with NumPy 1.x")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    assert manifest["case_count"] == 24 and manifest["status"] == "human_approved_no_inference"
    assert all(r["approval"]["status"] == "approved" for r in manifest["cases"])
    selected = set(args.case_id)
    rows = [r for r in manifest["cases"] if not selected or r["case_id"] in selected]
    if selected != {r["case_id"] for r in rows} and selected:
        raise RuntimeError(f"unknown requested cases: {sorted(selected - {r['case_id'] for r in rows})}")
    args.output.mkdir(parents=True, exist_ok=False)
    windows = args.window or list(range(10))
    records, errors = [], []
    for row in rows:
        if row["case"]["target"]["role"] == "ego":
            errors.append({"case_id": row["case_id"], "status": "unsupported_ego"})
            continue
        for window in windows:
            try:
                item, record = compile_window(row, window)
                path = args.output / f"{record['case_id']}.pkl"
                with path.open("wb") as handle:
                    pickle.dump([item], handle, protocol=pickle.HIGHEST_PROTOCOL)
                record["pickle"] = str(path.resolve())
                records.append(record)
            except Exception as exc:
                errors.append({"case_id": row["case_id"], "window": window,
                               "status": "input_compile_failed", "error": f"{type(exc).__name__}: {exc}"})
    index = {"schema_version": "driveeditor_r9_native_windows_v1",
             "approved_manifest": str(args.manifest.resolve()),
             "native_window_s": 1.0, "target_clip_s": 10.0,
             "cases": records, "unavailable": errors}
    (args.output / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"windows": len(records), "unavailable": len(errors),
                      "index": str(args.output / 'index.json')}))


if __name__ == "__main__":
    main()

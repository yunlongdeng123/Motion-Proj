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


def compile_window(row: dict, window: int, stride: int = 10,
                   factual_identity: bool = False,
                   reference_scope: str = "full_case") -> tuple[dict, dict]:
    case = row["case"]
    if case["target"]["role"] != "non_ego":
        raise ValueError("DriveEditor has no native ego-editing operation")
    if stride not in (9, 10) or not 0 <= window < (11 if stride == 9 else 10):
        raise ValueError("window/stride outside approved 100-frame clip")
    scene = Path(case["dataset"]["root"]) / case["dataset"]["scene_id"]
    target = case["target"]
    key = str(target.get("actor_key") or target.get("source_asset_actor_key"))
    poses, sizes, actor = _track(scene, key)
    camera = int(row["camera_index"])
    start = int(row["source_frame_range"][0]) + window * stride
    frames = list(range(start, start + 10))
    if frames[-1] > int(row["source_frame_range"][1]):
        raise ValueError(f"window {window} escapes approved clip")
    cls = str(target["class_name"])
    token = str(target.get("source_asset_actor_id") or target["entity_id"])
    visible = []
    reference_frames = (frames if reference_scope == "window" else
                        range(int(row["source_frame_range"][0]), int(row["source_frame_range"][1]) + 1))
    for frame in reference_frames:
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
    if factual_identity:
        operation = "Repositioning"
    payload_frames = []
    missing = []
    for frame in frames:
        p = pose_at(poses, float(frame))
        size = sizes.get(frame)
        # The legacy adapter accelerates timestamps even before the event;
        # the approved r9 pair instead fixes the entire factual prefix.
        if factual_identity or frame < int(case["anchor"]["event_frame"]):
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
    name = (f"{row['case_id']}__factual_w{window:02d}" if factual_identity
            else f"{row['case_id']}__w{window:02d}")
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
        "evaluation_role": "factual_identity_reconstruction" if factual_identity else "counterfactual_edit",
        "frame_count": 10,
        "reference_mask": ref_metrics,
        "reference_frame": ref_frame,
        "reference_scope": reference_scope,
        "native_duration_s": 1.0,
        "window_stride_frames": stride,
        "condition_previous_last_frame": stride == 9 and window > 0,
        "overlap_frame": frames[0] if stride == 9 and window > 0 else None,
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
    parser.add_argument("--stride", type=int, choices=[9, 10], default=10,
                        help="9 means one-frame overlap for iterative conditioning")
    parser.add_argument("--factual-identity", action="store_true",
                        help="Paper reconstruction control: mask and regenerate source object at unchanged 3D boxes")
    parser.add_argument("--reference-scope", choices=["full_case", "window"], default="full_case",
                        help="window matches a native 10-frame reconstruction clip; full_case keeps a consistent asset across iterative segments")
    args = parser.parse_args()
    if args.factual_identity and args.stride != 10:
        raise RuntimeError("factual identity reconstruction is a native 10-frame control")
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
    windows = args.window or list(range(11 if args.stride == 9 else 10))
    records, errors = [], []
    for row in rows:
        if row["case"]["target"]["role"] == "ego":
            errors.append({"case_id": row["case_id"], "status": "unsupported_ego"})
            continue
        for window in windows:
            try:
                item, record = compile_window(row, window, args.stride,
                                              args.factual_identity, args.reference_scope)
                path = args.output / f"{record['case_id']}.pkl"
                with path.open("wb") as handle:
                    pickle.dump([item], handle, protocol=pickle.HIGHEST_PROTOCOL)
                record["pickle"] = str(path.resolve())
                records.append(record)
            except Exception as exc:
                errors.append({"case_id": row["case_id"], "window": window,
                               "status": "input_compile_failed", "error": f"{type(exc).__name__}: {exc}"})
    index = {"schema_version": ("driveeditor_r9_factual_identity_native_windows_v1" if args.factual_identity
                                else "driveeditor_r9_iterative_overlap_v1" if args.stride == 9
                                else "driveeditor_r9_native_windows_v1"),
             "approved_manifest": str(args.manifest.resolve()),
             "native_window_s": 1.0, "target_clip_s": 10.0,
             "evaluation_role": "factual_identity_reconstruction" if args.factual_identity else "counterfactual_edit",
             "window_stride_frames": args.stride,
             "cases": records, "unavailable": errors}
    (args.output / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"windows": len(records), "unavailable": len(errors),
                      "index": str(args.output / 'index.json')}))


if __name__ == "__main__":
    main()

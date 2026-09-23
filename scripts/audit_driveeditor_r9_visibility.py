#!/usr/bin/env python3
"""Read-only keyframe visibility audit for DriveEditor's approved r9 windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_driveeditor_worldsim_v75_inputs import _counterfactual, _track  # noqa: E402
from motion_proj.cfbench.geometry import pose_at, project_box  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    assert manifest["case_count"] == 24 and all(r["approval"]["status"] == "approved" for r in manifest["cases"])
    rows = {r["case_id"]: r for r in manifest["cases"]}
    inputs = json.loads(args.input_index.read_text(encoding="utf-8"))
    results = []
    for entry in inputs["cases"]:
        row = rows[entry["approved_case_id"]]
        case = row["case"]
        scene = Path(case["dataset"]["root"]) / case["dataset"]["scene_id"]
        target = case["target"]
        key = str(target.get("actor_key") or target.get("source_asset_actor_key"))
        poses, sizes, _ = _track(scene, key)
        family = case["intervention"]["family"]
        visible = []
        for frame in entry["window_source_frames"]:
            if family == "actor_removal":
                pose = pose_at(poses, frame)
                size = sizes.get(frame)
            elif frame < int(case["anchor"]["event_frame"]) and family != "actor_insertion":
                pose = pose_at(poses, frame)
                size = sizes.get(frame)
            else:
                pose, size = _counterfactual(case, poses, sizes, frame)
            projection = project_box(scene, frame, int(row["camera_index"]), pose, size) if pose is not None and size is not None else None
            visible.append(None if projection is None else float(projection["area_px2"]))
        results.append({"case_id": entry["approved_case_id"], "window": entry["window_index"],
                        "keyframe_visible": visible[0] is not None,
                        "keyframe_area_px2": visible[0],
                        "visible_frames": sum(area is not None for area in visible),
                        "areas_px2": visible})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": "driveeditor_r9_visibility_audit_v1",
                                       "windows": results}, indent=2) + "\n", encoding="utf-8")
    missing = [f"{r['case_id']}__w{r['window']:02d}" for r in results if not r["keyframe_visible"]]
    print(json.dumps({"windows": len(results), "keyframe_not_visible": len(missing), "ids": missing}))


if __name__ == "__main__":
    main()

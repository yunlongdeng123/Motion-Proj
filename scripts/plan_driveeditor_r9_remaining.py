#!/usr/bin/env python3
"""Make a no-duplicate continuation index for visible DriveEditor windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-index", type=Path, required=True)
    parser.add_argument("--visibility-audit", type=Path, required=True)
    parser.add_argument("--completed-root", type=Path, action="append", default=[])
    parser.add_argument("--reserved-case-id", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = json.loads(args.input_index.read_text(encoding="utf-8"))
    audit = json.loads(args.visibility_audit.read_text(encoding="utf-8"))
    assert audit["schema_version"] == "driveeditor_r9_visibility_audit_v1"
    visibility = {(r["case_id"], r["window"]): r for r in audit["windows"]}
    assert len(visibility) == len(original["cases"]) == 180
    reserved = set(args.reserved_case_id)
    queued, excluded = [], []
    for row in original["cases"]:
        cid = row["approved_case_id"]
        name = row["case_id"]
        vis = visibility[(cid, row["window_index"])]
        reason = None
        if not vis["keyframe_visible"]:
            reason = "native_keyframe_offscreen; passthrough and disclose coverage"
        elif cid in reserved:
            reason = "reserved_by_active_run"
        else:
            for root in args.completed_root:
                result_path = root / name / "result.json"
                video_path = root / name / "counterfactual.mp4"
                if result_path.is_file() and video_path.is_file():
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    if result.get("status") == "generation_complete" and result.get("steps") == 25:
                        reason = "already_generated"
                        break
        if reason:
            excluded.append({"case_id": name, "reason": reason})
        else:
            queued.append(row)
    # Keep the GPU busy with windows where the target remains visible for
    # all 10 frames before attempting partial-visibility boundaries.
    queued.sort(key=lambda row: visibility[(row["approved_case_id"], row["window_index"])]["visible_frames"] < 10)
    plan = {"schema_version": "driveeditor_r9_continuation_v1", "cases": queued,
            "excluded": excluded, "source_index": str(args.input_index.resolve()),
            "visibility_audit": str(args.visibility_audit.resolve()),
            "queue_order": "stable full-visibility windows first; then partial-visibility windows"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError(f"preserving existing continuation plan: {args.output}")
    args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"queued": len(queued), "excluded": len(excluded),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()

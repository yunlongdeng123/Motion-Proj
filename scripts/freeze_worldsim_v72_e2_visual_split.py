"""Freeze a quality-blind log split for train scenes with RGB and V7/V7.1 actors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_proj.worldsim_v72.data.nuscenes_camera import NuScenesCameraIndex
from motion_proj.worldsim_v72.data.splits import load_data_roles, require_role_access


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--actor-corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-windows", type=int, default=4)
    parser.add_argument("--development-fraction", type=float, default=0.2)
    parser.add_argument("--salt", default="worldsim-v72-e2-visual-split-v1")
    args = parser.parse_args()
    channels = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT")
    index = NuScenesCameraIndex(args.dataset_root)
    roles = load_data_roles(args.roles)
    allowed_logs = set(require_role_access(roles, "nuscenes", "train"))
    corpus_scenes = {path.name for path in args.actor_corpus_root.iterdir() if path.is_dir()}
    counts = {scene: 0 for scene in corpus_scenes}
    for sample in index.samples:
        scene = index.scene_by_token[str(sample["scene_token"])]
        scene_id = str(scene["name"])
        if scene_id not in corpus_scenes or str(scene["log_token"]) not in allowed_logs:
            continue
        complete = True
        for channel in channels:
            token = index.data_by_sample_channel.get((str(sample["token"]), channel), "")
            row = index.sample_data.get(token)
            if row is None or not (index.dataset_root / str(row["filename"])).is_file():
                complete = False
                break
        if complete:
            counts[scene_id] += 1
    scene_rows = {str(row["name"]): row for row in index.scenes}
    eligible = [scene for scene, count in counts.items() if count >= args.minimum_windows]
    logs = sorted({str(scene_rows[scene]["log_token"]) for scene in eligible})
    ranked = sorted(logs, key=lambda log: hashlib.sha256(f"{args.salt}:{log}".encode()).hexdigest())
    development_count = max(1, min(len(ranked) - 1, round(len(ranked) * args.development_fraction)))
    development_logs = set(ranked[:development_count])
    rows = [
        {
            "scene_id": scene,
            "log_id": str(scene_rows[scene]["log_token"]),
            "payload_complete_window_count": counts[scene],
            "role": "development" if str(scene_rows[scene]["log_token"]) in development_logs else "fit",
        }
        for scene in sorted(eligible)
    ]
    payload = {
        "schema_version": "worldsim_v72.e2_visual_split.v1",
        "selection": "V7/V7.1 actor corpus intersection with three-camera payload availability",
        "split_unit": "log_id",
        "split_method": "sha256_rank",
        "salt": args.salt,
        "minimum_windows": args.minimum_windows,
        "camera_channels": list(channels),
        "quality_used_for_selection": False,
        "supervision_used_for_selection": False,
        "eligible_scene_count": len(rows),
        "eligible_log_count": len(logs),
        "fit_log_count": len(logs) - development_count,
        "development_log_count": development_count,
        "fit_scenes": [row["scene_id"] for row in rows if row["role"] == "fit"],
        "development_scenes": [row["scene_id"] for row in rows if row["role"] == "development"],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("eligible_scene_count", "eligible_log_count", "fit_log_count", "development_log_count")}))


if __name__ == "__main__":
    main()

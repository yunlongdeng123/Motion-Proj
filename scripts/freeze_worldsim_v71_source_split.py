"""冻结 V7.1 nuScenes metadata-only train/selection/final split。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_proj.worldsim_v71.dataset_nuscenes import freeze_source_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    source = yaml.safe_load((repo_root / config["p4_config"]).read_text(encoding="utf-8"))
    prior = [
        name
        for names in source["nuscenes"]["role_scenes"].values()
        for name in names
    ]
    split = freeze_source_split(
        Path(config["source"]["dataset_root"]),
        prior_scene_names=prior,
        train_count=int(config["source"]["train_scene_count"]),
        selection_count=int(config["source"]["selection_scene_count"]),
        final_count=int(config["source"]["final_scene_count"]),
        minimum_lidar_frames=int(config["source"]["minimum_lidar_frames"]),
    )
    output = repo_root / config["source_split"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "role_counts": split["role_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

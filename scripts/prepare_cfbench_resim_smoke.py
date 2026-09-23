"""为官方 49 帧接口准备一个配对 smoke；不改写 24-case 冻结输入。"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = Path("/root/autodl-tmp/data/worldsim_v75_downstream_bench/adapter_inputs/resim/CFB-SPEED-EGO-01/factual.yaml")
    root = Path("/root/autodl-tmp/data/dynamic_editing_v2/drivestudio_processed_10Hz/trainval")
    poses = {int(p.stem): np.loadtxt(p) for p in (root / "179/lidar_pose").glob("*.txt")}
    origin = poses[64]
    images = [f"179/images/{i:03d}_0.jpg" for i in range(56, 105)]
    assert all((root / p).is_file() for p in images)
    for branch, scale in [("factual", 1.0), ("counterfactual", 0.5)]:
        trajectories = []
        for offset in np.arange(5, 41, 5) * scale:
            lower, upper = math.floor(64 + offset), math.ceil(64 + offset)
            weight = 64 + offset - lower
            pose = poses[lower].copy()
            pose[:3, 3] = poses[lower][:3, 3] * (1-weight) + poses[upper][:3, 3] * weight
            from scipy.spatial.transform import Rotation, Slerp
            if lower != upper:
                pose[:3, :3] = Slerp([0, 1], Rotation.from_matrix(np.stack([poses[lower][:3, :3], poses[upper][:3, :3]])))([weight]).as_matrix()[0]
            relative = np.linalg.inv(origin) @ pose
            trajectories.append([float(relative[0, 3]), float(relative[1, 3]), math.atan2(relative[1, 0], relative[0, 0])])
        data_path = out / f"{branch}.json"
        data_path.write_text(json.dumps({"meta": {"data_root": str(root)}, "clips": [{"img_seq": images, "cmd": "Moving_Forward", "traj_fut": trajectories, "lidar_pc_token": f"cfbench-smoke-{branch}"}]}, indent=2) + "\n")
        cfg = yaml.safe_load(source.read_text())
        cfg["args"].update(valid_data=[str(data_path)], use_ema=False, save_gt=False, concat_gt_for_demo=False, seed=42)
        cfg["data"]["params"].pop("n_subset", None)
        cfg["data"]["params"].pop("ind_subset", None)
        (out / f"cfbench-resim-smoke-{branch}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "input-contract.json").write_text(json.dumps({
        "input_role": "native_horizon_engineering_smoke", "pilot_source_case": "CFB-SPEED-EGO-01",
        "benchmark_aligned": False, "reason": "官方49帧/9帧条件/40帧未来接口与pilot24帧/5帧前缀不同；不计入正式case分母",
        "source_frames": list(range(56, 105)), "condition_rgb_frames": list(range(56, 65)),
        "trajectory_origin": 64, "trajectory_offsets_s": [0.5,1,1.5,2,2.5,3,3.5,4],
        "speed_scale": [1, 0.5], "seed": 42, "human_verdict": None,
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

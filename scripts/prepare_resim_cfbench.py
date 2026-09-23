"""ReSim原生49帧接口映射冻结case；未来RGB用末帧占位，禁止偷看事实未来。"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

from prepare_omnidreams_cfbench import changed_pose, pose_at


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.manifest.read_text())["cases"]
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for case in cases:
        cid = case["case_id"]
        if case["target"]["role"] != "ego":
            records.append({"case_id": cid, "status": "unsupported", "reason": "公开ReSim只接收ego轨迹，不支持指定非ego actor干预", "scores": None})
            continue
        out = args.output / cid
        out.mkdir()
        root = Path(case["dataset"]["root"])
        scene = case["dataset"]["scene_id"]
        poses = {int(p.stem): np.loadtxt(p) for p in (root / scene / "lidar_pose").glob("*.txt")}
        event = case["anchor"]["event_frame"]
        origin = poses[event-1]
        history = list(range(event-9, event))
        # 49帧加载器/VAE需要全张量；未观测40帧全用最后一帧，双方完全一致。
        images = [f"{scene}/images/{f:03d}_0.jpg" for f in history+[event-1]*40]
        assert all((root / p).is_file() for p in images)
        configs = {}
        for branch in ["factual", "counterfactual"]:
            waypoints = []
            for offset in range(5, 41, 5):
                frame = event-1+offset
                pose = pose_at(poses, frame) if branch == "factual" else changed_pose(case, poses, frame)
                assert pose is not None, (cid, frame)
                relative = np.linalg.inv(origin) @ pose
                waypoints.append([float(relative[0, 3]), float(relative[1, 3]), math.atan2(relative[1, 0], relative[0, 0])])
            data = {"meta": {"data_root": str(root)}, "clips": [{"img_seq": images, "cmd": "Moving_Forward", "traj_fut": waypoints, "lidar_pc_token": f"{cid}-{branch}"}]}
            data_path = out / f"{branch}.json"
            data_path.write_text(json.dumps(data, indent=2)+"\n")
            cfg = yaml.safe_load(args.template.read_text())
            cfg["args"].update(valid_data=[str(data_path)], use_ema=False, save_gt=False, concat_gt_for_demo=False, seed=42)
            cfg["data"]["params"].pop("n_subset", None)
            cfg["data"]["params"].pop("ind_subset", None)
            path = out / f"{cid}-{branch}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            configs[branch] = str(path)
        record = {"case_id": cid, "case": case, "status": "prepared", "configs": configs,
                  "native_frames": 49, "fps": 10, "condition_source_frames": history,
                  "score_native_indices_inclusive": [4, 27], "score_source_frames_inclusive": [event-5, event+18],
                  "event_native_index": 9, "scored_frames": 24, "camera_index": 0,
                  "information_boundary": "9 factual past RGB; no future RGB; GT ego trajectories; fixed text; seed42",
                  "adapter_notes": ["4 extra historical frames required by native interface, outside common scored window",
                                    "native4s trajectory extends same declared intervention beyond common1.8s future window",
                                    "17-frame temporal VAE chunking changes numerical context relative to49-frame encoding",
                                    "49frame VAE input padded after observed9frames to prevent future RGB leakage"],
                  "human_verdict": None}
        (out / "input.json").write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n")
        records.append(record)
    (args.output / "index.json").write_text(json.dumps({"task_id": "WS-V75-DOWNSTREAM-FULL-01", "model_id": "resim", "cases": records}, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"cases": len(records), "prepared": sum(x["status"] == "prepared" for x in records), "unsupported": sum(x["status"] == "unsupported" for x in records)}))


if __name__ == "__main__":
    main()

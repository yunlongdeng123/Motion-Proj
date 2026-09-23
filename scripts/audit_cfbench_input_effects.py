"""固定case的后验输入可辨认性诊断；不修改case、不按结果删样本。"""
import argparse
import json
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

from prepare_omnidreams_cfbench import changed_pose, pose_at


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    results = []
    for row in json.loads((args.inputs / "index.json").read_text())["cases"]:
        src, case = Path(row["input_dir"]), row["case"]
        tr = np.load(src / "trajectory.npz")
        root = Path(case["dataset"]["root"]) / case["dataset"]["scene_id"]
        if case["target"]["role"] == "ego":
            poses = {int(f.stem): np.loadtxt(f) for f in (root / "lidar_pose").glob("*.txt")}
        else:
            info = json.loads((root / "instances/instances_info.json").read_text())[row["target_key"]]["frame_annotations"]
            poses = {int(f): np.array(pose) for f,pose in zip(info["frame_idx"], info["obj_to_world"])}
        polygon_data = json.loads((src / "drivable.json").read_text())
        area = unary_union([Polygon(p["outer"], p["holes"]).buffer(0) for p in polygon_data])
        original, edited = [], []
        for f in tr["source_frames"][15:70]:
            original.append(pose_at(poses, f)[:3, 3])
            edited.append(changed_pose(case, poses, f)[:3, 3])
        original, edited = np.array(original), np.array(edited)
        family = case["intervention"]["family"]
        def path_length(x):
            return float(np.linalg.norm(np.diff(x, axis=0), axis=1).sum())
        record = {"case_id": row["case_id"], "role": "posthoc_input_diagnostic_not_pre_registered_scoring_gate",
                  "target_role": case["target"]["role"], "family": family,
                  "factual_target_path_length_m": path_length(original), "counterfactual_target_path_length_m": path_length(edited),
                  "target_endpoint_delta_m": float(np.linalg.norm(edited[-1]-original[-1])) if family not in ["actor_removal", "actor_insertion"] else None,
                  "counterfactual_center_in_drivable_fraction": float(np.mean([area.covers(Point(p[:2]-np.array(row["origin_world"])[:2])) for p in edited])) if family != "actor_removal" else None,
                  "road_audit_caveat": "仅目标中心与drivable_area；不验证车身、车道方向、信号灯或自行车/行人路权；移除后N/A",
                  "frozen_case_unchanged": True, "human_verdict": None}
        out = args.run / row["case_id"] / "input-audit.json"
        out.write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n")
        results.append(record)
    (args.run / "input-audit.json").write_text(json.dumps(results, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""CPU 验收真实原生 checkpoint event 及下一阶段读取入口。"""
import argparse
import json
from pathlib import Path

import numpy as np


def inspect(run):
    run = Path(run).resolve()
    latest = json.loads((run / "latest_complete_node.json").read_text())
    event_path = Path(latest["event"])
    event = json.loads(event_path.read_text())
    iteration = latest["iteration"]
    if event["status"] != "complete" or event["iteration"] != iteration:
        raise ValueError("最近检查点未完整发布")
    if set(event["checkpoints"]) != {"static", "dynamic", "environment", "training_state"}:
        raise ValueError("原生静态/动态/环境/完整恢复 sidecar 缺失")
    for row in event["checkpoints"].values():
        path = Path(row["path"])
        if not path.is_file() or path.stat().st_size != row["bytes"] or row["bytes"] <= 0:
            raise ValueError("检查点文件缺失或长度不符: " + str(path))
    train_evaluations = [row for row in event["evaluations"] if row["split"] == "train"]
    if len(train_evaluations) != 1:
        raise ValueError("缺少唯一完整的训练视图评价")
    evaluation = train_evaluations[0]
    identities = event["config"]["followup_scene_spec"]["frame_ids_with_camera"]
    if set(frame["frame_id"] for frame in evaluation["frames"]) != set(identities) or len(evaluation["frames"]) != len(identities):
        raise ValueError("评价视图集合与 manifest 不符")
    checked_arrays = 0
    for frame in evaluation["frames"]:
        if set(frame["channels"]) != {"gt", "composite", "static", "dynamic"}:
            raise ValueError("分层原生评价输出不完整")
        if any(not Path(path).is_file() for path in frame["channels"].values()):
            raise ValueError("分层评价图像缺失")
        if "hybrid_metrics" not in frame:
            raise ValueError("缺少未经mask裁剪的 hybrid raw 指标")
        metrics = json.loads(Path(frame["hybrid_metrics"]).read_text())
        with np.load(metrics["arrays"], allow_pickle=False) as arrays:
            for name in ("gt", "composite", "static", "dynamic", "alpha_d", "alpha_s", "depth_accum_model"):
                array = arrays[name]
                if not np.isfinite(array).all() or list(array.shape[-2:]) != [320, 480]:
                    raise ValueError("原生 FP32 分层评价出现非有限或错误形状: " + name)
                checked_arrays += 1
    for key in ("metrics", "paper_metrics"):
        if evaluation.get(key) and not Path(evaluation[key]).is_file():
            raise ValueError("原生评价汇总缺失: " + key)
    static_time = run / ("static_time_%d" % iteration) / "summary.json"
    if not static_time.is_file():
        raise ValueError("真实固定姿态静态时间检查未执行")
    entry = json.loads((run / "next_stage_entry.json").read_text())
    if entry["iteration"] != iteration or entry["event"] != str(event_path.resolve()):
        raise ValueError("下一阶段入口没有指向同一个完整 bundle")
    if entry["independent_reference_used_for_training"] is not False:
        raise ValueError("独立参考角色声明不符")
    binding = json.loads((run / "actual_scene_binding.json").read_text())
    if binding["frame_ids"] != identities:
        raise ValueError("真实 Scene 绑定与检查点身份不符")
    return dict(status="PASS_NATIVE_BUNDLE_AND_NEXT_STAGE_ENTRY", iteration=iteration,
        event=str(event_path.resolve()), frames=len(identities), checked_finite_raw_arrays=checked_arrays,
        complete_components=list(event["checkpoints"]), next_stage_entry=str((run / "next_stage_entry.json").resolve()),
        independent_reference_used_for_training=False, scientific_contamination_claim=False,
        scope="CPU artifact/readability acceptance; exact model/optimizer/RNG restore is verified by real resumed CUDA worker")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect(args.run)
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

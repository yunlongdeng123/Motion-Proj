"""逐帧 RGB/车身/道路分母与连续编辑，全部图像来自新场景；没有原图混合。"""
import argparse
from pathlib import Path
import math
import json
from dataclasses import asdict
import numpy as np
from PIL import Image
import torch
from . import SCHEMA
from .data import Dataset
from .config import Config
from .model import SceneModel
from .train import seed_from_state, write
from .losses import masked_mean
from .math3d import linear_to_srgb


def save_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = (linear_to_srgb(image).clamp(0, 1)*255).round().to(torch.uint8).cpu().numpy()
    Image.fromarray(pixels).save(path)


def encode(path, fps=10):
    import cv2
    files = sorted(path.glob("*.png"))
    first = np.asarray(Image.open(files[0]))
    h, w = first.shape[:2]
    video = cv2.VideoWriter(str(path.with_suffix(".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not video.isOpened():
        raise RuntimeError("视频编码器不可用；保留所有 PNG，不能报告完整视频通过")
    for file in files:
        video.write(cv2.cvtColor(np.asarray(Image.open(file)), cv2.COLOR_RGB2BGR))
    video.release()
    decoder = cv2.VideoCapture(str(path.with_suffix(".mp4")))
    decoded = 0
    while True:
        ok, _ = decoder.read()
        if not ok:
            break
        decoded += 1
    decoder.release()
    if decoded != len(files):
        raise RuntimeError("视频没有全部帧成功解码")
    return dict(frames=decoded, path=str(path.with_suffix(".mp4")))


@torch.no_grad()
def evaluate(model, dataset, output, backend="cuda", target=0, videos=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    stream_names = ("observed", "shadow_off", "delete_target", "delete_all", "translate", "yaw", "coverage", "fixed_delete_all")
    for name in stream_names:
        (output/name).mkdir()
    anchor = dataset.frame(len(dataset.rows)//2)
    fixed_reference = None
    for index in range(len(dataset.rows)):
        frame = dataset.frame(index)
        t = index/max(1, len(dataset.rows)-1)
        geometry, package = model.buffers(frame, len(dataset.actor_ids), backend)
        image, shadow = model.full_image(package, geometry)
        clean, _ = model.full_image(package, geometry, shadows=False)
        images = dict(observed=image, shadow_off=clean,
                      coverage=package["alpha"][..., None].expand_as(image))
        for name, edits in (("delete_target", {target: {"delete": True}}),
                            ("delete_all", {i: {"delete": True} for i in range(len(dataset.actor_ids))}),
                            ("translate", {target: {"translation": [math.sin(2*math.pi*t), 0, 0]}}),
                            ("yaw", {target: {"yaw": math.radians(30)*math.sin(2*math.pi*t)}})):
            g, p = model.buffers(frame, len(dataset.actor_ids), backend, edits, isolated=False)
            images[name], _ = model.full_image(p, g)
            if name == "delete_all":
                off, _ = model.full_image(p, g, shadows=False)
                deletion_identity = float((off-images[name]).abs().max())
        # 保持同一相机/背景，仅换 Actor 时刻并全部删除；验证无时间自由度。
        fixed_frame = type(frame)(frame.frame_id, frame.index, frame.split, anchor.camera, anchor.rgb,
                                  anchor.masks, anchor.mask_known, anchor.road, anchor.depth, frame.poses, frame.valid)
        g, p = model.buffers(fixed_frame, len(dataset.actor_ids), backend,
                             {i: {"delete": True} for i in range(len(dataset.actor_ids))}, isolated=False)
        images["fixed_delete_all"], _ = model.full_image(p, g)
        if fixed_reference is None:
            fixed_reference = images["fixed_delete_all"].clone()
        body = (frame.masks & frame.mask_known[None, None]).any(-1)
        for name, value in images.items():
            save_png(output/name/(frame.frame_id+".png"), value)
        error = (image-frame.rgb).abs()
        records.append(dict(frame_id=frame.frame_id, split=frame.split,
                            rgb_mae=float(error.mean()), body_mae=float(masked_mean(error, body)) if body.any() else None,
                            body_pixels=int(body.sum()), road_observed_rgb_mae=float(masked_mean(error, frame.road)) if frame.road.any() else None,
                            road_pixels=int(frame.road.sum()), visible_identity_l1=float(masked_mean((package["visible_labels"]-frame.masks.float()).abs(), frame.mask_known[None, None])),
                            visible_identity_elements=int(frame.mask_known.sum())*frame.rgb.shape[0]*frame.rgb.shape[1],
                            low_coverage_pixels=int((package["alpha"] < .5).sum()),
                            delete_all_shadow_off_max=deletion_identity,
                            fixed_static_max=float((images["fixed_delete_all"]-fixed_reference).abs().max())))
        write(output/"metrics.json", dict(frames=records, reference="train pseudo-reference; RGB test split independent of new branch only",
                                          paired_edit_ground_truth_available=False, human_verdict=None))
    encoded = [encode(output/name) for name in stream_names] if videos else []
    write(output/"evaluation.json", dict(frames=len(records), videos=encoded, real_scene=not dataset.manifest.get("synthetic_fixture", False),
                                         initialization="raw_measurements", complete_manual_visual_review=False,
                                         clean_GT_available=False, problem_solved=None, human_verdict=None))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--target", type=int, default=0)
    selection.add_argument("--track-id", help="优先使用真实 raw track ID，避免新旧 Actor 排序不同")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(status="GPU_EVALUATION_NOT_STARTED", requires="explicit --execute on user-started GPU")))
        return
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != SCHEMA:
        raise ValueError("只能加载新场景 checkpoint")
    raw = checkpoint["config"].copy()
    from .config import LossWeights
    raw["weights"] = LossWeights(**raw["weights"])
    config = Config(**raw).validate()
    dataset = Dataset(args.data, "cuda")
    if checkpoint.get("input_contract") != dataset.manifest:
        raise ValueError("评价数据与 checkpoint 的实例/相机契约不符")
    model = SceneModel(seed_from_state(checkpoint["state"]), config).cuda()
    model.load_state_dict(checkpoint["state"], strict=True)
    model.requires_grad_(False)
    from .gpu_probe import probe
    probe(config)
    target = dataset.actor_ids.index(args.track_id) if args.track_id else args.target
    if not 0 <= target < len(dataset.actor_ids):
        raise ValueError("编辑目标越出当前真实实例表")
    evaluate(model, dataset, args.output, target=target)


if __name__ == "__main__":
    main()

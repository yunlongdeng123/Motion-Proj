"""CPU 汇总 GMC 六组真实编辑 FP32；工程合同与300步外观诊断分开。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image


OPERATIONS = tuple(f"actor{actor}_{operation}" for actor in (0, 1) for operation in ("delete", "translate", "rotate"))
RGB_KEYS = ("gt", "before", "after", "static", "after_static", "before_vehicles", "after_vehicles")
VISIBLE_KEYS = ("before_target_visible", "after_target_visible")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def raw_frame_id(value):
    return Path(str(value).replace("\\", "/")).stem


def same_bits(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def masked_mean(value, mask):
    return float(value[mask].mean(dtype=np.float64)) if bool(mask.any()) else None


def actor_bbox(mask, padding=40):
    rows, columns = np.nonzero(mask)
    if not len(rows):
        return None
    height, width = mask.shape
    return [max(0, int(columns.min())-padding), max(0, int(rows.min())-padding),
            min(width, int(columns.max())+padding+1), min(height, int(rows.max())+padding+1)]


def load_fp32(path, height=None, width=None):
    with np.load(path, allow_pickle=False) as saved:
        require(all(key in saved for key in RGB_KEYS + VISIBLE_KEYS), f"缺少实际FP32字段: {path}")
        arrays = {key: saved[key] for key in RGB_KEYS + VISIBLE_KEYS}
    shape = arrays["gt"].shape
    require(len(shape) == 3 and shape[0] == 3, f"GT必须为3xHxW: {path}")
    h, w = shape[1:]
    if height is not None:
        require((h, w) == (height, width), f"实际FP32分辨率不匹配: {path}")
    for key, value in arrays.items():
        expected = (3 if key in RGB_KEYS else 1, h, w)
        require(value.dtype == np.float32 and value.shape == expected and np.isfinite(value).all(), f"FP32 shape/finite失败: {path}/{key}")
    for key in VISIBLE_KEYS:
        require(float(arrays[key].min()) >= -1e-6 and float(arrays[key].max()) <= 1+1e-6, f"可见贡献范围失败: {path}/{key}")
    return arrays


def _mean(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return float(np.mean(values)) if values else None


def collect(edit_root, prepared, *, reference_manifest=None, expected_steps=300, roi_padding=40):
    edit_root, prepared = Path(edit_root).resolve(), Path(prepared).resolve()
    require(expected_steps > 0 and roi_padding >= 0, "steps/padding无效")
    with np.load(prepared / "instances.npz", allow_pickle=False) as saved:
        frame_ids, tracks = saved["frame_ids"].tolist(), saved["instance_ids"].tolist()
        times, known, valid = saved["timestamps"], saved["mask_known"], saved["valid"]
    require(len(frame_ids) == 49 and len(set(frame_ids)) == 49 and len(tracks) == 2, "此次需真实GMC 49帧双实例bundle")
    require(known.shape == valid.shape == (49, 2) and known.dtype == valid.dtype == np.bool_ and bool(known.all()), "实际49x2 mask_known必须全部明确")
    reference = None
    if reference_manifest:
        manifest_path = Path(reference_manifest).resolve()
        manifest = read_json(manifest_path)
        reference = {raw_frame_id(frame["id"]): frame for frame in manifest["frames"]}
        require(set(reference) == set(frame_ids), "原GMC相机GT manifest身份不匹配")
    metadata, checkpoint, source_spec = {}, None, None
    for name in OPERATIONS:
        directory = edit_root / name
        receipt = read_json(directory / "edit_receipt.json")
        require(receipt["status"] == "EDIT_RENDER_EXPORTED_VISUAL_REVIEW_PENDING", f"编辑未完整导出: {name}")
        actor, operation = int(name[5]), name.split("_", 1)[1]
        arguments = receipt["arguments"]
        require(arguments["operation"] == operation and arguments["track"] == tracks[actor], f"真实target/operation身份不匹配: {name}")
        current_checkpoint = arguments["checkpoint"]
        match = re.search(r"checkpoint_(\d+)\.pt$", current_checkpoint)
        require(match and int(match.group(1)) == expected_steps, f"checkpoint短测步数不匹配: {name}")
        if checkpoint is None:
            checkpoint, source_spec = current_checkpoint, receipt["spec"]
        require(current_checkpoint == checkpoint and receipt["spec"] == source_spec, f"编辑源checkpoint/scene绑定不同: {name}")
        require(receipt.get("no_model_parameters_mutated_by_edits") is True, f"缺少编辑不修改模型参数的执行receipt: {name}")
        frames = receipt["frames"]
        require(len(frames) == 49 and [row["frame"] for row in frames] == list(range(49)), f"缺失/重复观察帧: {name}")
        require([raw_frame_id(row["frame_id"]) for row in frames] == frame_ids, f"观察帧顺序/身份不同: {name}")
        require(np.allclose([row["timestamp"] for row in frames], times, atol=1e-7, rtol=0), f"真实模型时间不同: {name}")
        require(len(list(directory.glob("[0-9][0-9][0-9]_*.npz"))) == 49, f"实际FP32文件数不是49: {name}")
        metadata[name] = {"directory": str(directory), "actor_index": actor, "target_track": tracks[actor],
                          "operation": operation, "arguments": arguments, "receipt": receipt, "frames": []}
    height = width = None
    gt_png_max_delta = 0
    for frame, frame_id in enumerate(frame_ids):
        canonical, canonical_target = None, {}
        labels = None
        for name in OPERATIONS:
            item, record = metadata[name], metadata[name]["receipt"]["frames"][frame]
            npz_path = Path(item["directory"]) / f"{frame:03d}_{frame_id}.npz"
            arrays = load_fp32(npz_path, height, width)
            if height is None:
                height, width = arrays["gt"].shape[1:]
                require((height, width) == (320, 480), "此次GMC实际camera导出必须为480x320")
            if canonical is None:
                canonical = {key: arrays[key] for key in ("gt", "before", "static", "before_vehicles")}
                mask_path = prepared / "instance_masks" / (frame_id + ".png")
                with Image.open(mask_path) as image:
                    labels = np.array(image.resize((width, height), Image.Resampling.NEAREST))
                require(labels.shape == (height, width) and np.isin(labels, [0, 1, 2]).all(), f"真实身份mask类别/分辨率失败: {frame_id}")
                if reference is not None:
                    original = reference[frame_id]
                    gt_path = manifest_path.parent / original["images"]["gt"]
                    with Image.open(gt_path) as image:
                        original_png = np.array(image.convert("RGB"))
                    require(original_png.shape == (height, width, 3), f"原GMC GT分辨率不匹配: {frame_id}")
                    exported_png = np.rint(arrays["gt"].clip(0, 1).transpose(1, 2, 0)*255).astype(np.uint8)
                    delta = int(np.abs(exported_png.astype(np.int16)-original_png.astype(np.int16)).max())
                    require(delta <= 1, f"实际GT与原相机GT不匹配: {frame_id}; 最大byte差={delta}")
                    gt_png_max_delta = max(gt_png_max_delta, delta)
            for key, value in canonical.items():
                require(same_bits(arrays[key], value), f"六组同帧before/GT/static/vehicles配对非逐位相同: {name}/{frame_id}/{key}")
            require(same_bits(arrays["static"], arrays["after_static"]), f"编辑改变独立静态FP32: {name}/{frame_id}")
            require(record.get("static_render_bitwise_unchanged") is True and record.get("other_instance_Gaussians_bitwise_unchanged") is True, f"缺少静态/其他实例不变执行receipt: {name}/{frame_id}")
            actor = item["actor_index"]
            require(record.get("target_track_valid") == bool(valid[frame, actor]), f"target轨迹有效性与真实bundle不符: {name}/{frame_id}")
            before_visible, after_visible = arrays["before_target_visible"][0], arrays["after_target_visible"][0]
            if actor in canonical_target:
                require(same_bits(before_visible, canonical_target[actor]), f"同target三组before contribution不一致: {name}/{frame_id}")
            else:
                canonical_target[actor] = before_visible
            mask = labels == actor+1
            if item["operation"] == "delete":
                require(not bool(np.count_nonzero(after_visible)), f"删除后仍有目标贡献: {name}/{frame_id}")
            diff = np.abs(arrays["after"].astype(np.float64)-arrays["before"].astype(np.float64)).mean(0)
            mae = float(diff.mean())
            require(math.isclose(mae, record["pixel_RGB_MAE"], abs_tol=1e-6, rel_tol=1e-5), f"RGB_MAE receipt复算不符: {name}/{frame_id}")
            sq = (before_visible.astype(np.float64)-mask.astype(np.float64))**2
            row = {
                "frame": frame, "frame_id": frame_id, "timestamp": float(times[frame]),
                "source_npz": str(npz_path), "asset_relative_path": f"{name}/{npz_path.name}",
                "mask_path": str(prepared / "instance_masks" / (frame_id+".png")),
                "target_mask_known": bool(known[frame, actor]), "target_pose_valid": bool(valid[frame, actor]),
                "target_body_pixels": int(mask.sum()), "native_bbox_padded": actor_bbox(mask, roi_padding),
                "before_visible_body_mean": masked_mean(before_visible, mask),
                "before_visible_outside_mean": masked_mean(before_visible, ~mask),
                "before_mask_mse_full_HxW": float(sq.mean()),
                "before_mask_mse_inside": masked_mean(sq, mask), "before_mask_mse_outside": masked_mean(sq, ~mask),
                "before_visible_sum": float(before_visible.sum(dtype=np.float64)),
                "after_visible_sum": float(after_visible.sum(dtype=np.float64)),
                "after_visible_max": float(after_visible.max()),
                "edit_full_RGB_MAE": mae, "edit_changed_pixels_gt_1e_minus6": int((diff > 1e-6).sum()),
                "edit_RGB_MAE_inside_original_body": masked_mean(diff, mask),
                "edit_RGB_MAE_outside_original_body": masked_mean(diff, ~mask),
                "before_RGB_MSE_vs_observed_GT_clipped": float(((arrays["before"].clip(0, 1).astype(np.float64)-arrays["gt"])**2).mean()),
                "static_FP32_bitwise_unchanged": True, "other_instance_world_tensors_receipt_unchanged": True,
                "edited_GT_available": False,
            }
            item["frames"].append(row)
    for name, item in metadata.items():
        rows = item["frames"]
        item["aggregate"] = {key: _mean(rows, key) for key in ("before_mask_mse_full_HxW", "before_visible_body_mean", "before_visible_outside_mean", "edit_full_RGB_MAE", "before_RGB_MSE_vs_observed_GT_clipped")}
        item["aggregate"].update(empty_body_frames=sum(row["target_body_pixels"] == 0 for row in rows),
                                  active_positive_mask_frames=sum(row["target_body_pixels"] > 0 for row in rows),
                                  invalid_pose_positive_mask_frames=[row["frame_id"] for row in rows if row["target_body_pixels"] > 0 and not row["target_pose_valid"]])
        item["bad_cases"] = {
            "largest_before_mask_MSE": [row["frame"] for row in sorted(rows, key=lambda row: row["before_mask_mse_full_HxW"], reverse=True)[:5]],
            "largest_before_outside_contribution": [row["frame"] for row in sorted(rows, key=lambda row: row["before_visible_outside_mean"] or 0, reverse=True)[:5]],
            "largest_edit_RGB_change": [row["frame"] for row in sorted(rows, key=lambda row: row["edit_full_RGB_MAE"], reverse=True)[:5]],
            "nonempty_mask_no_edit_pixel_change": [row["frame"] for row in rows if row["target_body_pixels"] > 0 and row["edit_changed_pixels_gt_1e_minus6"] == 0],
            "empty_body_frames": [row["frame"] for row in rows if row["target_body_pixels"] == 0],
        }
        del item["receipt"]
    return {
        "schema": "drivingeditor_GMC_instance_edit_smoke_CPU_summary_v1", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"edit_root": str(edit_root), "prepared": str(prepared), "reference_manifest": str(Path(reference_manifest).resolve()) if reference_manifest else None,
                   "scene_spec": source_spec, "checkpoint": checkpoint, "checkpoint_steps": expected_steps},
        "engineering_contract": {"operation_count": 6, "frames_per_operation": 49, "verified_FP32_pairs": 294,
                                 "native_width": width, "native_height": height, "all_FP32_finite": True,
                                 "GT_before_static_vehicles_bitwise_paired_across_operations": True,
                                 "static_before_after_bitwise_verified_from_saved_FP32": True,
                                 "other_instance_world_tensors_invariance": "GPU execution receipt; independent CPU model restore not performed",
                                 "GT_reference_PNG_verified": reference is not None, "GT_reference_max_byte_delta": gt_png_max_delta if reference else None,
                                 "all_delete_target_contributions_exact_zero": True, "passed": True},
        "scope": {"engineering_smoke_only": True, "converged_quality_passed": False, "edited_GT_available": False,
                  "restoration_verified": False, "human_quality_verdict": None, "relighting_verified": False,
                  "no_new_training_or_rerender": True, "GPU_used": False, "Torch_imported": "torch" in sys.modules,
                  "before_mask_metrics": "actual unedited joint visible contribution; equal 49 frame weights, full HxW inside+outside",
                  "edit_RGB_change": "effect diagnostic; changed other-visible pixels may follow correct occlusion and are not automatically bad",
                  "empty_body_denominator": "inside-domain means null; zero pixels retained",
                  "roi_padding_pixels": roi_padding,
                  "native_ROI": f"actual reviewed target mask bbox padded {roi_padding}px; same before/after crop; empty mask full view"},
        "instance_ids": tracks, "operations": metadata,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edit-root", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--expected-steps", type=int, default=300)
    parser.add_argument("--roi-padding", type=int, default=40)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"保留旧summary，不覆盖: {args.output}")
    result = collect(args.edit_root, args.prepared, reference_manifest=args.reference_manifest,
                     expected_steps=args.expected_steps, roi_padding=args.roi_padding)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["engineering_contract"], ensure_ascii=False))


if __name__ == "__main__":
    main()

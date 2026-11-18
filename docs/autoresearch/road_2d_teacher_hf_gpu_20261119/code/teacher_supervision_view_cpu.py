"""显式视图窗口的二维教师候选；旧全帧接口和三维入口均保持原值。

native_from_model 映射像素单元边界；相应像素中心是 index+.5。
source_model_valid 必须显式提供。crop外零值只作无效占位，从不获得监督。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from teacher_supervision_cpu import (NATIVE_KEYS, area_rgb, footprint_mask,
                                    save_package, scale_intrinsics, validate_packet)

SCHEMA = "road_2d_view_teacher_supervision.v1"
FIXED_VIEWS = {
    "full768": ((768, 512), (0, 0, 1920, 1280)),
    "full1152": ((1152, 768), (0, 0, 1920, 1280)),
    "crop768": ((768, 272), (0, 600, 1920, 1280)),
}


def _wh(value):
    if len(value) != 2 or any(int(x) != x or int(x) <= 0 for x in value):
        raise ValueError("尺寸必须是正整数 [width,height]")
    return tuple(map(int, value))


def _rgb(value, name):
    value = np.asarray(value)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(name + " 必须是 uint8 HWC RGB")
    return value


def axis_spans(source_count, source_extent, target_count, target_extent):
    """同一原生轴上严格正面积支持；所有边界计算使用整数有理区间。"""
    sa, sb = source_extent
    ta, tb = target_extent
    source_span, target_span = sb - sa, tb - ta
    if min(source_count, target_count, source_span, target_span) <= 0:
        raise ValueError("非法轴尺寸/窗口")
    source_origin = sa * source_count * target_count
    source_step = source_span * target_count
    for j in range(target_count):
        left = ta * source_count * target_count + j * target_span * source_count
        right = left + target_span * source_count
        lo = max(0, min(source_count, (left - source_origin) // source_step))
        hi = max(0, min(source_count, -((source_origin - right) // source_step)))
        full = left >= source_origin and right <= sb * source_count * target_count
        if hi <= lo:
            yield 0, 0, np.empty(0, np.float64), False
            continue
        ids = np.arange(lo, hi, dtype=np.int64)
        overlap = (np.minimum(source_origin + (ids + 1) * source_step, right)
                   - np.maximum(source_origin + ids * source_step, left))
        if np.any(overlap <= 0):
            raise AssertionError("非零支持含零/负交集")
        yield lo, hi, overlap.astype(np.float64) / (target_span * source_count), bool(full)


def _resample_axis(value, target_count, axis, source_extent, target_extent, *, mask_mode=None):
    value = np.moveaxis(value, axis, 0)
    shape = (target_count, *value.shape[1:])
    output = np.zeros(shape, bool if mask_mode else np.float32)
    for j, (lo, hi, weight, full) in enumerate(axis_spans(len(value), source_extent, target_count, target_extent)):
        if hi <= lo:
            continue
        if mask_mode == "all":
            output[j] = full & np.all(value[lo:hi], axis=0)
        elif mask_mode == "any":
            output[j] = np.any(value[lo:hi], axis=0)
        elif mask_mode is None:
            output[j] = np.tensordot(weight, value[lo:hi], axes=(0, 0))
        else:
            raise ValueError("未知mask归约")
    return np.moveaxis(output, 0, axis)


def mapped_resample(value, output_wh, source_window, target_window, *, mask_mode=None):
    """对两个显式轴对齐原生窗口执行一次可分离面积积分/全支持归约。"""
    width, height = _wh(output_wh)
    sx0, sy0, sx1, sy1 = source_window
    tx0, ty0, tx1, ty1 = target_window
    value = np.asarray(value)
    if mask_mode and (value.dtype != bool or value.ndim != 2):
        raise ValueError("窗口mask必须是bool HW")
    temp = _resample_axis(value, width, 1, (sx0, sx1), (tx0, tx1), mask_mode=mask_mode)
    return _resample_axis(temp, height, 0, (sy0, sy1), (ty0, ty1), mask_mode=mask_mode)


def validate_view(model_wh, camera_wh, native_window_xyxy, native_from_model, *, view_name=None):
    model_wh, camera_wh = _wh(model_wh), _wh(camera_wh)
    if len(native_window_xyxy) != 4 or any(int(v) != v for v in native_window_xyxy):
        raise ValueError("native window必须是整数边界 [x0,y0,x1,y1]")
    window = tuple(map(int, native_window_xyxy))
    x0, y0, x1, y1 = window
    if not (0 <= x0 < x1 <= camera_wh[0] and 0 <= y0 < y1 <= camera_wh[1]):
        raise ValueError("native window超出完整相机")
    expected = np.array([[(x1 - x0) / model_wh[0], 0, x0],
                         [0, (y1 - y0) / model_wh[1], y0], [0, 0, 1]], np.float64)
    matrix = np.asarray(native_from_model, np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.allclose(matrix, expected, rtol=0, atol=1e-12):
        raise ValueError("native_from_model与实际模型尺寸/window不一致；仅支持轴对齐像素单元映射")
    if view_name is not None:
        if view_name not in FIXED_VIEWS or camera_wh != (1920, 1280):
            raise ValueError("固定view仅适用于1920x1280原生相机")
        if (model_wh, window) != FIXED_VIEWS[view_name]:
            raise ValueError("固定view的crop/模型大小不符: " + view_name)
    return window, matrix


def fixed_view_spec(name):
    if name not in FIXED_VIEWS:
        raise ValueError("未知固定view")
    model_wh, window = FIXED_VIEWS[name]
    x0, y0, x1, y1 = window
    return dict(name=name, model_wh=list(model_wh), native_window_xyxy=list(window),
                native_from_model=[[(x1 - x0) / model_wh[0], 0, x0],
                                   [0, (y1 - y0) / model_wh[1], y0], [0, 0, 1]])


def build_view_supervision(raw_model_rgb, observed_rgb, native_domains, *, source_model_valid,
                           native_window_xyxy, native_from_model, output_wh, intrinsics,
                           camera_wh, camera_convention, source_kind, view_name=None):
    """输出完整相机网格的2D候选；不产生3D行、fixed_alpha或HF许可。"""
    if source_kind != "lossless_raw_model":
        raise ValueError("教师必须来自冻结无损raw_model，不能用回贴图或旧监督放大")
    raw, observed = _rgb(raw_model_rgb, "raw_model"), _rgb(observed_rgb, "observation")
    model_wh, camera_wh, output_wh = (raw.shape[1], raw.shape[0]), _wh(camera_wh), _wh(output_wh)
    if camera_wh != (observed.shape[1], observed.shape[0]):
        raise ValueError("原生实拍与完整相机尺寸不一致")
    if output_wh[0] * camera_wh[1] != output_wh[1] * camera_wh[0]:
        raise ValueError("输出必须是完整相机FOV，不能改变宽高比")
    window, matrix = validate_view(model_wh, camera_wh, native_window_xyxy, native_from_model, view_name=view_name)
    full_window = (0, 0, *camera_wh)
    output_spacing = np.asarray(camera_wh) / np.asarray(output_wh)
    if np.any(output_spacing + 1e-12 < np.array([matrix[0, 0], matrix[1, 1]])):
        raise ValueError("禁止放大教师像素以冒充新增细节")
    source_valid = np.asarray(source_model_valid)
    if source_valid.dtype != bool or source_valid.shape != raw.shape[:2]:
        raise ValueError("必须显式提供与raw_model同网格的bool source_model_valid")
    domains = {}
    for key in NATIVE_KEYS:
        value = np.asarray(native_domains[key])
        if value.dtype != bool or value.shape != observed.shape[:2]:
            raise ValueError("原生域必须是完整相机bool HW: " + key)
        domains[key] = value
    forbidden = domains["K"] | domains["protected"]
    for a, b in (("A", "C"), ("A", "S"), ("C", "S")):
        if np.any(domains[a] & domains[b]):
            raise ValueError("原生域冲突")
    if np.any((domains["A"] | domains["C"] | domains["S"]) & forbidden):
        raise ValueError("原生监督含保护域")
    if np.any((domains["C"] | domains["S"]) & ~domains["teacher_write"]):
        raise ValueError("C/S超出原教师写许可")
    rgb_valid = mapped_resample(source_valid, output_wh, window, full_window, mask_mode="all")
    teacher = mapped_resample(raw.astype(np.float32) / 255, output_wh, window, full_window)
    teacher[~rgb_valid] = 0  # 无效占位；不回贴实拍，不提供可监督值。
    output = dict(target_A_srgb=area_rgb(observed, output_wh).transpose(2, 0, 1),
                  target_C_srgb=teacher.transpose(2, 0, 1), target_S_srgb=teacher.transpose(2, 0, 1).copy(),
                  source_RGB_valid=rgb_valid, source_model_valid=source_valid.copy(),
                  K=footprint_mask(domains["K"], output_wh, "any"),
                  protected=footprint_mask(domains["protected"], output_wh, "any"),
                  teacher_write_full=footprint_mask(domains["teacher_write"] & ~forbidden, output_wh, "all"))
    write_source = (mapped_resample(domains["teacher_write"] & ~forbidden, model_wh, full_window, window, mask_mode="all")
                    & source_valid)
    output["teacher_grid_write_valid"] = write_source
    output["teacher_write_filter_full"] = mapped_resample(write_source, output_wh, window, full_window, mask_mode="all")
    counts = {}
    for role in ("A", "C", "S"):
        full = footprint_mask(domains[role] & ~forbidden, output_wh, "all")
        if role == "A":
            support = full.copy()
        else:
            source = mapped_resample(domains[role] & domains["teacher_write"] & ~forbidden,
                                     model_wh, full_window, window, mask_mode="all") & source_valid
            output["teacher_grid_" + role + "_valid"] = source
            support = mapped_resample(source, output_wh, window, full_window, mask_mode="all")
        mask = full & support & ~output["K"] & ~output["protected"]
        if role != "A":
            mask &= output["teacher_write_filter_full"] & rgb_valid
        output[role], output["native_footprint_" + role], output["filter_support_" + role] = mask, full, support
        output["q_" + role + "_2d"] = mask.astype(np.float32)
        counts[role] = dict(native_pixels=int(domains[role].sum()), full_target_footprint_pixels=int(full.sum()),
                            filter_support_excluded_pixels=int((full & ~mask).sum()), candidate_2d_pixels=int(mask.sum()))
    output.update(outside_2d=~(output["A"] | output["C"] | output["S"] | output["K"] | output["protected"]),
                  camera_K=scale_intrinsics(intrinsics, camera_wh, output_wh, camera_convention),
                  output_wh=np.asarray(output_wh, np.int64), teacher_wh=np.asarray(model_wh, np.int64),
                  native_wh=np.asarray(camera_wh, np.int64), teacher_native_window_xyxy=np.asarray(window, np.int64),
                  native_from_model=matrix)
    validate_packet(output)
    if np.any((output["C"] | output["S"]) & ~rgb_valid):
        raise AssertionError("无效或crop外区域获得监督")
    report = dict(schema=SCHEMA, candidate_only=True, training_ready=False, GPU_executed=False,
                  new_model_inference=False, new_render_operations=0, training_updates=0,
                  source_kind=source_kind, source_composited=False, view_name=view_name,
                  teacher_wh=list(model_wh), native_wh=list(camera_wh), output_wh=list(output_wh),
                  teacher_native_window_xyxy=list(window), native_from_model=matrix.tolist(),
                  transform_convention="pixel-cell edges; center native=T@[model_index+.5,1]",
                  camera_convention=camera_convention, camera_contract="caller-declared full-FOV K; not raster-qualified",
                  actual_raster_camera_convention_verified=False, camera_intrinsics_output=output["camera_K"].tolist(),
                  source_valid_required=True, source_RGB_valid_pixels=int(rgb_valid.sum()),
                  invalid_RGB_value=0, invalid_RGB_is_placeholder_not_prediction=True,
                  RGB_path=dict(A="original full native observation -> one area integral -> full output",
                                C="frozen raw_model in explicit native window -> one mapped area integral -> full output",
                                S="same frozen raw_model in explicit native window -> one mapped area integral -> full output"),
                  domain_provenance=dict(A="observed original camera RGB", C="generated frozen raw_model pseudo teacher",
                                         S="generated frozen raw_model pseudo teacher"),
                  RGB_space="sRGB appearance, no gamma/exposure fit or sharpening",
                  q_role="binary 2D source permission; not teacher confidence or material H permission",
                  q_not_material_permission=True, preserve_parent_high_frequency_policy=True,
                  no_new_3d_high_frequency_authorization=True,
                  mask_rule="full native domain footprint AND all nonzero model filter support in same authorized role and explicit source valid; any K/protected excludes",
                  counts=counts, parent_3d_and_default_training_modified=False,
                  required_before_training=["frozen output-resolution geometry and retained guards", "fixed alpha and actual renderer contribution",
                                            "independent raster camera qualification", "native joint scope and lifecycle validation"])
    return output, report


def _read_rgb(path):
    with Image.open(path) as im:
        if im.mode != "RGB" or im.format not in ("PNG", "TIFF", "BMP"):
            raise ValueError("源RGB必须是无损uint8 RGB")
        return np.asarray(im).copy()


def save_view_package(output, arrays, report, config):
    """保留原PNG字节与显式mapping；不会将无效占位画布当源图保存。"""
    output = Path(output)
    report = dict(report, source_artifacts=dict(raw_model="source/raw_model.png",
                  source_model_valid="source/model_valid.png", mapping="source/view_mapping.json"))
    save_package(output, arrays, report, sources=config)
    folder = output / "source"
    folder.mkdir()
    shutil.copy2(config["raw_model"], folder / "raw_model.png")
    shutil.copy2(config["source_model_valid"], folder / "model_valid.png")
    mapping = dict(view=config["view"], source_kind=config["source_kind"], original_raw_model=config["raw_model"],
                   original_source_model_valid=config["source_model_valid"], camera=config["camera"],
                   output_wh=config["output_wh"], invalid_target_zero_is_placeholder=True,
                   actual_raster_camera_convention_verified=False)
    (folder / "view_mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    with np.load(config["native_domains"], allow_pickle=False) as saved:
        domains = {key: saved[key].copy() for key in NATIVE_KEYS}
    with Image.open(config["source_model_valid"]) as im:
        mask = np.asarray(im)
        if mask.ndim != 2 or not np.isin(mask, [0, 255]).all():
            raise ValueError("source_model_valid必须是同模型网格二值PNG")
        source_valid = mask == 255
    camera, view = config["camera"], config["view"]
    arrays, report = build_view_supervision(_read_rgb(config["raw_model"]), _read_rgb(config["observation"]), domains,
        source_model_valid=source_valid, native_window_xyxy=view["native_window_xyxy"], native_from_model=view["native_from_model"],
        output_wh=config["output_wh"], intrinsics=camera["intrinsics"], camera_wh=camera["camera_wh"],
        camera_convention=camera["convention"], source_kind=config["source_kind"], view_name=view["name"])
    save_view_package(args.output, arrays, report, config)
    print(json.dumps(dict(output=str(args.output), **report), ensure_ascii=False))


if __name__ == "__main__":
    main()

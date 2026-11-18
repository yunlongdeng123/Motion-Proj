"""独立二维教师监督候选；不导入 Torch、三维场景或原生渲染器。

RGB 只从无损 raw_model 或原始实拍直接重采样一次。布尔许可同时覆盖
目标像素足迹与面积滤波的全部源像素支持；不把 H_input 当作写许可。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image


SCHEMA = "road_2d_teacher_supervision.v1"
CONVENTIONS = ("half_pixel", "integer_center")
NATIVE_KEYS = ("A", "C", "S", "teacher_write", "K", "protected")


def _wh(value: Any) -> tuple[int, int]:
    if len(value) != 2 or any(int(x) != x or int(x) <= 0 for x in value):
        raise ValueError("图像尺寸必须是正整数 [width,height]")
    return tuple(map(int, value))


def _rgb(value: Any, name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(name + " 必须是 uint8 HWC RGB")
    return value


def _spans(source: int, target: int):
    """用整数区间确定严格正面积支持，避免浮点边界扩大掩膜。"""
    for j in range(target):
        left, right = j * source, (j + 1) * source
        lo, hi = left // target, (right + target - 1) // target
        ids = np.arange(lo, hi, dtype=np.int64)
        overlap = np.minimum((ids + 1) * target, right) - np.maximum(ids * target, left)
        if np.any(overlap <= 0):
            raise AssertionError("滤波支持包含零面积源像素")
        yield lo, hi, overlap.astype(np.float64) / source


def _area_axis(value: np.ndarray, target: int, axis: int) -> np.ndarray:
    value = np.moveaxis(value, axis, 0)
    source = len(value)
    if source == target:
        return np.moveaxis(value.copy(), 0, axis)
    output = np.empty((target, *value.shape[1:]), np.float32)
    for j, (lo, hi, weight) in enumerate(_spans(source, target)):
        output[j] = np.tensordot(weight, value[lo:hi], axes=(0, 0))
    return np.moveaxis(output, 0, axis)


def area_rgb(value: np.ndarray, output_wh: tuple[int, int]) -> np.ndarray:
    """一次可分离的精确像素面积积分，返回 HWC float32 sRGB [0,1]。"""
    value = _rgb(value, "RGB")
    width, height = _wh(output_wh)
    value = value.astype(np.float32) / 255.0
    return _area_axis(_area_axis(value, width, 1), height, 0)


def footprint_mask(value: np.ndarray, output_wh: tuple[int, int], mode: str) -> np.ndarray:
    """all: 整个目标足迹获得许可；any: 足迹接触保护域即被保护。"""
    value = np.asarray(value)
    if value.dtype != bool or value.ndim != 2 or mode not in ("all", "any"):
        raise ValueError("布尔 HW 掩膜及 all/any 归约是必需的")
    width, height = _wh(output_wh)
    reduce = np.all if mode == "all" else np.any
    temp = np.empty((value.shape[0], width), bool)
    for j, (lo, hi, _) in enumerate(_spans(value.shape[1], width)):
        temp[:, j] = reduce(value[:, lo:hi], axis=1)
    output = np.empty((height, width), bool)
    for j, (lo, hi, _) in enumerate(_spans(value.shape[0], height)):
        output[j] = reduce(temp[lo:hi], axis=0)
    return output


def scale_intrinsics(intrinsics, source_wh, output_wh, convention: str) -> np.ndarray:
    """显式区分 K 用半整数像素中心或整数像素中心的调用方约定。"""
    if convention not in CONVENTIONS:
        raise ValueError("必须显式指定 half_pixel 或 integer_center，不猜测 raster 约定")
    source_wh, output_wh = _wh(source_wh), _wh(output_wh)
    k = np.asarray(intrinsics, np.float64)
    if (k.shape != (3, 3) or not np.isfinite(k).all() or not np.allclose(k[2], [0, 0, 1])
            or k[0, 0] <= 0 or k[1, 1] <= 0):
        raise ValueError("intrinsics 必须是有限 pinhole 3x3 K")
    sx, sy = np.asarray(output_wh) / np.asarray(source_wh)
    transform = np.diag([sx, sy, 1.0])
    if convention == "integer_center":
        transform[0, 2], transform[1, 2] = (sx - 1) / 2, (sy - 1) / 2
    return transform @ k


def map_pixel_indices(xy, source_wh, output_wh) -> np.ndarray:
    """相同视场的源索引到输出索引：s*(index+.5)-.5。"""
    xy = np.asarray(xy, np.float64)
    if xy.shape[-1] != 2:
        raise ValueError("像素索引应以 x,y 结束")
    scale = np.asarray(_wh(output_wh)) / np.asarray(_wh(source_wh))
    return (xy + 0.5) * scale - 0.5


def build_supervision(raw_model_rgb, observed_rgb, native_domains: Mapping[str, np.ndarray],
                      *, output_wh, intrinsics, camera_wh, camera_convention: str,
                      source_kind: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """构建 2D 候选；不提供 fixed_alpha、几何许可或 GPU 训练就绪证明。"""
    if source_kind != "lossless_raw_model":
        raise ValueError("教师必须显式为 lossless_raw_model；不接受回贴图或旧480监督")
    raw, observed = _rgb(raw_model_rgb, "raw_model"), _rgb(observed_rgb, "observation")
    native_hw = observed.shape[:2]
    camera_wh, output_wh = _wh(camera_wh), _wh(output_wh)
    raw_wh = raw.shape[1], raw.shape[0]
    if camera_wh != (native_hw[1], native_hw[0]):
        raise ValueError("实拍/native mask 与原始相机尺寸不一致")
    if raw_wh[0] * native_hw[0] != raw_wh[1] * native_hw[1]:
        raise ValueError("raw_model 必须覆盖完整相机视场，禁止隐式 crop/pad")
    if output_wh[0] * native_hw[0] != output_wh[1] * native_hw[1]:
        raise ValueError("监督输出必须保持完整视场宽高比")
    if output_wh[0] > raw_wh[0] or output_wh[1] > raw_wh[1]:
        raise ValueError("不允许把低分辨率教师放大声称为细节监督")
    domains = {}
    for key in NATIVE_KEYS:
        if key not in native_domains:
            raise ValueError("缺少原生域 " + key)
        value = np.asarray(native_domains[key])
        if value.dtype != bool or value.shape != native_hw:
            raise ValueError(key + " 必须是与原始相机一致的 bool HW")
        domains[key] = value
    for a, b in (("A", "C"), ("A", "S"), ("C", "S")):
        if np.any(domains[a] & domains[b]):
            raise ValueError("原生监督域冲突: " + a + "/" + b)
    forbidden = domains["K"] | domains["protected"]
    if np.any((domains["C"] | domains["S"]) & ~domains["teacher_write"]):
        raise ValueError("C/S 超出显式教师写许可；不能使用 H_input 扩张许可")
    if np.any((domains["A"] | domains["C"] | domains["S"]) & forbidden):
        raise ValueError("监督原生域包含 K/protected")

    target_a = area_rgb(observed, output_wh).transpose(2, 0, 1)
    target_teacher = area_rgb(raw, output_wh).transpose(2, 0, 1)
    output = dict(target_A_srgb=target_a, target_C_srgb=target_teacher,
                  target_S_srgb=target_teacher.copy(),
                  K=footprint_mask(domains["K"], output_wh, "any"),
                  protected=footprint_mask(domains["protected"], output_wh, "any"),
                  teacher_write_full=footprint_mask(domains["teacher_write"] & ~forbidden, output_wh, "all"))
    source_write = footprint_mask(domains["teacher_write"] & ~forbidden, raw_wh, "all")
    output["teacher_grid_write_valid"] = source_write
    output["teacher_write_filter_full"] = footprint_mask(source_write, output_wh, "all")
    counts = {}
    for role in ("A", "C", "S"):
        full = footprint_mask(domains[role] & ~forbidden, output_wh, "all")
        if role == "A":
            support = full.copy()  # 原生实拍直接积分到输出，无中间模型网格。
        else:
            source_role = footprint_mask(domains[role] & domains["teacher_write"] & ~forbidden, raw_wh, "all")
            output["teacher_grid_" + role + "_valid"] = source_role
            support = footprint_mask(source_role, output_wh, "all")
        mask = full & support & ~output["K"] & ~output["protected"]
        if role != "A":
            mask &= output["teacher_write_filter_full"]
        output[role] = mask
        output["native_footprint_" + role] = full
        output["filter_support_" + role] = support
        output["q_" + role + "_2d"] = mask.astype(np.float32)
        counts[role] = dict(native_pixels=int(domains[role].sum()),
                            full_target_footprint_pixels=int(full.sum()),
                            filter_support_excluded_pixels=int((full & ~mask).sum()),
                            candidate_2d_pixels=int(mask.sum()))
    output["outside_2d"] = ~(output["A"] | output["C"] | output["S"] | output["K"] | output["protected"])
    output["camera_K"] = scale_intrinsics(intrinsics, camera_wh, output_wh, camera_convention)
    output["output_wh"] = np.asarray(output_wh, np.int64)
    output["teacher_wh"] = np.asarray(raw_wh, np.int64)
    output["native_wh"] = np.asarray(camera_wh, np.int64)
    report = dict(schema=SCHEMA, candidate_only=True, training_ready=False,
                  GPU_executed=False, new_model_inference=False, new_render_operations=0,
                  training_updates=0, output_wh=list(output_wh), teacher_wh=list(raw_wh),
                  native_wh=list(camera_wh), camera_convention=camera_convention,
                  camera_intrinsics_output=output["camera_K"].tolist(),
                  source_kind=source_kind, source_composited=False,
                  RGB_path=dict(A="observation_native -> exact area -> output",
                                C="lossless raw_model -> exact area -> output",
                                S="same lossless raw_model -> exact area -> output"),
                  mask_rule="full native target footprint AND full nonzero area-filter source support; same-role support; any K/protected excludes",
                  RGB_space="sRGB appearance, no gamma transform, clipping, sharpening or exposure fit",
                  q_role="binary 2D source permission, not confidence of teacher correctness",
                  domain_provenance=dict(A="observed original camera RGB", C="generated frozen raw_model pseudo teacher",
                                         S="generated frozen raw_model pseudo teacher"),
                  q_not_material_permission=True,
                  preserve_parent_high_frequency_policy=True,
                  no_new_3d_high_frequency_authorization=True,
                  downstream_contract="q_*_2d only selects binary image-supervision pixels; it is neither material H permission nor confidence; preserve_parent_high_frequency_policy; no new 3D HF authorization",
                  photo_contract="future P_sRGB vs stopgrad(fixed_alpha)*target on qualified masks; no RGB/alpha division",
                  counts=counts,
                  required_before_training=["independent output-camera convention and projection qualification",
                                            "frozen geometry/retained guards at this resolution",
                                            "fixed alpha and actual renderer-contribution qualification",
                                            "native joint protected-domain and full lifecycle smoke"],
                  budget_change_authorized=False, default_training_path_modified=False)
    validate_packet(output)
    return output, report


def validate_packet(packet: Mapping[str, np.ndarray]) -> None:
    """候选可装载验证；不能替代实际渲染器、alpha或GPU资格验证。"""
    width, height = _wh(packet["output_wh"])
    shape = (height, width)
    blocked = np.asarray(packet["K"]) | np.asarray(packet["protected"])
    for role in ("A", "C", "S"):
        mask = np.asarray(packet[role])
        target = np.asarray(packet["target_" + role + "_srgb"])
        if mask.dtype != bool or mask.shape != shape or target.shape != (3, *shape):
            raise ValueError("mask/target shape不一致: " + role)
        if not np.isfinite(target).all() or target.min() < 0 or target.max() > 1:
            raise ValueError("非有限或超出[0,1]的教师值")
        if np.any(mask & blocked) or np.any(mask & ~packet["native_footprint_" + role]) or np.any(mask & ~packet["filter_support_" + role]):
            raise ValueError("域或滤波支持泄漏: " + role)
        if role != "A" and np.any(mask & ~packet["teacher_write_filter_full"]):
            raise ValueError("教师写许可泄漏")
    if np.any(packet["A"] & packet["C"]) or np.any(packet["A"] & packet["S"]) or np.any(packet["C"] & packet["S"]):
        raise ValueError("输出监督域重叠")


def premult_l1_numpy(prediction, fixed_alpha, target, mask) -> float:
    """CPU参考公式；固定alpha不改变合法域，空域必须由调用方明确处理。"""
    p, alpha, t, mask = map(np.asarray, (prediction, fixed_alpha, target, mask))
    if alpha.shape == (1, *mask.shape):
        alpha = alpha[0]
    if mask.dtype != bool or p.shape != (3, *mask.shape) or p.shape != t.shape or alpha.shape != mask.shape:
        raise ValueError("P/alpha/T/mask形状不一致")
    if not mask.any():
        raise ValueError("监督域为空，禁止静默吞掉分母变化")
    if not all(np.isfinite(x).all() for x in (p, alpha, t)) or alpha.min() < 0 or alpha.max() > 1:
        raise ValueError("非有限值或非法alpha")
    return float(np.abs(p - alpha[None] * t)[:, mask].sum(dtype=np.float64) / (3 * mask.sum()))


def save_package(output: Path, arrays, report, *, sources=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    validate_packet(arrays)
    np.savez_compressed(output / "supervision.npz", **arrays)
    report = dict(report, sources=sources or {})
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    with np.load(config["native_domains"], allow_pickle=False) as saved:
        domains = {key: saved[key].copy() for key in NATIVE_KEYS}
    camera = config["camera"]
    with Image.open(config["raw_model"]) as im:
        if im.mode != "RGB" or im.format not in ("PNG", "TIFF", "BMP"):
            raise ValueError("raw_model 文件必须是无损 uint8 RGB 图像，不能由CLI隐式转换")
        raw = np.asarray(im).copy()
    with Image.open(config["observation"]) as im:
        if im.mode != "RGB":
            raise ValueError("observation 文件必须是显式 uint8 RGB")
        observed = np.asarray(im).copy()
    arrays, report = build_supervision(raw, observed, domains, output_wh=config["output_wh"],
                                     intrinsics=camera["intrinsics"], camera_wh=camera["camera_wh"],
                                     camera_convention=camera["convention"], source_kind=config["source_kind"])
    save_package(args.output, arrays, report, sources=config)
    print(json.dumps(dict(output=str(args.output), **report), ensure_ascii=False))


if __name__ == "__main__":
    main()

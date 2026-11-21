"""高斯路块迁移的纯CPU图像/指标包装；不读取扩散RGB作为评价目标。"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "2"
from pathlib import Path
import argparse
import json
import shutil
import tarfile
import time
import traceback
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

CAMPAIGN = Path("/root/autodl-tmp/drivingeditor_repro/campaign")
FRAMES = (86, 110, 134, 179, 198)
STAGES = ("parent", "transfer", "appearance")
REGIONS = ("A", "C", "S")
NAMES = {"parent": "迁移起点 · 道路增密②终态", "transfer": "相似路块迁移 · 零步", "appearance": "局部接缝拟合"}


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else default


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load(path):
    with np.load(path, allow_pickle=False) as packet:
        return {key: packet[key] for key in packet.files}


def rgb(value):
    value = np.asarray(value)
    if value.shape == (3, 768, 1152):
        value = value.transpose(1, 2, 0)
    if value.shape != (768, 1152, 3):
        raise ValueError("须为真实1152×768 RGB")
    value = value.astype(np.float64) / 255 if value.dtype == np.uint8 else value.astype(np.float64)
    if not np.isfinite(value).all():
        raise ValueError("渲染RGB包含非有限值")
    return value


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8)).save(path)
    return path


def bounds(mask):
    y, x = np.nonzero(mask)
    return [max(0, int(x.min()) - 12), max(0, int(y.min()) - 12), min(1152, int(x.max()) + 13),
            min(768, int(y.max()) + 13)] if len(x) else [0, 0, 128, 128]


def effective(packet):
    alpha = packet["road_alpha"]
    return np.divide(rgb(packet["road_rgb"]), alpha[..., None], out=np.zeros((768, 1152, 3)), where=alpha[..., None] > 1e-8)


def average_error(prediction, reference, mask):
    return float(np.abs(prediction[mask] - reference[mask]).mean()) if mask.any() else None


def seam_jump(prediction, target, observed, stable):
    values = []
    for axis in (0, 1):
        left = [slice(None), slice(None)]; right = [slice(None), slice(None)]
        left[axis] = slice(None, -1); right[axis] = slice(1, None)
        left, right = tuple(left), tuple(right)
        crossing = ((target[left] & observed[right]) | (observed[left] & target[right])) & stable[left] & stable[right]
        values.append(np.abs(prediction[left] - prediction[right])[crossing])
    errors = np.concatenate(values, axis=0)
    return {"seam_pairs_n": len(errors), "seam_RGB_jump": float(errors.mean()) if errors.size else None,
            "seam_RGB_jump_p95": float(np.quantile(errors.mean(1), .95)) if errors.size else None}


class PacketFiles:
    def __init__(self, run, output):
        self.run, self.output = run.resolve(), output.resolve()

    def copy(self, source):
        source = Path(source)
        source = source if source.is_absolute() else self.run / source
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.is_relative_to(self.output):
            return source.relative_to(self.output).as_posix()
        relative = source.relative_to(self.run) if source.is_relative_to(self.run) else Path("evidence/external") / source.parent.name / source.name
        target = self.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target != source:
            shutil.copy2(source, target)
        return target.relative_to(self.output).as_posix()

    def images(self, value):
        if isinstance(value, dict):
            return {key: self.images(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.images(item) for item in value]
        if isinstance(value, str) and Path(value).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            return self.copy(value)
        return value


def mask_file(path):
    path = Path(path)
    if path.suffix.lower() == ".npz":
        packet = load(path)
        key = next((key for key in ("target_mask", "requested_mask", "mask") if key in packet), None)
        if key is None:
            raise ValueError("target mask NPZ缺少target_mask/requested_mask/mask")
        result = packet[key].astype(bool)
    else:
        result = np.asarray(Image.open(path))
        if result.ndim == 3:
            if not np.array_equal(result[..., 0], result[..., 1]) or not np.array_equal(result[..., 1], result[..., 2]):
                raise ValueError("target mask须为单通道二值图")
            result = result[..., 0]
        if not np.isin(result, [0, 1, 255]).all():
            raise ValueError("target mask须为二值图")
        result = result > 0
    if result.shape != (768, 1152):
        raise ValueError("target mask尺寸错误")
    return result


def stage_counts(asset, parent_active=None, parent_total=None):
    path = Path(asset) / "road.npz"
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as packet:
        active = packet["active"].astype(bool)
    count = {"total": len(active), "active": int(active.sum()), "activated_existing": 0, "reuse_existing_newgroup": 0}
    if parent_total is not None:
        count["appended_new_geometry"] = max(0, len(active) - parent_total)
    if parent_active is not None:
        count["retired_old_rows"] = int((parent_active & ~active[:len(parent_active)]).sum())
    return count


def refresh_report_metadata(run, output, payload):
    """不重算图像/评价，仅合并最终资源账本和已算出的完整比较结论。"""
    diagnostics = payload["diagnostics"]
    gaps = payload.get("gaps", [])
    boundary = diagnostics.get("boundary_rows", [])
    findings = ["道路层仍可见细黑裂缝、短水平缺口与局部断续边界；联合渲染仍有车形残影。"]
    coverage_worse = False
    for raw in (86, 110):
        rows = {row["id"]: row for row in gaps if row["raw"] == raw and row["source"] == "C"}
        if all(stage in rows for stage in STAGES):
            counts = [rows[stage]["certified_road_low_alpha_total_n"] for stage in STAGES]
            denominator = rows["parent"]["certified_road_n"]
            findings.append(f"帧{raw}认证孔洞域α<0.95：起点{counts[0]}→零步{counts[1]}→拟合{counts[2]}，分母{denominator}。")
            coverage_worse |= counts[2] > counts[0]
    rows86 = {row["id"]: row for row in boundary if row["raw"] == 86}
    if all(stage in rows86 for stage in ("parent", "appearance")):
        initial, final = rows86["parent"]["A_boundary_gradient_MAE"], rows86["appearance"]["A_boundary_gradient_MAE"]
        if initial is not None and final is not None and final > initial:
            findings.append(f"帧86真实A边界梯度误差由起点{initial:.5f}升至拟合{final:.5f}。")
    rows110 = {row["id"]: row for row in boundary if row["raw"] == 110}
    if all(stage in rows110 for stage in STAGES):
        values = [rows110[stage]["seam_RGB_jump"] for stage in STAGES]
        if all(value is not None for value in values) and values[2] > values[0]:
            findings.append(f"帧110接缝RGB差为起点{values[0]:.5f}→零步{values[1]:.5f}→拟合{values[2]:.5f}。")
    diagnostics["visual_findings"] = " ".join(findings)
    diagnostics["outcome"] = "迁移写入可用，质量未胜原资产；认证孔洞域覆盖未超过原资产。" if coverage_worse else "迁移写入与约束核验完成；图像质量按实际对照评价。"
    ledger_path = run / "runtime/resource_ledger.json"
    ledger = read(ledger_path)
    if ledger and ledger.get("authoritative") is True:
        shutil.copy2(ledger_path, output / "resource_ledger.json")
        payload["resources"] = {key: ledger.get(key) for key in ("totals", "accounting_definitions", "scope", "authoritative")}
        payload["summary"]["resource_totals"] = ledger["totals"]
        payload["evidence"] = [item for item in payload["evidence"] if item.get("name") != "累计资源账本" and not item.get("name", "").startswith("原始末进程快照")]
        payload["evidence"].append({"name": "累计资源账本", "path": "resource_ledger.json"})
        snapshot = run / "engineering/last_process_resource_snapshot.json"
        if not snapshot.exists() and read(run / "resource_ledger.json", {}).get("authoritative") is not True:
            snapshot = run / "resource_ledger.json"
        if snapshot.exists():
            destination = output / "evidence/process_resource_snapshot.json"; destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(snapshot, destination)
            payload["evidence"].append({"name": "原始末进程快照（非累计）", "path": destination.relative_to(output).as_posix()})
    archive_manifest = run / "runtime/archive_manifest.json"
    if archive_manifest.exists():
        destination = output / "runtime/archive_manifest.json"; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive_manifest, destination)
        if not any(item.get("name") == "恢复与归档依赖" for item in payload["evidence"]):
            payload["evidence"].append({"name": "恢复与归档依赖", "path": destination.relative_to(output).as_posix()})
    write(output / "diagnostics.json", diagnostics); write(output / "summary.json", payload["summary"])
    write(output / "report_data.json", payload)
    return payload


def archive_report(run, output, summary):
    archive = run / "report_ready_bundle.tar.gz"; temporary = archive.with_name(archive.name + ".tmp")
    with tarfile.open(temporary, "w:gz") as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                bundle.add(path, arcname=path.relative_to(output).as_posix())
    os.replace(temporary, archive)
    write(run / "report_packaging_complete.json", {**summary, "state": "complete" if summary["complete"] else "partial",
                                                "bundle": str(archive), "bundle_bytes": archive.stat().st_size})
    return archive


def main(args):
    started = time.monotonic(); run = args.run.resolve(); output = (args.output_dir or run / "report_ready").resolve()
    write(run / "report_packaging_complete.json", {"complete": False, "state": "building", "GPU_calls": 0})
    output.mkdir(parents=True, exist_ok=True); files = PacketFiles(run, output)
    inputs = read(args.manifest or run / "report_inputs.json", {})
    source_receipt = read(run / "source_native/receipt.json")
    if not source_receipt or not source_receipt.get("complete"):
        raise ValueError("真实源块原生渲染核验未完成")
    stages = [stage for stage in STAGES if all((run / stage / "rendered" / f"{raw:03d}.npz").exists() for raw in FRAMES)
              and (stage == "parent" or read(run / stage / "receipt.json", {}).get("complete") is True)]
    if not {"parent", "transfer"}.issubset(stages) or not args.allow_partial and stages != list(STAGES):
        raise ValueError("阶段五帧尚未完整；仅0step预览可用--allow-partial")
    domains = {raw: load(args.base_domains / f"{raw:03d}.npz") for raw in (86, 110)}
    decision = read(run / "proposal/source_decision.json", {})
    primary_source = decision.get("chosen", "patch_00")
    patches = {}
    for patch_id in ("patch_00", "patch_02"):
        folder = run / "source_native" / patch_id
        if not folder.exists():
            continue
        frames = {}
        for raw in (86, 110):
            frame = {}
            for key in ("observation", "isolated", "full_road"):
                source = folder / f"{raw:03d}_{key}_crop.png"
                if source.exists():
                    frame[key] = files.copy(source)
            if frame:
                frames[str(raw)] = frame
        patches[patch_id] = {"label": "源块00" if patch_id == "patch_00" else "源块02", "frames": frames,
                             "role": "primary" if patch_id == primary_source else "source_verification_control"}
    if primary_source not in patches:
        raise ValueError("缺少实际迁入源块的原照/原高斯证据")
    models = {stage: {"id": stage, "name": NAMES[stage], "frames": {}, "renderer": "actual1152_native3DGS",
                      "budget": "0更新" if stage in ("parent", "transfer") else "局部接缝拟合（见记录）"} for stage in stages}
    parent_asset = args.parent_asset
    with np.load(parent_asset / "road.npz", allow_pickle=False) as packet:
        parent_active = packet["active"].astype(bool)
    for stage in stages:
        asset = parent_asset if stage == "parent" else run / stage / "asset"
        models[stage]["asset_dir"] = str(asset)
        models[stage]["point_counts"] = stage_counts(asset, parent_active, len(parent_active))
        receipt = read(run / stage / "receipt.json", {})
        updates = next((receipt[key] for key in ("formal_optimizer_updates", "formal_updates", "steps", "fit_steps", "cursor") if key in receipt), None)
        if stage == "appearance" and updates is not None:
            models[stage]["name"] += f" · {updates}步"; models[stage]["budget"] = f"局部{updates}步"
    gaps, protection, boundary = [], [], []
    boundary_protocol = {}
    for raw in FRAMES:
        packets = {stage: load(run / stage / "rendered" / f"{raw:03d}.npz") for stage in stages}
        parent = packets["parent"]; K = np.asarray(parent["K"], bool)
        if K.shape != (768, 1152):
            raise ValueError("K尺寸错误")
        common_stable = np.logical_and.reduce([packet["road_alpha"] >= .95 for packet in packets.values()])
        if raw in domains:
            domain = domains[raw]
            target_path = inputs.get("target_masks", {}).get(str(raw))
            if target_path:
                target = mask_file(Path(target_path) if Path(target_path).is_absolute() else run / target_path)
                boundary_protocol[str(raw)] = "显式目标投影mask的邻接A带"
            else:
                target = (domain["appearance_C"] | domain["appearance_S"]) & domain["canonical_valid"]
                boundary_protocol[str(raw)] = "原认证C/S域的邻接A带"
            observed = domain["appearance_A"].astype(bool)
            observation = rgb(np.asarray(Image.open(args.observation_root / f"raw{raw:03d}/observation_1152.png")))
            initial_stable = (packets["parent"]["road_alpha"] >= .95) & (packets["transfer"]["road_alpha"] >= .95)
            A_fixed = observed & ndi.binary_dilation(target, iterations=8) & initial_stable
            A_band = A_fixed & common_stable
            gradient_mask = A_band & ndi.binary_erosion(observed & common_stable, iterations=1)
            observation_gradient = np.stack([ndi.sobel(observation, axis=axis, mode="nearest") / 8 for axis in (0, 1)], axis=-1)
        for stage, packet in packets.items():
            road, joint, alpha = rgb(packet["road_rgb"]), rgb(packet["joint_rgb"]), np.asarray(packet["road_alpha"])
            if alpha.shape != (768, 1152) or not np.isfinite(alpha).all() or not np.array_equal(np.asarray(packet["K"], bool), K):
                raise ValueError("alpha/K包格式不一致")
            if np.asarray(packet["joint_alpha"]).shape != (768, 1152) or not np.isfinite(packet["joint_alpha"]).all():
                raise ValueError("joint alpha包格式错误")
            rgb_delta = float(np.abs(joint[K] - rgb(parent["joint_rgb"])[K]).max()) if K.any() else 0.
            alpha_delta = float(np.abs(packet["joint_alpha"][K] - parent["joint_alpha"][K]).max()) if K.any() else 0.
            protection.append({"id": stage, "raw": raw, "K_RGB_max": rgb_delta, "K_alpha_max": alpha_delta})
            if rgb_delta != 0 or alpha_delta != 0:
                raise ValueError(f"保留车辆发生变化: {stage}/{raw}")
            frame = {}
            for view, image in (("road", road), ("joint", joint), ("alpha", alpha)):
                path = save(output / "models" / stage / f"{raw:03d}_{view}.png", image)
                frame[view] = path.relative_to(output).as_posix()
                if raw in domains:
                    for region in REGIONS:
                        x0, y0, x1, y1 = bounds(domains[raw]["appearance_" + region])
                        crop = save(path.with_name(f"{raw:03d}_{region}_{view}_crop.png"), image[y0:y1, x0:x1])
                        frame[region + "_" + view + "_crop"] = crop.relative_to(output).as_posix()
            models[stage]["frames"][str(raw)] = frame
            if raw not in domains:
                continue
            pred = effective(packet)
            for region in REGIONS:
                mask = domain["appearance_" + region].astype(bool); certified = mask & domain["canonical_valid"]
                gaps.append({"id": stage, "raw": raw, "source": region, "denominator": int(mask.sum()),
                             "certified_road_n": int(certified.sum()), "uncertified_editing_n": int((mask & ~certified).sum()),
                             "no_support_n": int((mask & (alpha < .05)).sum()), "low_alpha_n": int((mask & (alpha >= .05) & (alpha < .95)).sum()),
                             "certified_road_no_support_n": int((certified & (alpha < .05)).sum()),
                             "certified_road_low_alpha_n": int((certified & (alpha >= .05) & (alpha < .95)).sum()),
                             "certified_road_low_alpha_total_n": int((certified & (alpha < .95)).sum()),
                             "high_alpha_dark_n": None, "joint_static_occluded_n": None})
            prediction_gradient = np.stack([ndi.sobel(pred, axis=axis, mode="nearest") / 8 for axis in (0, 1)], axis=-1)
            boundary.append({"id": stage, "raw": raw, "A_boundary_n": int(A_band.sum()),
                             "A_fixed_n": int(A_fixed.sum()),
                             "A_fixed_premult_RGB_MAE": average_error(road, observation, A_fixed),
                             "A_fixed_alpha95_fraction": float((alpha[A_fixed] >= .95).mean()) if A_fixed.any() else None,
                             "A_boundary_RGB_MAE": average_error(pred, observation, A_band),
                             "A_gradient_n": int(gradient_mask.sum()),
                             "A_boundary_gradient_MAE": float(np.abs(prediction_gradient[gradient_mask] - observation_gradient[gradient_mask]).mean()) if gradient_mask.any() else None,
                             **seam_jump(pred, target, observed, common_stable)})
        print(json.dumps({"phase": "高斯迁移真实图像与边界包装", "raw": raw, "stages": stages}, ensure_ascii=False), flush=True)
    texture_rows = inputs.get("source_texture_rows", [])
    if not texture_rows:
        texture_packet = read(run / "diagnostics/source_texture_preservation.json", {})
        texture_rows = texture_packet if isinstance(texture_packet, list) else texture_packet.get("rows", [])
    target_receipt = read(run / "target/receipt.json", {})
    summary = {"complete": stages == list(STAGES), "stages": stages, "run_id": run.name, "transport_method": "rigid_isometric_patch_registration", "primary_source_patch": primary_source,
               "source_selection_matches_expected_00": primary_source == "patch_00",
               "source_verification_control": next((key for key in patches if key != primary_source), None), "new_diffusion_calls": 0,
               "target_grid_area_m2": target_receipt.get("target_grid_area_m2"),
               "keep_same_position_count": target_receipt.get("keep_same_position_count"),
               "K_exact": True, "CPU_seconds": time.monotonic() - started, "production_promoted": False}
    payload = {"schema": "road_gaussian_transfer_report.v1", "models": list(models.values()), "source_patches": patches,
               "source_layers": files.images(inputs.get("source_layers", {"frames": {}})), "summary": summary,
               "gaps": gaps, "diagnostics": {"source_quality_rows": source_receipt["quality"], "protection_rows": protection,
                                              "boundary_rows": boundary, "source_texture_rows": texture_rows},
               "scope": "相似路块推断；保距路块配准，三维空间与协方差随刚体旋转；先零步验收，再局部接缝拟合。", "evidence": []}
    preservation = inputs.get("parameter_preservation", {})
    payload["diagnostics"]["parameter_preservation"] = preservation
    preservation_rows = []
    for key, label in (("source_RGB_exact_zero_step", "零步源RGB逐值保持"),
                       ("source_sigma_exact_zero_step", "零步源尺度保持"),
                       ("rigid_covariance", "协方差随刚体旋转"),
                       ("source_asset_retained", "原源块资产保留")):
        if key in preservation:
            preservation_rows.append({"stage": "transfer", "label": label, "passed": bool(preservation[key])})
    if "individual_RGB_parameters" in preservation:
        preservation_rows.append({"stage": "appearance", "label": "逐高斯自由颜色参数", "value": preservation["individual_RGB_parameters"],
                                  "passed": preservation["individual_RGB_parameters"] == 0})
    freeze = read(run / "runtime/seam_fit_cpu_review.json", {})
    checks = freeze.get("checks", {})
    for label, keys in (("中心与法向冻结", ["model_xyz", "normal_world"]),
                        ("旋转与法线尺度冻结", ["gaussian_rotation_world_wxyz", "all_normal_sigma_exact"]),
                        ("既有高斯参数冻结", ["old_RGB_exact", "old_opacity_exact", "old_sigma_exact"]),
                        ("新核心覆盖与尺度冻结", ["new_core_opacity_exact", "new_core_sigma_exact"]),
                        ("接缝覆盖与尺度在界内", ["opacity_bounds", "sigma_multiplier_bounds"]),
                        ("新颜色符合统一校色", ["new_global_color_transform_exact"])):
        if all(key in checks for key in keys):
            preservation_rows.append({"stage": "appearance", "label": label, "passed": all(checks[key] is True for key in keys)})
    payload["diagnostics"]["parameter_preservation_rows"] = preservation_rows
    visual = read(run / "runtime/seam_fit_visual_review.json", {})
    if visual:
        payload["diagnostics"]["visual_findings"] = "道路层仍可见细黑裂缝、短水平缺口与局部断续边界；联合渲染仍有车形残影。"
        contact = run / "runtime/seam_fit_review/before_after_CS_crops.png"
        if contact.exists():
            payload["diagnostics"]["visual_contact_sheet"] = files.copy(contact)
    for path, name in ((run / "source_native/receipt.json", "源实拍与原高斯核验"),
                       (run / "source_selection/manifest.json", "源块选择与来源"),
                       (run / "target/receipt.json", "目标道路表面与范围"),
                       (run / "qualification/native_rotation_compatibility.json", "原生旋转兼容核验"),
                       (run / "transfer/receipt.json", "零步配准记录"), (run / "proposal/receipt.json", "保距构造与源纹理保持"),
                       (run / "proposal/source_decision.json", "实际源块决策"), (run / "appearance/public_reload.json", "接缝资产公共重载"),
                       (run / "smoke/fit_seams/receipt.json", "原生短程与完整恢复"), (run / "recovery/receipt.json", "恢复后下一入口"), (run / "appearance/receipt.json", "局部接缝拟合记录"),
                       (run / "runtime/seam_fit_cpu_review.json", "独立参数约束核验"),
                       (run / "runtime/seam_fit_visual_review.json", "独立视觉审阅"),
                       (run / "runtime/resource_ledger.json", "累计资源账本"),
                       (run / "resource_ledger.json", "原始末进程快照"), (run / "compute_ledger.json", "正式与工程计算记录")):
        if path.exists():
            payload["evidence"].append({"name": name, "path": files.copy(path)})
    for item in inputs.get("evidence", []):
        payload["evidence"].append({"name": item.get("name", item.get("label", "工程记录")), "path": files.copy(item["path"])})
    protocol = {"coverage": "原固定A/C/S二维域与认证道路域分别统计；alpha<.05无支持，.05至.95薄覆盖",
                "A_boundary": boundary_protocol, "A_boundary_band_pixels": 8,
                "A_boundary_reference": "原始真实观测；固定起点/零步稳定A带同时报告完整预乘误差和充分覆盖，颜色误差另取三阶段共同alpha>=.95，梯度另取A内严格1px核心",
                "seam": "目标/真实A相邻4邻域的稳定材质RGB跳变，非隐藏道路真值误差",
                "source_texture": "由canonical/source_uv对应的源块原高斯保持评价单列",
                "source_photo": "沿用实际源块原生核验mask/数值，不重算不同口径",
                "new_diffusion_calls": 0, "GPU_calls": 0}
    write(output / "protocol.json", protocol)
    write(output / "diagnostics.json", payload["diagnostics"])
    payload["evidence"].extend([{"name": "评价口径", "path": "protocol.json"}, {"name": "覆盖外的独立诊断", "path": "diagnostics.json"}])
    if inputs.get("disclaimer"):
        payload["disclaimer"] = inputs["disclaimer"]
    write(output / "summary.json", summary); write(output / "report_data.json", payload)
    payload = refresh_report_metadata(run, output, payload)
    archive = archive_report(run, output, payload["summary"])
    print(json.dumps({"complete": summary["complete"], "bundle": str(archive), "stages": stages}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", "--root", dest="run", type=Path, default=CAMPAIGN / "road_gaussian_transfer_20261122_r2_isometric")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--base-domains", type=Path, default=CAMPAIGN / "road_surface_residual_20261121_r2/proposal/domains")
    parser.add_argument("--observation-root", type=Path, default=CAMPAIGN / "road_common1152_20261121/teacher")
    parser.add_argument("--parent-asset", type=Path, default=CAMPAIGN / "road_surface_residual_20261121_r2/appearance_fit/asset")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--refresh-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.refresh_only:
            output = (args.output_dir or args.run / "report_ready").resolve()
            payload = refresh_report_metadata(args.run.resolve(), output, read(output / "report_data.json"))
            archive = archive_report(args.run.resolve(), output, payload["summary"])
            print(json.dumps({"complete": payload["summary"]["complete"], "metadata_refreshed": True, "bundle": str(archive)}, ensure_ascii=False), flush=True)
        else:
            main(args)
    except Exception:
        failure = {"error": traceback.format_exc(), "GPU_calls": 0}
        history = args.run / "report_failures" / (str(time.time_ns()) + ".json")
        write(history, failure)
        write(args.run / "report_packaging_failure.json", failure)
        write(args.run / "report_packaging_complete.json", {"complete": False, "state": "failed", "error_file": str(history), "GPU_calls": 0})
        raise

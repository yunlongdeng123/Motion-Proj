"""只读复算静态 gate 候选，并追踪真实联合删车渲染的逐行 alpha*T。

仅写入全新的 --output 目录；不创建 optimizer，不修改模型、gate 或 owner。
候选复算逐行继承 v44_gate_fit 的原始 marker、阈值、最大值与保护规则。
区域 fit/keep/outside 均与目标输入 mask 相交，不把未知 owner 当成车辆。
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np


REASON_NAMES = {1: "fit_score_le_1e-4", 2: "guard_gt_20_fit", 4: "positive_fixed_owner"}
EXCLUSIVE_NAMES = {
    0: "selected",
    1: "low_fit_score",
    2: "guard_dominant_after_fit_pass",
    3: "positive_owner_after_fit_and_guard_pass",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def row_counts(value):
    keys, counts = np.unique(value, return_counts=True)
    return {str(int(k)): int(n) for k, n in zip(keys, counts)}


def resolved(base, path):
    path = Path(path)
    return path if path.is_absolute() else base / path


def deletion_ids(manifest, query):
    ids = set(int(x) for x in query.get("delete_actor_ids", []))
    ids.update(manifest.actor_ids(query.get("delete_track_ids", [])))
    if not ids or any(x <= 0 or x not in manifest.actor_to_track for x in ids):
        raise ValueError("需要显式登记的正整数删除 ID")
    return tuple(sorted(ids))


def view_entries(run):
    root = run / "world_road" / "views"
    entries = read_json(root / "manifest.json")["frames"]
    result = {r["key"]: resolved(root, r["view"]) for r in entries}
    if len(result) != len(entries):
        raise ValueError("道路 view 清单存在重复 frame key")
    return result


def recompute_selection(run, output, query, views, torch):
    from drivingeditor_dynamic_edit.v44_factory import from_manifest

    started = time.monotonic()
    scene, manifest = from_manifest(run / "road_manifest.json")
    if manifest.static_gates is not None or scene._gates is not None:
        raise ValueError("候选复算必须使用原 ungated road_manifest")
    scene.assert_persistent_unchanged()
    deleted = deletion_ids(manifest, query)
    count = len(scene.static.get_xyz)
    device = scene.static.get_xyz.device
    score = torch.zeros(count, device=device)
    guard = torch.zeros_like(score)
    owners_cpu = scene._owners_s.detach().cpu().numpy().copy()
    ops = scene._bindings()
    selected_frames = []
    frame_rows = []
    for index, frame in enumerate(manifest.frames):
        if frame.key not in views:
            raise ValueError("缺少 road view: " + str(frame.key))
        with np.load(views[frame.key], allow_pickle=False) as view:
            fit_mask = (view["fit"] > .05) & ~view["marking"]
            protect = view["other"] | view["marking"]
        frame_record = dict(key=frame.key, fit_pixels=int(fit_mask.sum()),
                            protect_pixels=int(protect.sum()), skipped=not bool(fit_mask.any()))
        frame_rows.append(frame_record)
        if not fit_mask.any():
            continue
        # 原始算法在 no-grad 参数值上仅为 marker 建图；不会产生模型参数梯度。
        with torch.no_grad():
            values, owners, camera, _ = scene._rows(frame.key, (), (), False)
        features = values["features"].detach().clone()
        if features.shape[1] != 24:
            raise ValueError("需要 24-feature ABI")
        features[:, 8] = 0
        features[:count, 8] = 1
        features.requires_grad_(True)
        mask = values["mask"] & ~torch.isin(owners, owners.new_tensor(deleted))
        pack = ops.raster(camera, scene.pipe, 0, dict(values, features=features, mask=mask),
                          features.new_zeros(3))
        mass = pack["extra"][6]
        fit_gpu = torch.as_tensor(fit_mask, device=device)
        protect_gpu = torch.as_tensor(protect, device=device)
        s = torch.autograd.grad(mass[fit_gpu].sum(), features, retain_graph=True)[0][:count, 8]
        g = torch.autograd.grad(mass[protect_gpu].sum(), features)[0][:count, 8]
        score = torch.maximum(score, s)
        guard = torch.maximum(guard, g)
        selected_frames.append(frame.key)
        del pack, mass, values, features, mask, owners, camera, s, g, fit_gpu, protect_gpu
        if index == 0 or (index + 1) % 20 == 0 or index + 1 == len(manifest.frames):
            print(json.dumps(dict(phase="recompute_selection", frames=index + 1,
                                  total_frames=len(manifest.frames), seconds=time.monotonic() - started)), flush=True)
    eligible = (score > 1e-4) & ~(guard > 20 * score) & (scene._owners_s <= 0)
    actual_ids = torch.nonzero(eligible).flatten().detach().cpu().numpy()
    score_cpu = score.detach().cpu().numpy().copy()
    guard_cpu = guard.detach().cpu().numpy().copy()
    with np.load(run / "gate1000" / "contributors.npz", allow_pickle=False) as saved:
        expected_ids = saved["ids"].copy()
        expected_fit = saved["fit_score"].copy()
        expected_guard = saved["guard_score"].copy()
    if expected_ids.dtype.kind not in "iu" or expected_ids.ndim != 1:
        raise ValueError("保存的 contributor IDs 格式不合法")
    if np.any(expected_ids < 0) or np.any(expected_ids >= count) or len(np.unique(expected_ids)) != len(expected_ids):
        raise ValueError("保存的 contributor IDs 超界或重复")
    selected = np.zeros(count, dtype=bool)
    selected[expected_ids] = True
    low = score_cpu <= 1e-4
    protected = guard_cpu > 20 * score_cpu
    positive = owners_cpu > 0
    reason_bits = low.astype(np.uint8) | (protected.astype(np.uint8) << 1) | (positive.astype(np.uint8) << 2)
    # 互斥原因按原判断顺序分配，交叠原因另外完整保留。
    exclusive = np.zeros(count, dtype=np.uint8)
    exclusive[low] = 1
    exclusive[~low & protected] = 2
    exclusive[~low & ~protected & positive] = 3
    ids_equal = np.array_equal(actual_ids, expected_ids)
    fit_error = float(np.max(np.abs(score_cpu[expected_ids] - expected_fit), initial=0.))
    guard_error = float(np.max(np.abs(guard_cpu[expected_ids] - expected_guard), initial=0.))
    recomputed = np.zeros(count, dtype=bool)
    recomputed[actual_ids] = True
    np.savez_compressed(output / "selection_full.npz", fit_score=score_cpu,
                        guard_score=guard_cpu, fixed_static_owner=owners_cpu,
                        selected_saved=selected, eligible_recomputed=recomputed,
                        exclusion_reason_bits=reason_bits, exclusion_exclusive=exclusive)
    summary = dict(
        phase="selection", status="PASS" if ids_equal else "MISMATCH", declared_frames=len(manifest.frames),
        participating_frames=selected_frames, frame_regions=frame_rows, static_rows=count,
        deleted_actor_ids=list(deleted), saved_candidates=len(expected_ids), recomputed_candidates=len(actual_ids),
        eligible_ids_equal_saved=ids_equal,
        only_saved_count=int((selected & ~recomputed).sum()), only_recomputed_count=int((recomputed & ~selected).sum()),
        selected_fit_score_maxabs_vs_saved=fit_error, selected_guard_score_maxabs_vs_saved=guard_error,
        exclusion_bits={str(k): v for k, v in REASON_NAMES.items()},
        overlap_reason_bit_pattern_row_counts=row_counts(reason_bits),
        nonexclusive_rule_failure_counts={REASON_NAMES[1]: int(low.sum()), REASON_NAMES[2]: int(protected.sum()), REASON_NAMES[4]: int(positive.sum())},
        exclusive_rule="低 fit 优先，其次 guard，最后 positive owner；代码0为通过三条件",
        exclusive_names={str(k): v for k, v in EXCLUSIVE_NAMES.items()},
        exclusive_row_counts=row_counts(exclusive), fixed_owner_row_counts=row_counts(owners_cpu),
        saved_candidate_fixed_owner_row_counts=row_counts(owners_cpu[expected_ids]),
        gate_selection_equation="(max_frame_fit_alphaT>1e-4) & ~(max_frame_guard_alphaT>20*max_frame_fit_alphaT) & (fixed_static_owner<=0)",
        max_operation="fit 与 guard 分别取逐帧最大值；二者最大值可以来自不同帧",
        native_parameters_unchanged=True, training_updates=0, optimizer_steps=0,
        seconds=time.monotonic() - started)
    scene.assert_persistent_unchanged()
    write_json(output / "selection_summary.json", summary)
    if not ids_equal:
        raise RuntimeError("复算的 eligible IDs 与 gate1000 保存结果不同，已保留全量原因，停止后续归因")
    return dict(score=score_cpu, guard=guard_cpu, owners=owners_cpu, selected=selected,
                reasons=reason_bits, exclusive=exclusive, summary=summary)


def mass_partition(weights, subset, denominator):
    value = float(weights[subset].sum(dtype=np.float64))
    return dict(static_rows=int(subset.sum()), visible_static_rows=int(np.count_nonzero(weights[subset] > 0)),
                alpha_T_sum=value, fraction_of_region_static_alpha_T=value / denominator if denominator else None)


def top_rows(weights, subset, data, gates, limit=64):
    indices = np.flatnonzero(subset & (weights > 0))
    if len(indices) > limit:
        indices = indices[np.argpartition(weights[indices], -limit)[-limit:]]
    indices = indices[np.argsort(-weights[indices], kind="stable")]
    return [dict(row=int(i), alpha_T_sum=float(weights[i]), fixed_owner=int(data["owners"][i]),
                 selected=bool(data["selected"][i]), exclusion_reason_bits=int(data["reasons"][i]),
                 exclusion_exclusive=EXCLUSIVE_NAMES[int(data["exclusive"][i])],
                 max_fit_score=float(data["score"][i]), max_guard_score=float(data["guard"][i]),
                 final_opacity_multiplier=float(gates[i])) for i in indices]


def trace_final(run, output, query, views, data, torch, frames):
    from PIL import Image
    from drivingeditor_dynamic_edit.v44_factory import from_manifest

    scene, manifest = from_manifest(run / "final_manifest.json")
    scene.assert_persistent_unchanged()
    if not scene._include_dynamic:
        raise ValueError("最终场景必须保留实际 native dynamic 分支")
    deleted = deletion_ids(manifest, query)
    if tuple(data["summary"]["deleted_actor_ids"]) != deleted:
        raise ValueError("候选复算与最终场景的删除 ID 不一致")
    count = len(scene.static.get_xyz)
    if not np.array_equal(scene._owners_s.detach().cpu().numpy(), data["owners"]):
        raise ValueError("road/final 场景的静态行归属或行顺序不同")
    gates = np.ones(count, dtype=np.float32) if scene._gates is None else scene._gates.detach().cpu().numpy().copy()
    saved_root = run / "delete_white"
    saved_by_key = {r["frame_key"]: r for r in read_json(saved_root / "provenance.json")["frames"]}
    records = {r["key"]: r for r in manifest.data["frames"]}
    ops = scene._bindings()
    device = scene.static.get_xyz.device
    results = []
    for key in frames:
        started = time.monotonic()
        if key not in records or key not in views or key not in saved_by_key:
            raise ValueError("诊断帧缺少显式输入或已保存 final: " + str(key))
        with torch.no_grad():
            values, owners, camera, coverage = scene._rows(key, deleted, (), False)
        height, width = int(camera.image_height), int(camera.image_width)
        instance_path = resolved(manifest.base, records[key]["instance_mask"])
        labels = np.asarray(Image.open(instance_path).resize((width, height), Image.Resampling.NEAREST))
        target = np.isin(labels, np.asarray(deleted))
        with np.load(views[key], allow_pickle=False) as view:
            masks = dict(target=target, fit=target & (view["fit"] > .05),
                         keep=target & (view["keep"] > 0), outside=target & ~view["inside"])
            inside_unfit = int((target & view["inside"] & ~(view["fit"] > .05)).sum())
            target_other = int((target & view["other"]).sum())
            target_marking = int((target & view["marking"]).sum())
        features = values["features"].detach().clone()
        if features.shape[1] != 24:
            raise ValueError("需要 24-feature ABI")
        features[:, 8] = 0
        features[:count, 8] = 1
        features.requires_grad_(True)
        mask = values["mask"] & ~torch.isin(owners, owners.new_tensor(deleted))
        pack = ops.raster(camera, scene.pipe, 0, dict(values, features=features, mask=mask), scene._background)
        if pack["extra"].shape[0] != 22:
            raise ValueError("光栅输出未保留 24-feature ABI")
        with torch.no_grad():
            environment = scene.environment(camera.get_world_directions(False).permute(1, 2, 0)).permute(2, 0, 1)
            rgb = pack["rgb"] + (1 - pack["alpha"]) * environment
            saved_entry = saved_by_key[key]
            with np.load(saved_root / saved_entry["files"]["float_npz"], allow_pickle=False) as stored:
                saved_rgb = stored["rgb"].copy()
            parity = float(np.max(np.abs(rgb.detach().cpu().numpy() - saved_rgb), initial=0.))
            group_error = float((pack["extra"][:6].sum(0, keepdim=True) - pack["alpha"]).abs().max())
            target_alpha = float(pack["extra"][0].abs().max())
        if parity > 3e-6:
            raise RuntimeError("最终 RGB parity 失败: " + str(parity))
        if group_error > 2e-5 or target_alpha > 2e-6:
            raise RuntimeError("alpha 分组守恒或 target 删除失败")
        static_mass = pack["extra"][6]
        region_summaries = {}
        arrays = {}
        for index, (name, region) in enumerate(masks.items()):
            region_gpu = torch.as_tensor(region, device=device)
            # marker 只影响 feature，不影响透明度、颜色、几何或深度排序。
            gradient = torch.autograd.grad(static_mass[region_gpu].sum(), features,
                                           retain_graph=index < len(masks) - 1)[0]
            weights = gradient[:count, 8].detach().cpu().numpy().copy()
            del gradient, region_gpu
            if not np.isfinite(weights).all() or float(weights.min(initial=0.)) < -1e-7:
                raise RuntimeError("逐行 alpha*T 出现非有限或负权重")
            image_mass = float(static_mass.detach().cpu().numpy()[region].sum(dtype=np.float64))
            total = float(weights.sum(dtype=np.float64))
            error = abs(total - image_mass)
            if error > .005 + 5e-5 * abs(image_mass):
                raise RuntimeError("逐行 alpha*T 守恒失败: " + str(error))
            total_scene_alpha = float(pack["alpha"][0].detach().cpu().numpy()[region].sum(dtype=np.float64))
            selected = data["selected"]
            by_owner = {str(int(owner)): mass_partition(weights, data["owners"] == owner, total)
                        for owner in np.unique(data["owners"][weights > 0])}
            by_pattern = {str(int(reason)): mass_partition(weights, data["reasons"] == reason, total)
                          for reason in np.unique(data["reasons"])}
            by_exclusive = {EXCLUSIVE_NAMES[reason]: mass_partition(weights, data["exclusive"] == reason, total)
                            for reason in EXCLUSIVE_NAMES}
            by_rule = {label: mass_partition(weights, (data["reasons"] & bit) != 0, total)
                       for bit, label in REASON_NAMES.items()}
            owner_selection = {}
            for owner in np.unique(data["owners"][weights > 0]):
                for flag in (True, False):
                    subset = (data["owners"] == owner) & (selected == flag)
                    owner_selection[str(int(owner)) + ("_selected" if flag else "_unselected")] = mass_partition(weights, subset, total)
            region_summaries[name] = dict(
                pixels=int(region.sum()), static_alpha_T_sum=total,
                static_alpha_T_per_pixel=total / int(region.sum()) if region.any() else None,
                total_scene_alpha_T_sum=total_scene_alpha,
                static_fraction_of_scene_alpha_T=total / total_scene_alpha if total_scene_alpha else None,
                row_sum_vs_image_abs_error=error,
                row_sum_vs_image_relative_error=error / abs(image_mass) if image_mass else None,
                selected=mass_partition(weights, selected, total), unselected=mass_partition(weights, ~selected, total),
                fixed_owner=by_owner, owner_cross_selected=owner_selection,
                overlapping_rule_failures=by_rule, overlapping_reason_bit_patterns=by_pattern,
                mutually_exclusive_reasons=by_exclusive,
                top_static_rows=top_rows(weights, np.ones(count, dtype=bool), data, gates),
                top_selected_rows=top_rows(weights, selected, data, gates),
                top_unselected_rows=top_rows(weights, ~selected, data, gates))
            arrays[name + "_static_alpha_T_sum_per_row"] = weights
            arrays[name + "_pixel_mask"] = region
        np.savez_compressed(output / (str(key) + "_static_alphaT.npz"), **arrays)
        summary = dict(frame_key=key, deleted_actor_ids=list(deleted), regions=region_summaries,
                       region_definition="target=输入目标 mask；fit/keep/outside 均与 target 相交；区域可重叠，禁止相加当成划分",
                       inside_target_unfit_pixels=inside_unfit, target_other_mask_overlap_pixels=target_other,
                       target_marking_overlap_pixels=target_marking, saved_final_RGB_maxabs=parity,
                       six_group_alpha_conservation_maxabs=group_error, deleted_target_alpha_maxabs=target_alpha,
                       provider_coverage=coverage, seconds=time.monotonic() - started)
        write_json(output / (str(key) + "_summary.json"), summary)
        results.append(summary)
        print(json.dumps(dict(phase="trace_final", frame=key, parity_maxabs=parity,
                              target_static_alpha_T=region_summaries["target"]["static_alpha_T_per_pixel"],
                              target_selected_fraction=region_summaries["target"]["selected"]["fraction_of_region_static_alpha_T"],
                              seconds=time.monotonic() - started)), flush=True)
        del pack, static_mass, values, owners, camera, features, mask, rgb, environment, arrays
        gc.collect()
    scene.assert_persistent_unchanged()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[119, 198])
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    base = dict(run=str(run), output=str(output), frames=args.frames, training_updates=0,
                optimizer_steps=0, unknown_owner_deletion=False, persistent_model_mutation=False)
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: 未执行光栅诊断")
        torch.cuda.reset_peak_memory_stats()
        query = read_json(run / "delete_white_query.json")
        views = view_entries(run)
        selection = recompute_selection(run, output, query, views, torch)
        gc.collect()
        torch.cuda.empty_cache()
        traces = trace_final(run, output, query, views, selection, torch, args.frames)
        receipt = dict(base, status="COMPLETE_READ_ONLY_SELECTION_TRACE", parent_unchanged=True,
                       eligible_ids_equal_saved=selection["summary"]["eligible_ids_equal_saved"],
                       selection_seconds=selection["summary"]["seconds"],
                       traced_frames=[r["frame_key"] for r in traces],
                       peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                       peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved()),
                       seconds=time.monotonic() - started,
                       interpretation="逐行 alpha*T 是分支/候选贡献，不是像素车辆身份真值；未选原因使用 ungated 全帧原规则精确复算")
        write_json(output / "receipt.json", receipt)
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
    except Exception as error:
        write_json(output / "receipt.json", dict(base, status="FAILED", error=repr(error),
                                                seconds=time.monotonic() - started))
        raise


if __name__ == "__main__":
    main()

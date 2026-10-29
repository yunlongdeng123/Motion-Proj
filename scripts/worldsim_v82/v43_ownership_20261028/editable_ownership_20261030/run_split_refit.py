"""V43 固定 proposal 的 200 步双侧保留 refit 与条件删除短探针。

冻结 source actor/static/env -> 固定 7 parent -> 14 固定几何 child
                              -> 仅 DC / opacity refit -> 保存/恢复核对
                              -> 两域 alpha*T 与三项删除对照

只对同帧冻结 native RGB 做 distillation；没有 RGB GT、语义训练或 owner
真值。gt 侧仅沿原 proposal 的几何 target 假设，leq 侧是 road 假设。
child 位置、旋转、正定 covariance、SH 高阶与 parent 清单全程固定。
actor-present 与 actor-removed 上下文都重新渲染对照/贡献，不能复用透射率。
阴影、未见表面与最终编辑质量均不由本探针证明。不会自动扩大到其它 R2 行。

需要已运行的 run_scene1_probe.py smoke。--probe-dir 指向其输出根目录。
授权 JSON 须有 allow_child_refit=true、child_refit_updates=200。
--resume-checkpoint checkpoint000200.pth 只补做验收，累计 200 步、本次 0 步；
原现场不覆盖，除本文件修复 SHA 外所有 source bindings 必须一致。
本程序是 GPU 执行入口；作者只做本地 CPU 语法与辅助函数检查。
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
from PIL import Image

from run_scene1_probe import CTX, TASK, decode_assignment, native_adjoint, read, sha, write
from two_sided_split import covariance_from_quaternion_scale, covariance_to_quaternion_scale


UPDATES = 200
SEED = 43030
DC_LR = 2e-3
OPACITY_LR = 5e-3


def numpy_rng_equal(left, right) -> bool:
    """NumPy MT19937 的 uint32 状态直接比较，兼容旧版 Torch。"""
    return (len(left) == len(right) == 5 and left[0] == right[0]
            and np.array_equal(left[1], right[1]) and left[2:] == right[2:])


def validate_resume_bindings(previous: dict, current: dict) -> dict:
    """只允许本入口 bugfix 的 code SHA 变化；数据、模型、proposal 全部精确绑定。"""
    assert set(previous) == set(current), "Resume source binding keys differ"
    changed = [key for key in previous if previous[key] != current[key]]
    assert set(changed) <= {"refit_code_sha256"}, f"Resume source binding differs: {changed}"
    return dict(from_sha256=previous["refit_code_sha256"], to_sha256=current["refit_code_sha256"],
                permitted_change="run_split_refit.py acceptance-path bugfix only", changed_keys=changed)


def load_proposals(path: Path) -> tuple[list[dict], list[int]]:
    """只接受 smoke 已保留的双侧结果；绝不重新选父行或重新提切面。"""
    data = read(path)
    proposals = data["proposals"]
    parents = [int(proposal["row"]) for proposal in proposals]
    assert 0 < len(parents) <= 8 and len(set(parents)) == len(parents)
    assert data["retained"] == len(parents)
    for proposal in proposals:
        result = proposal["result"]
        assert result["native_safe"] and not result["needs_review"]
        assert str(result["parent_id"]) == str(proposal["row"])
        children = result["children"]
        assert len(children) == 2 and [child["side"] for child in children] == ["leq", "gt"]
        assert math.isclose(sum(child["mass_weight"] for child in children), 1, abs_tol=1e-12)
        for child in children:
            assert str(child["parent_id"]) == str(proposal["row"])
            assert child["opacity_candidate_safe"] and not child["needs_review"]
            assert 0 < child["opacity_candidate"] < 1, "Finite logits require strictly interior opacity; do not clip"
            covariance_to_quaternion_scale(np.asarray(child["covariance"]))
    return proposals, parents


def pixel_metrics(rgb: np.ndarray, reference: np.ndarray, alpha: np.ndarray,
                  reference_alpha: np.ndarray, masks: dict[str, np.ndarray]) -> dict:
    """纯 CPU 的 float 指标；空域返回 null，避免未知分母伪装成 0。"""
    difference = np.abs(rgb.astype(np.float64)-reference.astype(np.float64))
    delta = difference.mean(axis=0)
    alpha_delta = np.abs(np.asarray(alpha, np.float64).reshape(delta.shape)-
                         np.asarray(reference_alpha, np.float64).reshape(delta.shape))
    domains = {"full_frame": np.ones(delta.shape, dtype=bool), **masks}
    result = {}
    for name, mask in domains.items():
        mask = np.asarray(mask, dtype=bool)
        assert mask.shape == delta.shape
        count = int(mask.sum())
        result[name] = dict(pixels=count,
            mean_abs_rgb_change=float(delta[mask].mean()) if count else None,
            max_abs_rgb_change=float(difference[:, mask].max()) if count else None,
            rgb_mse=float((difference[:, mask]**2).mean()) if count else None,
            mean_abs_alpha_change=float(alpha_delta[mask].mean()) if count else None,
            changed_pixels_above_1_255=int((delta[mask] > 1/255).sum()) if count else 0)
    return result


def contribution_purity(weights: np.ndarray, children: list[dict]) -> dict:
    """两域条件纯度，分母明确为 body_mass+road_mass，不冒称全图纯度。"""
    weights = np.asarray(weights, dtype=np.float64)
    assert weights.shape == (len(children), 2) and np.isfinite(weights).all()
    # 只允许 adjoint 舍入级负值；实际负贡献属于接线错误，不能静默裁剪。
    assert float(weights.min(initial=0)) >= -1e-8
    rows = []
    for child, (body, road) in zip(children, weights):
        denominator = float(body+road)
        rows.append(dict(child_id=child["child_id"], parent_row=child["parent_row"], side=child["side"],
                         body_alpha_T_mass=float(body), road_alpha_T_mass=float(road),
                         two_domain_mass=denominator,
                         body_fraction=float(body/denominator) if denominator > 1e-12 else None,
                         road_fraction=float(road/denominator) if denominator > 1e-12 else None))
    groups = {}
    for side in ("leq", "gt"):
        selected = np.array([child["side"] == side for child in children])
        body, road = weights[selected].sum(axis=0)
        denominator = float(body+road)
        groups[side] = dict(body_alpha_T_mass=float(body), road_alpha_T_mass=float(road),
                           two_domain_mass=denominator,
                           body_fraction=float(body/denominator) if denominator > 1e-12 else None,
                           road_fraction=float(road/denominator) if denominator > 1e-12 else None)
    denominator = float(weights.sum())
    numerator = groups["gt"]["body_alpha_T_mass"] + groups["leq"]["road_alpha_T_mass"]
    return dict(denominator="native_alpha_T_mass_in_body_plus_safe_observed_road_domains",
                domain_masks_are_observations_not_semantic_GT=True,
                rows=rows, groups=groups,
                hypothesized_side_match_fraction=float(numerator/denominator) if denominator > 1e-12 else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--input-contract", type=Path, required=True)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--scope-file", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--raw", type=int, default=113)
    parser.add_argument("--resume-checkpoint", type=Path,
                        help="Only checkpoint200 acceptance in a fresh output; performs zero optimizer updates")
    args = parser.parse_args()
    assert args.raw == 113, "This bounded follow-up reuses only the completed raw113 smoke"
    assert not args.output.exists(), "Choose fresh output and preserve old attempts"
    proposals_path = args.probe_dir / f"{args.raw:03d}" / "split_proposals.json"
    proposals, parents = load_proposals(proposals_path)
    assert len(parents) == 7, "The authorized smoke binds exactly 7 parents / 14 children"
    smoke_summary = read(args.probe_dir / "summary.json")
    assert smoke_summary["task"] == TASK and smoke_summary["context"] == CTX
    assert smoke_summary["status"] == "complete" and smoke_summary["training_updates"] == 0
    smoke_record = next(row for row in smoke_summary["records"] if row["raw"] == args.raw)
    assert smoke_record["split"]["applied_rows"] == 7 and smoke_record["split"]["children"] == 14
    contract = read(args.input_contract)
    assert contract["context"] == CTX and contract["branch_sizes"] == dict(static=918850, actor=6241)
    assert contract["static_namespace"] == contract["static_checkpoint_ref"] + "@30000:static"
    ownership_path = args.input_contract.parent / contract["source_assets"]["ownership"]["path"]
    assert sha(ownership_path) == contract["source_assets"]["ownership"]["sha256"]
    ownership = decode_assignment(ownership_path)
    assert ownership["context"] == CTX and ownership["static_namespace"] == contract["static_namespace"]
    assert ownership["branch_sizes"] == contract["branch_sizes"] and ownership["raws"] == list(range(77, 126))
    assert sha(contract["actor_checkpoint_ref"]) == contract["actor_checkpoint_sha256"]
    auth = read(args.authorization)
    def authorized():
        latest = read(args.authorization)
        assert latest["task"] == TASK and latest["context"] == CTX and latest["allow_gpu_probe"]
        assert latest.get("allow_child_refit") is True and latest.get("child_refit_updates") == UPDATES
        assert not latest["allow_classifier_training"] and not latest["resume_old_queue"]
    authorized()
    sys.path.insert(0, str(args.code_dir))
    module = importlib.import_module("train_actor")
    source_maps = importlib.import_module("probe_target_unknown").source_maps
    scope_spec = importlib.util.spec_from_file_location("bounded_refit_scope", args.scope_file)
    scope_module = importlib.util.module_from_spec(scope_spec)
    scope_spec.loader.exec_module(scope_module)
    torch = module.torch
    assert torch.cuda.is_available(), "Requires the real native CUDA rasterizer"
    torch.set_num_threads(1)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    def status(phase, step=0, **extra):
        write(args.output / "status.json", dict(task=TASK, context=CTX, raw=args.raw,
              phase=phase, child_refit_updates=step, update_budget=UPDATES,
              child_refit_updates_total=step,
              child_refit_updates_this_run=0 if args.resume_checkpoint is not None else step,
              classifier_training_updates=0, wall_seconds=time.monotonic()-started, **extra))
    def stopped():
        authorized()
        assert not (module.ROOT / "USER_STOP").exists(), "User stop requested"
    def array(tensor):
        return tensor.detach().cpu().numpy()
    def save_render(path, package):
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = np.round(np.clip(array(package["rgb"]).transpose(1, 2, 0), 0, 1)*255).astype("uint8")
        Image.fromarray(pixels).save(path.with_suffix(".png"))
        np.savez_compressed(path.with_suffix(".npz"), **{key: array(value) for key, value in package.items()})
    status("waiting_GPU")
    with importlib.import_module("gpu_slot").exclusive():
        stopped(); status("loading")
        dataset = module.ROOT / "datasets" / CTX
        dataset_manifest = read(dataset / "manifest.json")
        assert dataset_manifest["context"] == CTX
        assert [frame["raw"] for frame in dataset_manifest["frames"]] == list(range(77, 126))
        index = args.raw-77
        frame = dataset_manifest["frames"][index]
        expected = contract["groups"]["v43_actor_10000"]["frames"][str(args.raw)]
        assert frame["raw"] == expected["raw"] == args.raw and expected["camera_index"] == index
        actor = module.Actor(np.load(dataset / "seeds.npz"))
        actor_checkpoint = torch.load(contract["actor_checkpoint_ref"], map_location="cuda")
        assert actor_checkpoint["step"] == 10000
        actor.load_state_dict(actor_checkpoint["actor"]); del actor_checkpoint
        actor.requires_grad_(False)
        actor_before = {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()}
        native = module.NativeMotion(Path(contract["static_checkpoint_ref"]), 30000)
        static_count = len(native.static._xyz)
        assert static_count == 918850 and len(actor.xyz) == 6241
        assert str(frame["frame_id"]) == str(expected["frame_id"]) == str(native.records[index]["img_name"])
        assert np.isclose(float(frame["model_time"]), float(expected["model_time"]), rtol=0, atol=1e-12)
        camera = native.camera(dict(camera_index=index, frame_id=native.records[index]["img_name"],
                                    camera_model_time=frame["model_time"]), {})
        with torch.no_grad():
            static = module.prepare_rasterization(camera, native.static, native.args)
            static["means2D"] = torch.zeros_like(static["means3D"])
            static["features"] = static["features"][:, :2]
            actor_kwargs = actor.kwargs(native, camera, frame)
            actor_kwargs["features"] = actor_kwargs["features"][:, :2]
            full = module.helpers.merge_rows(static, actor_kwargs)
            full = {key: value.detach() if torch.is_tensor(value) else value for key, value in full.items()}
            environment = native.env(camera.get_world_directions(False).permute(1, 2, 0)).permute(2, 0, 1).detach()
        full_count = len(full["means3D"])
        parent_index = torch.as_tensor(parents, device="cuda", dtype=torch.long)
        assert all(0 <= row < static_count for row in parents)
        for proposal in proposals:
            row, result = proposal["row"], proposal["result"]
            np.testing.assert_allclose(array(full["means3D"][row]), result["parent_mean"], rtol=0, atol=1e-7)
            covariance = covariance_from_quaternion_scale(array(full["rotations"][row]), array(full["scales"][row]))
            np.testing.assert_allclose(covariance, result["parent_covariance"], rtol=1e-9, atol=1e-12)
            assert math.isclose(float(full["opacities"][row, 0]), result["parent_opacity"], rel_tol=1e-7)
        children = []
        for proposal in proposals:
            for child in proposal["result"]["children"]:
                quaternion, scales = covariance_to_quaternion_scale(np.asarray(child["covariance"]))
                children.append(dict(child, parent_row=proposal["row"], quaternion_wxyz=quaternion.tolist(), scales=scales.tolist()))
        assert len(children) == 14
        source = torch.as_tensor([child["parent_row"] for child in children], dtype=torch.long, device="cuda")
        child_template = {key: value[source].detach().clone() if torch.is_tensor(value) else value for key, value in full.items()}
        child_template["means3D"] = torch.as_tensor([child["mean"] for child in children], dtype=torch.float32, device="cuda")
        child_template["means2D"] = torch.zeros_like(child_template["means3D"])
        child_template["rotations"] = torch.as_tensor([child["quaternion_wxyz"] for child in children], dtype=torch.float32, device="cuda")
        child_template["scales"] = torch.as_tensor([child["scales"] for child in children], dtype=torch.float32, device="cuda")
        z = torch.cat([child_template["means3D"], torch.ones_like(child_template["means3D"][:, :1])], 1) @ camera.world_view_transform[:, 2:3]
        child_template["features"] = torch.cat([z, torch.ones_like(z)], 1)
        initial_opacity = torch.tensor([[child["opacity_candidate"]] for child in children], dtype=torch.float32, device="cuda")
        assert bool(((initial_opacity > 0) & (initial_opacity < 1)).all())
        dc = torch.nn.Parameter(child_template["shs"][:, :1].clone())
        logits = torch.nn.Parameter(torch.logit(initial_opacity))
        assert bool(torch.isfinite(logits).all())
        initial_dc, initial_logits = dc.detach().clone(), logits.detach().clone()
        optimizer = torch.optim.Adam([dict(params=[dc], lr=DC_LR), dict(params=[logits], lr=OPACITY_LR)], eps=1e-8)
        frozen_geometry = {key: child_template[key].clone() for key in ("means3D", "rotations", "scales")}
        parent_removed = {**full, "opacities": full["opacities"].clone()}
        parent_removed["opacities"][parent_index] = 0
        contexts = {}
        for context in ("actor_present", "actor_removed"):
            original_opacity = full["opacities"].clone()
            background_opacity = parent_removed["opacities"].clone()
            if context == "actor_removed":
                original_opacity[static_count:] = 0
                background_opacity[static_count:] = 0
            contexts[context] = (dict(full, opacities=original_opacity), dict(full, opacities=background_opacity))
        def child_kwargs(dc_value, logit_value, side_removed=None):
            keep = torch.tensor([[child["side"] != side_removed] for child in children], device="cuda", dtype=torch.float32)
            return {**child_template, "shs": torch.cat([dc_value, child_template["shs"][:, 1:]], 1),
                    "opacities": logit_value.sigmoid()*keep}
        def split_kwargs(dc_value, logit_value, context="actor_present", side_removed=None):
            return module.helpers.merge_rows(contexts[context][1], child_kwargs(dc_value, logit_value, side_removed))
        def render(kwargs):
            output = module.helpers.raster(camera, native.args, native.static.active_sh_degree, kwargs, native.bg)
            return dict(rgb=output["rgb"]+(1-output["alpha"])*environment,
                        alpha=output["alpha"], gaussian_rgb=output["rgb"],
                        environment_rgb=(1-output["alpha"])*environment)
        def detached(package):
            return {key: value.detach() for key, value in package.items()}
        observed, body, safe, depth, _ = source_maps(native.source, frame, dataset,
            np.load(module.ROOT / "segformer_b0_attempt3" / CTX / f"{args.raw:03d}.npz"))
        segmentation = np.load(module.ROOT / "segformer_b0_attempt3" / CTX / f"{args.raw:03d}.npz")
        road = safe & (segmentation["labels"] == 1)
        scope = scope_module.build_scope(body, context=CTX, raw=args.raw)
        assert scope["status"] == "ok" and np.asarray(scope["allowed"]).shape == body.shape
        masks = dict(body=body, observed_safe_road=road, measured_safe_road=road & np.isfinite(depth),
                     allowed=scope["allowed"], outside_allowed=~scope["allowed"])
        maps = np.stack([body, road]).astype("float32")
        with torch.no_grad():
            references = {context: detached(render(pair[0])) for context, pair in contexts.items()}
            initialized = detached(render(split_kwargs(initial_dc, initial_logits)))
        frozen_reference = references["actor_present"]
        reference_pixels = np.round(np.clip(array(frozen_reference["rgb"]).transpose(1, 2, 0), 0, 1)*255).astype("uint8")
        reference_path = args.input_contract.parent / expected["baseline_file"]
        assert np.array_equal(reference_pixels, np.asarray(Image.open(reference_path).convert("RGB"))), "Frozen native baseline differs"
        probe_baseline = np.load(args.probe_dir / f"{args.raw:03d}" / "intact_evidence.npz")
        assert float(np.abs(array(frozen_reference["rgb"])-probe_baseline["rgb"]).max()) <= 2e-6
        probe_split = np.load(args.probe_dir / f"{args.raw:03d}" / "split_float.npz")
        assert np.array_equal(probe_split["parents"], np.asarray(parents)), "Smoke parent list/order differs"
        initialization_replay_error = float(np.abs(array(initialized["rgb"])-probe_split["rgb"]).max())
        assert initialization_replay_error <= 2e-6, "Reused initialization does not reproduce smoke RGB"
        initial_mse = float((initialized["rgb"]-frozen_reference["rgb"]).square().mean())
        loss_scale = 1/max(initial_mse, 1e-12)
        bindings = dict(input_contract_path=str(args.input_contract), input_contract_sha256=sha(args.input_contract),
            proposal_path=str(proposals_path), proposal_sha256=sha(proposals_path),
            smoke_summary_sha256=sha(args.probe_dir / "summary.json"), ownership_sha256=sha(ownership_path),
            actor_checkpoint_ref=contract["actor_checkpoint_ref"], actor_checkpoint_sha256=contract["actor_checkpoint_sha256"],
            static_namespace=contract["static_namespace"], static_checkpoint_ref=contract["static_checkpoint_ref"],
            static_source_files=[dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)) for path in native.files],
            dataset_manifest_sha256=sha(dataset / "manifest.json"), scope_sha256=sha(args.scope_file),
            scene_probe_code_sha256=sha(Path(__file__).with_name("run_scene1_probe.py")),
            refit_code_sha256=sha(Path(__file__)), raw=args.raw, camera_index=index,
            frame_id=str(frame["frame_id"]), model_time=float(frame["model_time"]))
        resume_state, resume_provenance = None, None
        if args.resume_checkpoint is not None:
            assert args.resume_checkpoint.is_file(), "Resume checkpoint is missing"
            resume_state = torch.load(args.resume_checkpoint, map_location="cuda", weights_only=False)
            previous_manifest = read(args.resume_checkpoint.parent / "manifest.json")
            assert resume_state["step"] == UPDATES and resume_state["update_budget"] == UPDATES
            assert resume_state["seed"] == SEED and resume_state["parents"] == parents
            assert resume_state["children_lineage"] == [child["child_id"] for child in children]
            assert previous_manifest["source_bindings"] == resume_state["source_bindings"]
            assert previous_manifest["seed"] == SEED and previous_manifest["update_budget"] == UPDATES
            assert previous_manifest["parents"] == parents
            assert previous_manifest["optimizer"] == dict(name="Adam", dc_lr=DC_LR, opacity_lr=OPACITY_LR, eps=1e-8)
            revision = validate_resume_bindings(resume_state["source_bindings"], bindings)
            assert all(torch.equal(frozen_geometry[key], resume_state["child_geometry"][key]) for key in frozen_geometry)
            assert torch.equal(child_template["shs"][:, 1:], resume_state["child_SH_rest"])
            assert resume_state["child_dc"].shape == dc.shape and resume_state["child_opacity_logits"].shape == logits.shape
            assert bool(torch.isfinite(resume_state["child_dc"]).all()) and bool(torch.isfinite(resume_state["child_opacity_logits"]).all())
            resume_provenance = dict(checkpoint=str(args.resume_checkpoint), checkpoint_sha256=sha(args.resume_checkpoint),
                previous_output=str(args.resume_checkpoint.parent), code_revision=revision,
                source_bindings_verified=True, failed_scene_preserved=True,
                training_updates_total=UPDATES, training_updates_this_run=0,
                reused_before_metrics=str(args.resume_checkpoint.parent / "before_metrics.json"))
        manifest = dict(task=TASK, context=CTX, experiment="fixed_split_radiometric_refit_200",
            seed=SEED, update_budget=UPDATES, parents=parents, parent_count=7, child_count=14,
            source_bindings=bindings, children=children, authorization_snapshot=auth,
            resume_acceptance_only=resume_state is not None, resume_provenance=resume_provenance,
            child_refit_updates_total=UPDATES, child_refit_updates_this_run=0 if resume_state is not None else UPDATES,
            optimize=["child_SH_DC", "child_opacity_logits"], frozen=["all_source_parameters", "child_mean", "child_covariance", "child_rotation", "child_scale", "child_SH_higher_order"],
            objective="uniform_same_frame_frozen_native_RGB_MSE_distillation",
            loss_constant_scale=loss_scale, initial_unscaled_RGB_MSE=initial_mse,
            optimizer=dict(name="Adam", dc_lr=DC_LR, opacity_lr=OPACITY_LR, eps=1e-8),
            observations_used_for_optimization=False, semantic_owner_GT=False, child_gt_vehicle_is_geometric_hypothesis=True,
            deletion_controls=["whole_parent_remove", "gt_target_hypothesis_child_remove", "leq_road_hypothesis_child_remove"],
            contexts=list(contexts), side_purity_denominator="body_plus_safe_observed_road_alpha_T",
            classifier_training_updates=0, exact_pixel_alpha_conservation=False, shadow_solution_claimed=False,
            maximum_scope="one frozen raw113, same 7 smoke parents, 14 children, exactly 200 updates")
        write(args.output / "manifest.json", manifest)
        np.savez_compressed(args.output / "domains.npz", **masks)
        for context, reference in references.items():
            save_render(args.output / "references" / context, reference)
        def compare(package, reference):
            return pixel_metrics(array(package["rgb"]), array(reference["rgb"]),
                                 array(package["alpha"]), array(reference["alpha"]), masks)
        def evaluate_stage(stage, dc_value, logit_value):
            stopped(); result = {}
            for context in contexts:
                kwargs = split_kwargs(dc_value, logit_value, context)
                native_output, weights, chunks = native_adjoint(module, native, camera, kwargs, maps)
                package = dict(rgb=native_output["rgb"]+(1-native_output["alpha"])*environment,
                               alpha=native_output["alpha"], gaussian_rgb=native_output["rgb"],
                               environment_rgb=(1-native_output["alpha"])*environment)
                child_weights = weights[full_count:]
                purity = contribution_purity(child_weights, children)
                np.savez_compressed(args.output / f"{stage}_{context}_child_contributions.npz",
                                    weights=child_weights, parent_rows=np.asarray(parents),
                                    child_parent_rows=np.asarray([child["parent_row"] for child in children]),
                                    child_sides=np.asarray([child["side"] for child in children]))
                save_render(args.output / stage / context / "both_retained", package)
                controls = {}
                for name, side in (("whole_parent_remove", None),
                                   ("gt_target_hypothesis_child_remove", "gt"),
                                   ("leq_road_hypothesis_child_remove", "leq")):
                    with torch.no_grad():
                        control = detached(render(contexts[context][1] if side is None else
                                                  split_kwargs(dc_value, logit_value, context, side)))
                    save_render(args.output / stage / context / name, control)
                    controls[name] = dict(relative_to_stage_both_retained=compare(control, package),
                                          relative_to_frozen_original=compare(control, references[context]))
                result[context] = dict(reconstruction=compare(package, references[context]),
                                      contribution_purity=purity, adjoint_chunks=chunks, controls=controls)
                del weights, native_output, kwargs
            write(args.output / f"{stage}_metrics.json", result)
            return result
        if resume_state is None:
            before = evaluate_stage("before", initial_dc, initial_logits)
        else:
            # 原 before 已完成；复用已绑定现场的评估，不重复优化，也不覆盖旧目录。
            before = read(args.resume_checkpoint.parent / "before_metrics.json")
            write(args.output / "before_metrics.json", before)
        def optimizer_equal(left, right):
            if torch.is_tensor(left):
                return torch.is_tensor(right) and left.shape == right.shape and torch.equal(left.cpu(), right.cpu())
            if isinstance(left, dict):
                return isinstance(right, dict) and set(left) == set(right) and all(optimizer_equal(left[key], right[key]) for key in left)
            if isinstance(left, (tuple, list)):
                return type(left) == type(right) and len(left) == len(right) and all(optimizer_equal(x, y) for x, y in zip(left, right))
            return left == right
        def save_checkpoint(step):
            torch.cuda.synchronize()
            path = args.output / f"checkpoint{step:06d}.pth"
            temporary = path.with_suffix(".tmp")
            torch.save(dict(step=step, child_dc=dc.detach().cpu(), child_opacity_logits=logits.detach().cpu(),
                child_geometry={key: value.cpu() for key, value in frozen_geometry.items()},
                child_SH_rest=child_template["shs"][:, 1:].detach().cpu(), optimizer=optimizer.state_dict(),
                python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(), seed=SEED, update_budget=UPDATES,
                parents=parents, children_lineage=[child["child_id"] for child in children], source_bindings=bindings), temporary)
            temporary.replace(path)
            return path
        if resume_state is None:
            save_checkpoint(0)
            initial_completed_updates = 0
            status("refitting", 0)
        else:
            dc = torch.nn.Parameter(resume_state["child_dc"].clone())
            logits = torch.nn.Parameter(resume_state["child_opacity_logits"].clone())
            optimizer = torch.optim.Adam([dict(params=[dc], lr=DC_LR), dict(params=[logits], lr=OPACITY_LR)], eps=1e-8)
            optimizer.load_state_dict(resume_state["optimizer"])
            random.setstate(resume_state["python_rng"]); np.random.set_state(resume_state["numpy_rng"])
            torch.set_rng_state(resume_state["torch_rng"].cpu())
            torch.cuda.set_rng_state_all([value.cpu() for value in resume_state["cuda_rng"]])
            initial_completed_updates = UPDATES
            status("restored_for_acceptance_only", UPDATES)
        step = initial_completed_updates
        completed_updates = initial_completed_updates
        with (args.output / "steps.jsonl").open("w", encoding="utf-8") as log:
            try:
                for step in range(initial_completed_updates+1, UPDATES+1):
                    stopped(); optimizer.zero_grad(set_to_none=True)
                    current = render(split_kwargs(dc, logits))
                    mse = (current["rgb"]-frozen_reference["rgb"]).square().mean()
                    loss = mse*loss_scale
                    assert bool(torch.isfinite(loss)), "Nonfinite refit loss"
                    loss.backward()
                    assert dc.grad is not None and logits.grad is not None
                    assert bool(torch.isfinite(dc.grad).all()) and bool(torch.isfinite(logits.grad).all())
                    optimizer.step()
                    completed_updates = step
                    assert bool(torch.isfinite(dc).all()) and bool(torch.isfinite(logits).all())
                    row = dict(step=step, unscaled_RGB_MSE=float(mse), normalized_loss=float(loss),
                               loss_evaluated_before_this_update=True,
                               child_min_opacity=float(logits.sigmoid().min()), child_max_opacity=float(logits.sigmoid().max()),
                               wall_seconds=time.monotonic()-started)
                    log.write(json.dumps(row)+"\n")
                    if step == 1 or step % 25 == 0:
                        log.flush(); status("refitting", step, last_loss=row)
                        print(json.dumps(row), flush=True)
                    del current, mse, loss
                    if step == 100:
                        save_checkpoint(step)
            except BaseException:
                save_checkpoint(completed_updates)
                status("interrupted_or_failed", completed_updates, attempted_step=step)
                raise
        assert completed_updates == UPDATES
        if resume_state is None:
            checkpoint_path = save_checkpoint(UPDATES)
        else:
            checkpoint_path = args.output / f"checkpoint{UPDATES:06d}.pth"
            shutil.copyfile(args.resume_checkpoint, checkpoint_path)
            assert sha(checkpoint_path) == resume_provenance["checkpoint_sha256"], "Copied resume checkpoint differs"
        with torch.no_grad():
            final_before_restore = detached(render(split_kwargs(dc, logits)))
        expected_optimizer = optimizer.state_dict()
        # 新参数与新 optimizer 恢复；不是向原参数原地拷贝后宣称完整恢复。
        restored = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
        assert restored["step"] == UPDATES and restored["parents"] == parents
        if resume_state is None:
            assert restored["source_bindings"] == bindings
        else:
            assert restored["source_bindings"] == resume_state["source_bindings"]
            validate_resume_bindings(restored["source_bindings"], bindings)
        restored_dc = torch.nn.Parameter(restored["child_dc"].clone())
        restored_logits = torch.nn.Parameter(restored["child_opacity_logits"].clone())
        restored_optimizer = torch.optim.Adam([dict(params=[restored_dc], lr=DC_LR),
                                               dict(params=[restored_logits], lr=OPACITY_LR)], eps=1e-8)
        restored_optimizer.load_state_dict(restored["optimizer"])
        random.setstate(restored["python_rng"]); np.random.set_state(restored["numpy_rng"])
        torch.set_rng_state(restored["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in restored["cuda_rng"]])
        with torch.no_grad():
            final_after_restore = detached(render(split_kwargs(restored_dc, restored_logits)))
        restore_checks = dict(dc=torch.equal(dc, restored_dc), opacity_logits=torch.equal(logits, restored_logits),
            optimizer=optimizer_equal(expected_optimizer, restored_optimizer.state_dict()),
            geometry=all(torch.equal(frozen_geometry[key], restored["child_geometry"][key]) for key in frozen_geometry),
            SH_rest=torch.equal(child_template["shs"][:, 1:], restored["child_SH_rest"]),
            python_rng=random.getstate() == restored["python_rng"],
            numpy_rng=numpy_rng_equal(np.random.get_state(), restored["numpy_rng"]),
            torch_rng=torch.equal(torch.get_rng_state(), restored["torch_rng"].cpu()),
            cuda_rng=all(torch.equal(x, y.cpu()) for x, y in zip(torch.cuda.get_rng_state_all(), restored["cuda_rng"])),
            RGB_exact=torch.equal(final_before_restore["rgb"], final_after_restore["rgb"]),
            alpha_exact=torch.equal(final_before_restore["alpha"], final_after_restore["alpha"]))
        restore_report = dict(checks=restore_checks, all_passed=all(restore_checks.values()), checkpoint=str(checkpoint_path),
                              RGB_max_abs_error=float((final_before_restore["rgb"]-final_after_restore["rgb"]).abs().max()),
                              optimizer_replayed_updates=0, formal_updates=UPDATES,
                              child_refit_updates_total=UPDATES,
                              child_refit_updates_this_run=UPDATES-initial_completed_updates,
                              resume_provenance=resume_provenance,
                              RGB_reference_origin="checkpoint200_reconstructed_this_run" if resume_state is not None else "live_end_of_200_updates")
        write(args.output / "restore_verification.json", restore_report)
        assert restore_report["all_passed"], "Saved full state or restored native RGB differs"
        after = evaluate_stage("after", restored_dc.detach(), restored_logits.detach())
        assert all(torch.equal(child_template[key], frozen_geometry[key]) for key in frozen_geometry)
        assert native.unchanged()
        assert all(torch.equal(actor_before[key], value.detach().cpu()) for key, value in actor.state_dict().items())
        separation = {}
        for context in contexts:
            old = before[context]["contribution_purity"]["hypothesized_side_match_fraction"]
            new = after[context]["contribution_purity"]["hypothesized_side_match_fraction"]
            delta = new-old if new is not None and old is not None else None
            separation[context] = dict(before=old, after=new, change=delta,
                numerical_improvement_threshold=1e-4,
                contribution_purity_improved=bool(delta > 1e-4) if delta is not None else None,
                observation="improved_two_domain_contribution_purity" if delta is not None and delta > 1e-4 else
                            "no_measurable_two_domain_contribution_purity_improvement",
                geometry_optimization_executed=False, geometric_separability_gain_proven=False)
        summary = dict(task=TASK, context=CTX, raw=args.raw, status="complete", seed=SEED,
            child_refit_updates=UPDATES, classifier_training_updates=0, original_model_training_updates=0,
            child_refit_updates_total=UPDATES, child_refit_updates_this_run=UPDATES-initial_completed_updates,
            resume_acceptance_only=resume_state is not None, resume_provenance=resume_provenance,
            parents=parents, parent_count=7, children=14, children_both_sides_retained=True,
            initialization_replay_RGB_max_abs_error=initialization_replay_error,
            restoration=restore_report, before=before, after=after, contribution_separation=separation,
            source_checkpoints_unchanged=True, child_geometry_frozen=True, semantic_ground_truth=False,
            gt_vehicle_owner_is_geometric_hypothesis=True, measured_masks_used_only_for_evaluation=True,
            refit_target="same_frame_frozen_original_native_RGB_no_GT", shadow_solution_claimed=False,
            pixel_alpha_exact_conservation=False, edit_benefit_proven=False, quality_accepted=False, human_verdict=None,
            checkpoint=str(checkpoint_path), source_bindings=bindings,
            wall_seconds=time.monotonic()-started, cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            failure_ledger_refs=["V82-F01"], failure_ledger_delta="none")
        write(args.output / "summary.json", summary)
        status("complete", UPDATES, checkpoint=str(checkpoint_path), source_checkpoints_unchanged=True)
        print(json.dumps(dict(status="complete", child_refit_updates=UPDATES,
                              child_refit_updates_total=UPDATES, child_refit_updates_this_run=UPDATES-initial_completed_updates,
                              contribution_separation=separation, wall_seconds=time.monotonic()-started)), flush=True)


if __name__ == "__main__":
    main()

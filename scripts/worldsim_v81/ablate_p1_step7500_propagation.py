"""P1 正式7500零更新传播消融：六个固定valid窗，逐窗普通/无跨帧配对。

普通支须逐像素重放正式7500全部25张原生PNG。干预仅跳过参考帧latent
汇集：逐帧可见VAE latent、自身可见coverage和零flow仍送入原官方_post_fuse。
完整RGB只在两支QUERY完成后用于内容/输出评分，绝不进入生成条件。
--check-only 仅做CPU文件与协议预检；7500未就绪时报告pending，不加载模型。
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch


REPO = Path("/root/autodl-tmp/motion_proj_v81")
RUN = Path("/root/autodl-tmp/runs/worldsim_v81/WS-V81-SEEN-TO-SCENE-P1-20261009/r1")
PHASE = RUN / "paper_bidirectional_m4"
RETAINED = PHASE / "checkpoint_retained_step007500/p1-checkpoint-007500.pt"
CHECKPOINT = PHASE / "train/p1-checkpoint-007500.pt"
OUTPUT = RUN / "propagation_ablation_step7500"
SEQUENCES = ("00f88c4f0a", "7e625db8c4", "ff6eb95840")
SIDES = (0.125, 0.33)
SEED, STEPS, STEP = 2026, 25, 7500


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    return parser.parse_args(argv)


def _gpu_idle():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                             "--format=csv,noheader"], text=True, capture_output=True,
                            check=True)
    pids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if pids:
        raise RuntimeError(f"GPU仍有计算进程，禁止启动消融: {pids}")


def _formal(phase: Path, step: int, seq: str, side: float):
    return phase / f"validation/step{step:06d}" / seq / f"side_{side:g}"


def preflight(args):
    """不import torch/PIL，也不创建目录。仅返回完整就绪的六窗元数据。"""
    phase = args.run / "paper_bidirectional_m4"
    retained = phase / "checkpoint_retained_step007500/p1-checkpoint-007500.pt"
    checkpoint = phase / "train/p1-checkpoint-007500.pt"
    expected = [(_formal(phase, STEP, seq, side),
                 _formal(phase, 5000, seq, side), seq, side)
                for seq in SEQUENCES for side in SIDES]
    needed = [args.repo, retained, checkpoint, args.run / "validation_selection.json"]
    needed += [path / "run.json" for new, old, _, _ in expected for path in (new, old)]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        return None, {"status": "pending_formal_7500", "missing": missing,
                      "gpu_started": False, "models_loaded": False}
    if args.output_dir.exists() or args.output_dir.resolve().is_relative_to(args.repo.resolve()):
        raise ValueError("输出必须是全新的仓库外目录")
    if not checkpoint.is_symlink() or checkpoint.resolve() != retained.resolve():
        raise ValueError("7500训练入口必须指向独立保留的完整断点")
    if retained.stat().st_size < 4_000_000_000:
        raise ValueError("7500完整断点大小异常")
    selected = json.loads((args.run / "validation_selection.json").read_text(encoding="utf-8"))
    if selected.get("sequence_ids") != list(SEQUENCES) or selected.get("test_tuning") is not False:
        raise ValueError("固定valid选择发生漂移")
    cases = []
    for new, old, seq, side in expected:
        formal = json.loads((new / "run.json").read_text(encoding="utf-8"))
        prior = json.loads((old / "run.json").read_text(encoding="utf-8"))
        fixed = {"status": "complete", "sequence_id": seq,
                 "side_ratio_each": side, "checkpoint_step": STEP,
                 "seed": SEED, "steps": STEPS, "num_frames": 25,
                 "mode": "paper-feedforward", "amp": "bf16",
                 "effective_amp": "bf16",
                 "propagation_protocol": "paper-bidirectional-m4",
                 "frame_selection": "first_25_sorted_start_0",
                 "generation_protocol": "local_first25_sorted_start0_diagnostic"}
        bad = {k: (formal.get(k), v) for k, v in fixed.items() if formal.get(k) != v}
        if bad or Path(formal.get("checkpoint", "")).resolve() != retained.resolve():
            raise ValueError(f"正式7500协议/断点不符 {seq}/{side}: {bad}")
        same = ("source_frames", "data_root", "reference_indices", "flow_pairs",
                "mask_pixels_left", "mask_pixels_right", "seed", "steps", "mode",
                "amp", "effective_amp", "propagation_protocol")
        drift = [k for k in same if formal.get(k) != prior.get(k)]
        if drift:
            raise ValueError(f"5000→7500固定输入漂移 {seq}/{side}: {drift}")
        frame_dir = new / "frames_pred"
        names = sorted(p.name for p in frame_dir.glob("*.png")) if frame_dir.is_dir() else []
        if names != [f"{i:05d}.png" for i in range(25)]:
            raise ValueError(f"正式7500原生PNG不完整 {seq}/{side}")
        cases.append((new, seq, side, formal))
    return cases, {"status": "check_only_passed", "cases": len(cases),
                   "checkpoint": str(retained), "checkpoint_bytes": retained.stat().st_size,
                   "formal_native_png": 150, "gpu_started": False, "models_loaded": False}


def _copy(tensor):
    return tensor.detach().cpu().clone()


def run_branch(components, pipeline, inference, propagation, images, visible,
               hole, plan, *, disable_cross_frame: bool):
    import torch
    from torch.nn import functional as F

    for module in (components.vae, components.image_encoder, components.raft,
                   components.fcnet, components.propagator, components.unet):
        module.to("cuda")
    trace = {}
    original_prepare = pipeline.prepare_latents
    original_flow = inference._flows_for_pairs
    original_encode = inference.encode_video
    original_combine = components.fcnet.combine_flow
    original_propagate = propagation.propagate_reference_latents
    original_fuse = components.propagator._post_fuse
    original_noise = inference._unet_noise

    def capture_prepare(*a, **kw):
        value = original_prepare(*a, **kw)
        trace["initial_gaussian"] = _copy(value)
        return value

    def capture_flow(*a, **kw):
        value = original_flow(*a, **kw)
        trace["raft_fw"], trace["raft_bw"] = map(_copy, value)
        return value

    def capture_encode(vae, pixels, **kw):
        value = original_encode(vae, pixels, **kw)
        trace["vae_input"] = _copy(pixels)
        trace["per_frame_visible_vae"] = _copy(value)
        return value

    def capture_combine(flows, predicted, masks):
        value = original_combine(flows, predicted, masks)
        trace["completed_fw"], trace["completed_bw"] = map(_copy, value)
        trace["fcnet_masks"] = _copy(masks)
        return value

    def capture_fuse(latent, masks, future, past, future_cov, past_cov,
                     future_flow, past_flow):
        trace["fuse_original"] = _copy(latent)
        trace["fuse_mask"] = _copy(masks)
        trace["future_source"] = _copy(future)
        trace["past_source"] = _copy(past)
        trace["future_coverage"] = _copy(future_cov)
        trace["past_coverage"] = _copy(past_cov)
        trace["future_flow"] = _copy(future_flow)
        trace["past_flow"] = _copy(past_flow)
        return original_fuse(latent, masks, future, past, future_cov, past_cov,
                             future_flow, past_flow)

    def routed_propagate(module, latent, fw, bw, masks, chosen_plan):
        if chosen_plan != plan:
            raise RuntimeError("传播计划漂移")
        if not disable_cross_frame:
            return original_propagate(module, latent, fw, bw, masks, chosen_plan)
        batch, frames, _, height, width = latent.shape
        mask = F.interpolate(masks.flatten(0, 1).float(),
                             size=(height, width), mode="nearest")
        mask = (mask.reshape(batch, frames, 1, height, width) != 0).float()
        own_coverage = 1 - mask
        zero_flow = latent.new_zeros((batch, frames, 2, height, width))
        # 与原传播的无source初值完全相同；后续仍调用固定官方细化/融合。
        return module._post_fuse(latent, mask, latent, latent, own_coverage,
                                 own_coverage, zero_flow, zero_flow)

    def capture_noise(comp, latents, condition, clip, ids, timestep, scheduler,
                      *, cfg, amp_dtype):
        if not cfg or amp_dtype != torch.bfloat16:
            raise RuntimeError("必须沿用正式bf16/B=2 CFG")
        if "condition" not in trace:
            trace["condition"] = _copy(condition)
            trace["clip"] = _copy(clip)
            trace["time_ids"] = _copy(ids)
            trace["first_timestep"] = _copy(timestep)
        return original_noise(comp, latents, condition, clip, ids, timestep,
                              scheduler, cfg=cfg, amp_dtype=amp_dtype)

    with torch.no_grad(), ExitStack() as stack:
        stack.enter_context(patch.object(pipeline, "prepare_latents", capture_prepare))
        stack.enter_context(patch.object(inference, "_flows_for_pairs", capture_flow))
        stack.enter_context(patch.object(inference, "encode_video", capture_encode))
        stack.enter_context(patch.object(components.fcnet, "combine_flow", capture_combine))
        stack.enter_context(patch.object(components.propagator, "_post_fuse", capture_fuse))
        stack.enter_context(patch.object(propagation, "propagate_reference_latents",
                                         routed_propagate))
        stack.enter_context(patch.object(inference, "_unet_noise", capture_noise))
        prediction = inference.generate(
            components, pipeline, images, visible, hole, list(plan.pairs),
            seed=SEED, steps=STEPS, mode="paper-feedforward", amp="bf16",
            propagation_protocol="paper-bidirectional-m4", refs=list(plan.refs))
    return prediction, trace


def paired_proof(ordinary, ablated):
    import torch
    shared = ("initial_gaussian", "raft_fw", "raft_bw", "vae_input",
              "per_frame_visible_vae", "completed_fw", "completed_bw",
              "fcnet_masks", "fuse_original", "fuse_mask", "clip", "time_ids",
              "first_timestep")
    proof = {key: torch.equal(ordinary[key], ablated[key]) for key in shared}
    own = ablated["fuse_original"]
    coverage = 1 - ablated["fuse_mask"]
    proof.update({
        "future_is_own_frame": torch.equal(ablated["future_source"], own),
        "past_is_own_frame": torch.equal(ablated["past_source"], own),
        "future_coverage_is_own": torch.equal(ablated["future_coverage"], coverage),
        "past_coverage_is_own": torch.equal(ablated["past_coverage"], coverage),
        "future_flow_zero": torch.count_nonzero(ablated["future_flow"]).item() == 0,
        "past_flow_zero": torch.count_nonzero(ablated["past_flow"]).item() == 0,
    })
    if not all(proof.values()):
        raise RuntimeError(f"传播单变量配对失败: {proof}")
    return proof


def require_finite(prediction, trace, branch):
    import torch
    if not torch.isfinite(prediction).all():
        raise RuntimeError(f"{branch}输出含非有限值")
    for name, value in trace.items():
        if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
            raise RuntimeError(f"{branch} trace/{name}含非有限值")


def save_paired_inputs(path, ordinary, ablated, proof):
    """保存每支真实条件接线；全分辨率RAFT/FCNet配对由proof逐值检查。"""
    import torch
    if path.exists():
        raise FileExistsError(path)
    fields = ("initial_gaussian", "vae_input", "per_frame_visible_vae",
              "fuse_original", "fuse_mask", "future_source", "past_source",
              "future_coverage", "past_coverage", "future_flow", "past_flow",
              "condition", "clip", "time_ids", "first_timestep")
    torch.save({
        "ordinary": {key: ordinary[key] for key in fields},
        "no_cross_frame": {key: ablated[key] for key in fields},
        "paired_exact_shared_inputs": proof,
        "full_resolution_flow_note": "RAFT与FCNet补全流两支逐值相等已由paired_exact_shared_inputs验证；为控制大小未重复存入此文件",
    }, path)


def write_json_atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def scores(ordinary, ablated, traces, targets, hole, components, inference):
    """所有QUERY完成后才读取完整GT；latent L1仅是内容代理。"""
    import numpy as np
    import torch
    from torch.nn import functional as F

    gt_pixels = torch.from_numpy(np.stack([np.asarray(im) for im in targets]).copy())
    gt_pixels = gt_pixels.permute(0, 3, 1, 2).unsqueeze(0).float().cuda() / 127.5 - 1
    components.vae.to("cuda")
    with torch.no_grad():
        gt_latent = inference.encode_video(components.vae, gt_pixels,
                                           scaled=False, sample=False).cpu()
    if not torch.isfinite(gt_latent).all():
        raise RuntimeError("GT评测VAE latent含非有限值")
    gt_pixels = gt_pixels.cpu()
    latent_mask = F.interpolate(hole.cpu().flatten(0, 1),
                                size=gt_latent.shape[-2:], mode="nearest")
    latent_mask = latent_mask.reshape(1, 25, 1, *gt_latent.shape[-2:])
    def masked_latent_l1(value):
        delta = (value - gt_latent).abs() * latent_mask
        return float(delta.sum() / (latent_mask.sum() * value.shape[2]))
    content = {key: masked_latent_l1(trace["condition"])
               for key, trace in traces.items()}
    own = traces["ordinary"]["fuse_original"]
    content["per_frame_visible_vae"] = masked_latent_l1(own)
    full = traces["ordinary"]
    support_gain = {}
    for direction in ("future", "past"):
        coverage = full[f"{direction}_coverage"]
        own_coverage = 1 - full["fuse_mask"]
        support_gain[direction] = float(((coverage - own_coverage) * latent_mask).sum()
                                        / latent_mask.sum())

    target = np.stack([np.asarray(im) for im in targets]).astype(np.float32)
    mask = hole[0, :, 0].cpu().numpy().astype(bool)
    def rgb_metrics(prediction):
        rgb = (prediction.permute(0, 2, 3, 1).numpy() * 255).round().astype(np.float32)
        spatial = float(np.abs(rgb - target)[mask].mean())
        delta = (rgb[1:] - rgb[:-1]) - (target[1:] - target[:-1])
        temporal = float(np.abs(delta)[mask[1:]].mean())
        return {"hole_rgb_mae_0_255": spatial,
                "hole_temporal_delta_mae_0_255": temporal}
    output = {"ordinary": rgb_metrics(ordinary),
              "no_cross_frame": rgb_metrics(ablated)}
    gain = {metric: output["no_cross_frame"][metric] - output["ordinary"][metric]
            for metric in output["ordinary"]}
    result = {"condition_gt_latent_hole_l1_proxy": content,
            "condition_proxy_gain_positive_is_better": content["no_cross_frame"] - content["ordinary"],
            "source_coverage_gain_in_hole": support_gain,
            "condition_full_vs_ablated_mean_abs": float((full["condition"] - traces["no_cross_frame"]["condition"]).abs().mean()),
            "final_output_gt": output, "final_output_gain_positive_is_better": gain,
            "final_output_full_vs_ablated_mean_abs": float((ordinary - ablated).abs().mean())}
    numeric = [*content.values(), *support_gain.values(),
               result["condition_proxy_gain_positive_is_better"],
               result["condition_full_vs_ablated_mean_abs"],
               result["final_output_full_vs_ablated_mean_abs"],
               *gain.values(), *(value for row in output.values() for value in row.values())]
    if not all(math.isfinite(value) for value in numeric):
        raise RuntimeError("GT评测含非有限统计量")
    return result


def save_branch(prediction, visible, edges, output):
    import numpy as np
    from PIL import Image
    from motion_proj.worldsim_v81 import infer_p1 as inference
    pred_dir, comp_dir = output / "frames_pred", output / "frames_comp"
    output.mkdir(parents=True, exist_ok=False)
    pred_dir.mkdir()
    comp_dir.mkdir()
    predicted, composite = [], []
    for index, frame in enumerate(prediction):
        arr = (frame.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        pred = Image.fromarray(arr)
        comp_arr = arr.copy()
        comp_arr[:, edges[0]:edges[1]] = np.asarray(visible[index])[:, edges[0]:edges[1]]
        comp = Image.fromarray(comp_arr)
        pred.save(pred_dir / f"{index:05d}.png")
        comp.save(comp_dir / f"{index:05d}.png")
        predicted.append(pred)
        composite.append(comp)
    inference._save_video(predicted, output / "pred.mp4")
    inference._save_video(composite, output / "comp.mp4")


def save_conditions(pipeline, components, traces, case_dir):
    """两支QUERY之后解码中间条件；不改变生成输入、随机数或权重。"""
    import numpy as np
    import torch
    from PIL import Image
    from motion_proj.worldsim_v81 import infer_p1 as inference

    sources = {
        "visible_vae": traces["ordinary"]["per_frame_visible_vae"],
        "condition_full": traces["ordinary"]["condition"],
        "condition_no_cross": traces["no_cross_frame"]["condition"],
    }
    if not torch.equal(traces["ordinary"]["per_frame_visible_vae"],
                       traces["no_cross_frame"]["per_frame_visible_vae"]):
        raise RuntimeError("中间展示的逐帧visible VAE在两支间不相同")
    scale = components.vae.config.scaling_factor
    components.vae.to("cuda")
    saved = {}
    with torch.no_grad():
        for name, unscaled in sources.items():
            scaled = unscaled.to("cuda") * scale
            decoded = pipeline.decode_latents(scaled, num_frames=25,
                                              decode_chunk_size=8)[0]
            frames = ((decoded.permute(1, 0, 2, 3) + 1) / 2).clamp(0, 1).cpu()
            if frames.shape != (25, 3, 256, 256) or not torch.isfinite(frames).all():
                raise RuntimeError(f"{name} latent解码尺寸或有限性异常: {tuple(frames.shape)}")
            images = [Image.fromarray((frame.permute(1, 2, 0).numpy() * 255)
                                      .round().astype(np.uint8)) for frame in frames]
            frame_dir = case_dir / f"frames_{name}"
            frame_dir.mkdir(exist_ok=False)
            for index, image in enumerate(images):
                image.save(frame_dir / f"{index:05d}.png")
            mp4 = case_dir / f"{name}.mp4"
            if mp4.exists():
                raise FileExistsError(mp4)
            inference._save_video(images, mp4)
            saved[name] = {"frames": str(frame_dir), "mp4": str(mp4)}
    return saved


def main():
    args = arguments()
    cases, check = preflight(args)
    if args.check_only or cases is None:
        print(json.dumps(check, ensure_ascii=False, indent=2), flush=True)
        if cases is None and not args.check_only:
            raise RuntimeError("正式7500六窗未齐，禁止启动GPU消融")
        return
    _gpu_idle()
    sys.path.insert(0, str(args.repo))
    import numpy as np
    from PIL import Image
    import torch
    from motion_proj.worldsim_v81 import infer_p1 as inference
    from motion_proj.worldsim_v81 import reference_propagation as propagation
    from motion_proj.worldsim_v81.model_bridge import load_components, load_trainable_state
    from motion_proj.worldsim_v81.train_p1 import FORMAT

    if not torch.cuda.is_available():
        raise RuntimeError("正式消融需要空闲CUDA，--check-only可纯CPU运行")
    state = torch.load(Path(check["checkpoint"]), map_location="cpu", weights_only=False)
    if (state.get("format") != FORMAT or state.get("step") != STEP
            or state.get("propagation_protocol") != "paper-bidirectional-m4"
            or state.get("args", {}).get("amp") != "bf16"):
        raise ValueError("断点内部不是正式7500模型/协议")
    components = load_components(device="cuda")
    load_trainable_state(components, state["models"])
    del state
    pipeline_type = inference._official_pipeline_class(inference.DEFAULT_EXTERNAL)
    pipeline = pipeline_type(
        vae=components.vae, image_encoder=components.image_encoder,
        unet=components.unet, scheduler=components.scheduler,
        feature_extractor=components.feature_extractor, fix_raft=components.raft,
        vo_flow_complete=components.fcnet, lat_bi_propagator=components.propagator)
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    report = {"status": "running", "kind": "step7500_zero_update_cross_frame_propagation_off",
              "checkpoint": check["checkpoint"], "checkpoint_step": STEP,
              "optimizer_updates": 0, "seed": SEED, "steps": STEPS,
              "amp": "bf16", "mode": "paper-feedforward", "cases": results,
              "changed_only": "跳过跨帧参考latent汇集；每帧visible VAE/self coverage/zero flow仍进入原_post_fuse",
              "gt_role": "完整GT仅在普通与无跨帧QUERY完成后编码/评分，不进入任何生成条件",
              "latent_decode_role": "三套unscaled中间条件乘VAE scaling_factor后解码，仅用于展示，不是QUERY",
              "interpretation_limit": "推理时移除传播是已训练权重的分布外干预；不等价于重训练论文ablation，latent L1只是内容代理",
              "formal_metrics_computed": False, "human_verdict": None}
    write_json_atomic(args.output_dir / "run.json", report)
    for formal_dir, seq, side, meta in cases:
        case_dir = args.output_dir / seq / f"side_{side:g}"
        case_dir.mkdir(parents=True, exist_ok=False)
        case = {"status": "running", "phase": "input_preparation",
                "sequence_id": seq, "side_ratio_each": side,
                "checkpoint_step": STEP, "checkpoint": check["checkpoint"],
                "seed": SEED, "steps": STEPS, "num_frames": 25,
                "mode": "paper-feedforward", "propagation_protocol": "paper-bidirectional-m4",
                "amp": "bf16", "optimizer_updates": 0, "human_verdict": None,
                "ordinary_formal_dir": str(formal_dir),
                "no_cross_frame_dir": str(case_dir / "no_cross_frame"),
                "branches": {"ordinary": str(case_dir / "ordinary"),
                             "no_cross_frame": str(case_dir / "no_cross_frame")},
                "paired_inputs": str(case_dir / "paired_inputs.pt")}
        write_json_atomic(case_dir / "run.json", case)
        try:
            paths, targets, images, edges, refs, _ = inference.load_sequence(
                Path(meta["data_root"]), seq, side)
            plan = propagation.build_reference_plan(len(paths), refs)
            if ([str(p) for p in paths] != meta["source_frames"]
                    or refs != meta["reference_indices"]
                    or [list(pair) for pair in plan.pairs] != meta["flow_pairs"]):
                raise RuntimeError(f"实时输入/参考图漂移 {seq}/{side}")
            case["source_frames"] = [str(p) for p in paths]
            case["reference_indices"] = refs
            case["flow_pairs"] = [list(x) for x in plan.pairs]
            rgb = np.stack([np.asarray(im) for im in images]).copy()
            visible = torch.from_numpy(rgb).permute(0, 3, 1, 2).unsqueeze(0).float().cuda() / 127.5 - 1
            hole = torch.zeros(1, 25, 1, 256, 256, device="cuda")
            hole[..., :edges[0]] = 1
            hole[..., edges[1]:] = 1
            visible *= 1 - hole
            case["phase"] = "query_ordinary_and_formal_replay"
            write_json_atomic(case_dir / "run.json", case)
            ordinary, full_trace = run_branch(components, pipeline, inference, propagation,
                                              images, visible, hole, plan,
                                              disable_cross_frame=False)
            require_finite(ordinary, full_trace, "ordinary")
            for index in range(25):
                with Image.open(formal_dir / "frames_pred" / f"{index:05d}.png") as source:
                    png = np.asarray(source)
                actual = (ordinary[index].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                if not np.array_equal(actual, png):
                    raise RuntimeError(f"普通支未精确重放正式7500: {seq}/{side}/f{index:02d}")
            case["ordinary_replays_formal_all25_exact"] = True
            case["phase"] = "query_no_cross_frame"
            write_json_atomic(case_dir / "run.json", case)
            ablated, no_cross_trace = run_branch(components, pipeline, inference, propagation,
                                                 images, visible, hole, plan,
                                                 disable_cross_frame=True)
            require_finite(ablated, no_cross_trace, "no_cross_frame")
            proof = paired_proof(full_trace, no_cross_trace)
            case["paired_exact_shared_inputs"] = proof
            case["phase"] = "save_queries_and_actual_trace"
            write_json_atomic(case_dir / "run.json", case)
            save_paired_inputs(case_dir / "paired_inputs.pt", full_trace, no_cross_trace, proof)
            save_branch(ordinary, images, edges, case_dir / "ordinary")
            save_branch(ablated, images, edges, case_dir / "no_cross_frame")
            case["status"] = "query_complete"
            case["phase"] = "decode_intermediate_conditions"
            write_json_atomic(case_dir / "run.json", case)
            traces = {"ordinary": full_trace, "no_cross_frame": no_cross_trace}
            case["intermediate_conditions"] = save_conditions(
                pipeline, components, traces, case_dir)
            case["phase"] = "gt_scoring_after_both_queries"
            write_json_atomic(case_dir / "run.json", case)
            case["measurement"] = scores(ordinary, ablated, traces, targets, hole,
                                         components, inference)
            case["status"] = "complete"
            case["phase"] = "complete"
            write_json_atomic(case_dir / "run.json", case)
            results.append(case)
            report["cases"] = results
            write_json_atomic(args.output_dir / "run.json", report)
            del visible, hole, ordinary, ablated, full_trace, no_cross_trace
            torch.cuda.empty_cache()
        except BaseException as error:
            case["status"] = "failed"
            case["error_type"] = type(error).__name__
            case["error"] = str(error)
            case["traceback"] = traceback.format_exc()
            write_json_atomic(case_dir / "run.json", case)
            report["status"] = "failed"
            report["failed_case"] = str(case_dir / "run.json")
            report["completed_cases"] = len(results)
            write_json_atomic(args.output_dir / "run.json", report)
            raise
        print(json.dumps({"event": "case_complete", "seq": seq, "side": side},
                         ensure_ascii=False), flush=True)
    report["status"] = "complete"
    report["elapsed_seconds"] = time.perf_counter() - started
    report["peak_cuda_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    write_json_atomic(args.output_dir / "run.json", report)
    print(json.dumps({"event": "propagation_ablation_complete",
                      "report": str(args.output_dir / "run.json")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

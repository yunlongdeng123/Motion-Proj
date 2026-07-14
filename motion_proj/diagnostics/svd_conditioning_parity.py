"""C0：逐项审计 Motion-Proj 与官方 SVD conditioning/generation 协议。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import types
from pathlib import Path
from typing import Any

import torch

from ..backbones import build_backbone
from ..backbones.svd_backbone import SVDBackbone
from ..config import config_fingerprint, load_config, save_resolved_config
from ..data import NuScenesFutureVideoDataset
from ..runtime.atomic import atomic_write_json, atomic_write_text
from ..runtime.experiment import JsonlMetrics, RunManifest, utc_now
from ..runtime.fingerprint import environment_fingerprint, git_state, sha256_json
from ..train.trainer import seed_everything


def _cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu").clone()


def _tensor_fingerprint(value: torch.Tensor) -> str:
    tensor = _cpu_tensor(value).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _base_model_fingerprint(pretrained: str) -> str:
    root = Path(pretrained)
    required = (
        root / "unet" / "config.json",
        root / "vae" / "config.json",
        root / "scheduler" / "scheduler_config.json",
        root / "image_encoder" / "config.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"C0 Base fingerprint 缺少模型配置: {missing}")
    return sha256_json({str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in required})


def tensor_difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    """返回可 JSON 化的张量差异，同时拒绝静默 shape/dtype 混淆。"""
    if tuple(left.shape) != tuple(right.shape):
        return {
            "shape_match": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "dtype_match": str(left.dtype) == str(right.dtype),
            "max_abs": math.inf,
            "rms": math.inf,
            "exact": False,
        }
    left_cpu, right_cpu = _cpu_tensor(left), _cpu_tensor(right)
    difference = left_cpu.float() - right_cpu.float()
    return {
        "shape_match": True,
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "dtype_match": str(left.dtype) == str(right.dtype),
        "max_abs": float(difference.abs().max()) if difference.numel() else 0.0,
        "rms": float(difference.square().mean().sqrt()) if difference.numel() else 0.0,
        "exact": bool(torch.equal(left_cpu, right_cpu)),
    }


def tensor_sequence_difference(left: list[torch.Tensor], right: list[torch.Tensor]) -> dict[str, Any]:
    if len(left) != len(right):
        return {"length_match": False, "left_length": len(left), "right_length": len(right), "per_step": []}
    per_step = [tensor_difference(first, second) for first, second in zip(left, right)]
    max_abs = max((float(row["max_abs"]) for row in per_step), default=0.0)
    rms = max((float(row["rms"]) for row in per_step), default=0.0)
    return {
        "length_match": True,
        "left_length": len(left),
        "right_length": len(right),
        "max_abs": max_abs,
        "max_rms": rms,
        "all_exact": all(bool(row["exact"]) for row in per_step),
        "per_step": per_step,
    }


class _PipelineTrace:
    """短时 monkey-patch 官方 pipeline，以记录而不改变任何采样算子。"""

    def __init__(self, pipe: Any):
        self.pipe = pipe
        self.values: dict[str, Any] = {
            "scheduler_timesteps": [],
            "scheduler_inputs": [],
            "scaled_model_inputs": [],
            "unet_inputs": [],
            "raw_model_outputs": [],
            "unconditional_raw_model_outputs": [],
            "conditional_raw_model_outputs": [],
            "cfg_outputs": [],
            "scheduler_step_outputs": [],
            "post_step_latents": [],
            "random_draws": [],
        }
        self._restore: list[tuple[Any, str, Any]] = []

    def _replace(self, owner: Any, name: str, replacement: Any) -> None:
        self._restore.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    def __enter__(self):
        pipe = self.pipe
        original_encode_image = pipe._encode_image
        original_encode_vae = pipe._encode_vae_image
        original_time_ids = pipe._get_add_time_ids
        original_prepare_latents = pipe.prepare_latents
        original_preprocess = pipe.video_processor.preprocess
        original_scale = pipe.scheduler.scale_model_input
        original_step = pipe.scheduler.step
        original_unet = pipe.unet.forward

        def preprocess(video_processor, *args, **kwargs):
            result = original_preprocess(*args, **kwargs)
            self.values["preprocessed_image"] = _cpu_tensor(result)
            return result

        def encode_image(pipeline, *args, **kwargs):
            result = original_encode_image(*args, **kwargs)
            self.values["image_embeds"] = _cpu_tensor(result)
            return result

        def encode_vae(pipeline, image, *args, **kwargs):
            self.values["noisy_condition_image"] = _cpu_tensor(image)
            result = original_encode_vae(image, *args, **kwargs)
            self.values["encoded_image_latents"] = _cpu_tensor(result)
            return result

        def time_ids(pipeline, *args, **kwargs):
            result = original_time_ids(*args, **kwargs)
            self.values["added_time_ids"] = _cpu_tensor(result)
            return result

        def prepare_latents(pipeline, *args, **kwargs):
            result = original_prepare_latents(*args, **kwargs)
            self.values["initial_video_latents"] = _cpu_tensor(result)
            return result

        def scale_model_input(scheduler, sample, timestep, *args, **kwargs):
            result = original_scale(sample, timestep, *args, **kwargs)
            self.values["scheduler_timesteps"].append(_cpu_tensor(torch.as_tensor(timestep)))
            self.values["scheduler_inputs"].append(_cpu_tensor(sample))
            self.values["scaled_model_inputs"].append(_cpu_tensor(result))
            return result

        def unet_forward(*args, **kwargs):
            result = original_unet(*args, **kwargs)
            sample = args[0] if args else kwargs["sample"]
            self.values["unet_inputs"].append(_cpu_tensor(sample))
            self.values["raw_model_outputs"].append(_cpu_tensor(result[0]))
            if result[0].shape[0] % 2 == 0:
                midpoint = result[0].shape[0] // 2
                self.values["unconditional_raw_model_outputs"].append(_cpu_tensor(result[0][:midpoint]))
                self.values["conditional_raw_model_outputs"].append(_cpu_tensor(result[0][midpoint:]))
            return result

        def scheduler_step(scheduler, model_output, timestep, sample, *args, **kwargs):
            result = original_step(model_output, timestep, sample, *args, **kwargs)
            self.values["cfg_outputs"].append(_cpu_tensor(model_output))
            self.values["scheduler_step_outputs"].append(_cpu_tensor(result.prev_sample))
            return result

        self._replace(pipe.video_processor, "preprocess", types.MethodType(preprocess, pipe.video_processor))
        self._replace(pipe, "_encode_image", types.MethodType(encode_image, pipe))
        self._replace(pipe, "_encode_vae_image", types.MethodType(encode_vae, pipe))
        self._replace(pipe, "_get_add_time_ids", types.MethodType(time_ids, pipe))
        self._replace(pipe, "prepare_latents", types.MethodType(prepare_latents, pipe))
        self._replace(pipe.scheduler, "scale_model_input", types.MethodType(scale_model_input, pipe.scheduler))
        self._replace(pipe.scheduler, "step", types.MethodType(scheduler_step, pipe.scheduler))
        self._replace(pipe.unet, "forward", unet_forward)

        # SVD pipeline 从 module global 调用 randn_tensor；记录每个 draw 才能审计 condition noise。
        import diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion as pipeline_module

        original_randn = pipeline_module.randn_tensor

        def randn_tensor(*args, **kwargs):
            result = original_randn(*args, **kwargs)
            self.values["random_draws"].append(_cpu_tensor(result))
            return result

        self._replace(pipeline_module, "randn_tensor", randn_tensor)
        return self

    def callback(self, _pipe: Any, _index: int, _timestep: Any, callback_kwargs: dict[str, Any]):
        self.values["post_step_latents"].append(_cpu_tensor(callback_kwargs["latents"]))
        return callback_kwargs

    def __exit__(self, exc_type, exc, traceback):
        for owner, name, original in reversed(self._restore):
            setattr(owner, name, original)


def _frames_to_tensor(frames: list[Any]) -> torch.Tensor:
    import numpy as np

    array = np.stack([np.asarray(frame) for frame in frames], axis=0)
    return torch.from_numpy(array).float().permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0


def _autocast(backbone: SVDBackbone):
    device_type = "cuda" if torch.cuda.is_available() and str(backbone.device).startswith("cuda") else "cpu"
    return torch.autocast(device_type=device_type, dtype=backbone.dtype)


def trace_official_pipeline(
    backbone: SVDBackbone,
    cond_frame: torch.Tensor,
    *,
    seed: int,
    num_frames: int,
    num_inference_steps: int,
    height: int,
    width: int,
) -> dict[str, Any]:
    """运行未经 wrapper 的官方 pipeline，并捕获逐步张量。"""
    pipe = backbone._generation_pipeline()
    pipe.set_progress_bar_config(disable=True)
    settings = backbone.generation_settings()
    trace = _PipelineTrace(pipe)
    generator = torch.Generator(device=backbone.device).manual_seed(int(seed))
    image = backbone._pipeline_image(cond_frame)
    with trace, _autocast(backbone):
        output = pipe(
            image,
            num_frames=int(num_frames),
            num_inference_steps=int(num_inference_steps),
            height=int(height),
            width=int(width),
            fps=int(settings["fps"]),
            motion_bucket_id=int(settings["motion_bucket_id"]),
            noise_aug_strength=float(settings["noise_aug_strength"]),
            min_guidance_scale=float(settings["min_guidance_scale"]),
            max_guidance_scale=float(settings["max_guidance_scale"]),
            decode_chunk_size=4,
            generator=generator,
            callback_on_step_end=trace.callback,
            output_type="pil",
        )
    if len(trace.values["random_draws"]) < 2:
        raise RuntimeError("官方 pipeline 未记录到 condition noise 与 initial latent 两次随机采样")
    trace.values["condition_noise"] = trace.values["random_draws"][0]
    trace.values["initial_latent_noise"] = trace.values["random_draws"][1]
    trace.values["image_latents"] = trace.values["encoded_image_latents"].unsqueeze(1).repeat(
        1, int(num_frames), 1, 1, 1,
    )
    trace.values["final_latent"] = trace.values["post_step_latents"][-1]
    trace.values["decoded_frames"] = _cpu_tensor(_frames_to_tensor(output.frames[0]))
    return trace.values


def trace_backbone_generation(
    backbone: SVDBackbone,
    cond_frame: torch.Tensor,
    *,
    seed: int,
    num_frames: int,
    num_inference_steps: int,
    height: int,
    width: int,
) -> dict[str, Any]:
    """按实际 `SVDBackbone.generate` wrapper 路径采样并记录相同字段。"""
    pipe = backbone._generation_pipeline()
    pipe.set_progress_bar_config(disable=True)
    trace = _PipelineTrace(pipe)
    generator = torch.Generator(device=backbone.device).manual_seed(int(seed))
    with trace:
        frames = backbone.generate(
            cond_frame,
            num_frames=int(num_frames),
            num_inference_steps=int(num_inference_steps),
            height=int(height),
            width=int(width),
            decode_chunk_size=4,
            generator=generator,
            callback_on_step_end=trace.callback,
        )
    if len(trace.values["random_draws"]) < 2:
        raise RuntimeError("backbone generation 未记录到 condition noise 与 initial latent 两次随机采样")
    trace.values["condition_noise"] = trace.values["random_draws"][0]
    trace.values["initial_latent_noise"] = trace.values["random_draws"][1]
    trace.values["image_latents"] = trace.values["encoded_image_latents"].unsqueeze(1).repeat(
        1, int(num_frames), 1, 1, 1,
    )
    trace.values["final_latent"] = trace.values["post_step_latents"][-1]
    trace.values["decoded_frames"] = _cpu_tensor(frames)
    return trace.values


def trace_candidate_conditioning(
    backbone: SVDBackbone,
    cond_frame: torch.Tensor,
    *,
    seed: int,
    num_frames: int,
    height: int,
    width: int,
) -> dict[str, Any]:
    """检查版本化候选 API 是否准确复现官方 condition 和 initial latent。"""
    pipe = backbone._generation_pipeline()
    generator = torch.Generator(device=backbone.device).manual_seed(int(seed))
    with _autocast(backbone):
        values = backbone.build_official_generation_conditioning(
            cond_frame,
            generator=generator,
            num_frames=int(num_frames),
            height=int(height),
            width=int(width),
        )
        pipe.scheduler.set_timesteps(25, device=backbone.device)
        values["initial_video_latents"] = pipe.prepare_latents(
            1,
            int(num_frames),
            int(pipe.unet.config.in_channels),
            int(height),
            int(width),
            values["image_embeds"].dtype,
            backbone.device,
            generator,
        )
    return {key: _cpu_tensor(value) if isinstance(value, torch.Tensor) else value for key, value in values.items()}


def compare_generation_traces(
    reference: dict[str, Any], candidate: dict[str, Any], *, raw_tolerance: float,
    final_latent_rms_tolerance: float, rgb_tolerance: float,
) -> dict[str, Any]:
    """比较官方和 wrapper trace，并返回第一处超阈值差异。"""
    exact_fields = (
        "added_time_ids", "condition_noise", "initial_video_latents", "scheduler_timesteps",
    )
    sequence_fields = (
        "scheduler_inputs", "scaled_model_inputs", "unet_inputs", "raw_model_outputs",
        "unconditional_raw_model_outputs", "conditional_raw_model_outputs",
        "cfg_outputs", "scheduler_step_outputs", "post_step_latents",
    )
    result: dict[str, Any] = {"scalar": {}, "sequence": {}}
    mismatches: list[str] = []
    for name in exact_fields:
        if name == "scheduler_timesteps":
            difference = tensor_sequence_difference(reference[name], candidate[name])
            result["sequence"][name] = difference
            if not difference.get("all_exact", False):
                mismatches.append(name)
        else:
            difference = tensor_difference(reference[name], candidate[name])
            result["scalar"][name] = difference
            if not difference["exact"]:
                mismatches.append(name)
    for name in sequence_fields:
        difference = tensor_sequence_difference(reference[name], candidate[name])
        result["sequence"][name] = difference
        tolerance = raw_tolerance
        if not difference.get("length_match", False) or float(difference.get("max_abs", math.inf)) > tolerance:
            mismatches.append(name)
    final_latent = tensor_difference(reference["final_latent"], candidate["final_latent"])
    decoded_frames = tensor_difference(reference["decoded_frames"], candidate["decoded_frames"])
    result["scalar"]["final_latent"] = final_latent
    result["scalar"]["decoded_frames"] = decoded_frames
    if float(final_latent["rms"]) > final_latent_rms_tolerance:
        mismatches.append("final_latent")
    if float(decoded_frames["max_abs"]) > rgb_tolerance:
        mismatches.append("decoded_frames")
    result["passed"] = not mismatches
    result["first_mismatch"] = mismatches[0] if mismatches else None
    result["mismatch_fields"] = mismatches
    return result


def compare_candidate_conditioning(
    official: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "condition_noise", "noisy_condition_image", "image_embeds", "image_latents",
        "added_time_ids", "initial_video_latents",
    )
    differences = {name: tensor_difference(official[name], candidate[name]) for name in fields}
    mismatches = [name for name, row in differences.items() if not row["exact"]]
    return {"passed": not mismatches, "first_mismatch": mismatches[0] if mismatches else None, "fields": differences}


def legacy_conditioning_comparison(
    backbone: SVDBackbone, cond_frame: torch.Tensor, official: dict[str, Any]
) -> dict[str, Any]:
    """量化旧 one-step condition 与 official conditional branch 的差异，而不修改旧 cache。"""
    legacy = backbone.build_conditioning({"cond_frame": cond_frame.unsqueeze(0)})
    official_embeds = official["image_embeds"]
    official_latents = official["image_latents"]
    official_ids = official["added_time_ids"]
    # CFG 拼接顺序为 uncond, cond；比较 legacy 与 conditional half。
    if official_embeds.shape[0] == 2:
        official_embeds = official_embeds[1:]
        official_latents = official_latents[1:]
        official_ids = official_ids[1:]
    differences = {
        "image_embeds": tensor_difference(official_embeds, legacy.data["image_embeds"]),
        "image_latents": tensor_difference(official_latents, legacy.data["image_latents"]),
        "added_time_ids": tensor_difference(official_ids, legacy.data["added_time_ids"]),
    }
    mismatches = [name for name, row in differences.items() if not row["exact"]]
    return {
        "matches_official_conditional_branch": not mismatches,
        "mismatch_fields": mismatches,
        "official_uses_condition_noise": True,
        "legacy_uses_condition_noise": False,
        "official_fps_time_id": int(official_ids[0, 0]),
        "legacy_fps_time_id": int(legacy.data["added_time_ids"][0, 0]),
        "fields": differences,
    }


def _write_diff_figure(comparison: dict[str, Any], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.parent.mkdir(parents=True, exist_ok=True)
    series = comparison["sequence"]["raw_model_outputs"].get("per_step", [])
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(range(len(series)), [float(row["max_abs"]) for row in series], marker="o")
    axis.set_xlabel("denoising step")
    axis.set_ylabel("official vs backbone raw-output max abs diff")
    axis.set_yscale("symlog", linthresh=1.0e-9)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def run_svd_conditioning_parity(cfg: Any) -> dict[str, Any]:
    c0 = cfg.c0
    if str(cfg.model.generation.protocol) != "svd_official_v1":
        raise ValueError("C0 只审计显式 model.generation.protocol=svd_official_v1")
    if bool(cfg.model.lora.enable):
        raise ValueError("C0 必须加载冻结 Base，model.lora.enable=false")
    if int(c0.num_inference_steps) != 25:
        raise ValueError("C0 预注册为 25 inference steps")
    git = git_state(".")
    if git.get("dirty"):
        raise RuntimeError("正式 C0 拒绝在 dirty worktree 上运行")
    work_dir = Path(cfg.work_dir)
    if work_dir.exists():
        raise RuntimeError(f"C0 run directory already exists: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=False)
    (work_dir / "figures").mkdir()
    dataset = NuScenesFutureVideoDataset(cfg.data)
    index = int(c0.condition_index)
    if not 0 <= index < len(dataset):
        raise IndexError(f"C0 condition_index 超出数据集范围: {index}")
    source = dataset[index]
    expected_sample_id = str(c0.get("expected_sample_id", ""))
    if expected_sample_id and str(source["sample_id"]) != expected_sample_id:
        raise RuntimeError(
            f"C0 sample mismatch: expected={expected_sample_id}, actual={source['sample_id']}"
        )
    condition_fingerprint = sha256_json({
        "sample_id": source["sample_id"],
        "shape": list(source["cond_frame"].shape),
        "tensor_fingerprint": _tensor_fingerprint(source["cond_frame"]),
    })
    config_fp = config_fingerprint(cfg)
    manifest = RunManifest(
        run_id=str(cfg.run_id),
        command=list(sys.argv),
        config_fingerprint=config_fp,
        cache_fingerprint="not-applicable:official-svd-conditioning-parity",
        seed=int(cfg.seed),
        git=git,
        environment=environment_fingerprint(),
        data_split=str(cfg.data.split),
    )
    manifest_data = manifest.__dict__ | {
        "task_id": str(c0.task_id),
        "condition_index": index,
        "sample_id": str(source["sample_id"]),
        "condition_fingerprint": condition_fingerprint,
        "base_model_fingerprint": _base_model_fingerprint(str(cfg.model.pretrained)),
        "uses_future_gt_ego": False,
        "uses_future_gt_track": False,
    }
    atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
    save_resolved_config(cfg, str(work_dir / "resolved.yaml"))
    metrics = JsonlMetrics(str(work_dir / "metrics.jsonl"))
    try:
        seed_everything(int(cfg.seed), deterministic=bool(cfg.train.deterministic))
        backbone = build_backbone(cfg.model, load=True, device=str(cfg.device))
        if not isinstance(backbone, SVDBackbone):
            raise TypeError("C0 当前只支持 SVDBackbone")
        backbone.unet.eval()
        backbone.vae.eval()
        backbone.image_encoder.eval()
        manifest_data["generation_protocol"] = backbone.generation_protocol_metadata()
        atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
        condition = source["cond_frame"].to(cfg.device)
        common = {
            "seed": int(c0.generation_seed),
            "num_frames": int(c0.num_frames),
            "num_inference_steps": int(c0.num_inference_steps),
            "height": int(cfg.data.height),
            "width": int(cfg.data.width),
        }
        official = trace_official_pipeline(backbone, condition, **common)
        current = trace_backbone_generation(backbone, condition, **common)
        candidate = trace_candidate_conditioning(
            backbone, condition, seed=int(c0.generation_seed), num_frames=int(c0.num_frames),
            height=int(cfg.data.height), width=int(cfg.data.width),
        )
        rerun = trace_backbone_generation(backbone, condition, **common)
        parity = compare_generation_traces(
            official, current,
            raw_tolerance=float(c0.max_raw_output_error),
            final_latent_rms_tolerance=float(c0.max_final_latent_rms),
            rgb_tolerance=float(c0.max_rgb_abs_error),
        )
        candidate_parity = compare_candidate_conditioning(official, candidate)
        rerun_parity = compare_generation_traces(
            current, rerun,
            raw_tolerance=float(c0.max_rerun_error),
            final_latent_rms_tolerance=float(c0.max_rerun_error),
            rgb_tolerance=float(c0.max_rerun_error),
        )
        legacy = legacy_conditioning_comparison(backbone, condition, official)
        decision = {
            "generation_parity_pass": bool(parity["passed"]),
            "candidate_conditioning_pass": bool(candidate_parity["passed"]),
            "rerun_exact_pass": bool(rerun_parity["passed"]),
            "legacy_one_step_conditioning_matches_official": bool(legacy["matches_official_conditional_branch"]),
        }
        decision["gate_status"] = (
            "pass_with_legacy_one_step_mismatch"
            if decision["generation_parity_pass"] and decision["candidate_conditioning_pass"]
            and decision["rerun_exact_pass"] and not decision["legacy_one_step_conditioning_matches_official"]
            else "pass"
            if all(decision.values())
            else "fail"
        )
        tensor_diffs = {
            "official_vs_backbone_generation": parity,
            "official_vs_versioned_candidate_conditioning": candidate_parity,
            "backbone_generation_rerun": rerun_parity,
            "legacy_one_step_conditioning_vs_official": legacy,
        }
        atomic_write_json(str(work_dir / "tensor_diffs.json"), tensor_diffs)
        _write_diff_figure(parity, work_dir / "figures" / "raw_output_parity.png")
        metrics.append(0, {
            "task_id": str(c0.task_id), "sample_id": str(source["sample_id"]),
            "generation_parity_pass": decision["generation_parity_pass"],
            "candidate_conditioning_pass": decision["candidate_conditioning_pass"],
            "rerun_exact_pass": decision["rerun_exact_pass"],
            "legacy_one_step_conditioning_matches_official": decision["legacy_one_step_conditioning_matches_official"],
            "official_backbone_first_mismatch": parity["first_mismatch"],
            "candidate_first_mismatch": candidate_parity["first_mismatch"],
            "rerun_first_mismatch": rerun_parity["first_mismatch"],
        })
        summary = {
            "status": "completed",
            "task_id": str(c0.task_id),
            "sample_id": str(source["sample_id"]),
            "condition_index": index,
            "condition_fingerprint": condition_fingerprint,
            "generation_protocol": manifest_data["generation_protocol"],
            "decision": decision,
            "old_cache_impact": (
                "V5 Base generation provenance remains valid only as an unversioned official-pipeline delegate; "
                "its stored legacy one-step conditioning does not match official fps/noise/CFG conditioning and "
                "must not support new one-step-to-rollout transfer claims without a versioned rebuild."
            ),
            "experiment_fingerprint": sha256_json({
                "config": config_fp, "condition": condition_fingerprint,
                "decision": decision, "tensor_diffs": tensor_diffs,
            }),
        }
        atomic_write_json(str(work_dir / "summary.json"), summary)
        atomic_write_text(str(work_dir / "COMPLETE"), sha256_json(summary) + "\n")
        manifest_data.update({"status": "completed", "ended_at": utc_now(), "exit_reason": decision["gate_status"]})
        atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
        return summary
    except Exception as exc:
        failure = {"status": "failed", "task_id": str(c0.task_id), "error": repr(exc)}
        atomic_write_json(str(work_dir / "summary.json"), failure)
        manifest_data.update({"status": "failed", "ended_at": utc_now(), "exit_reason": repr(exc)})
        atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    result = run_svd_conditioning_parity(load_config(args.config, list(args.overrides)))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

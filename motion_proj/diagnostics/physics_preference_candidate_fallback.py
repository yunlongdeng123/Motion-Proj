"""PA2-CAND-03D：唯一一次 8-condition earlier-fork candidate fallback。"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..backbones import build_backbone
from ..backbones.svd_backbone import SVDBackbone
from ..config import config_fingerprint, load_config, save_resolved_config
from ..data.physics_dpo_schema import validate_candidates, validate_conditions
from ..preference.calibration import PRIMARY_COMPONENTS
from ..preference.common_support import build_common_support
from ..preference.paired_tracks import PairModeRAFTTracker
from ..preference.residual_motion import compute_motion_component_evidence
from ..preference.selective_order import (
    build_condition_partial_order,
    quality_comparability,
    video_quality_metrics,
)
from ..runtime.atomic import atomic_write_json, atomic_write_text
from ..runtime.experiment import JsonlMetrics, RunManifest, utc_now
from ..runtime.fingerprint import environment_fingerprint, file_fingerprint, git_state, sha256_json
from ..train.trainer import seed_everything
from ..utils.io import write_video
from .physics_dpo_branch import (
    _first_frame_metrics,
    _generate_condition_group,
    _make_panel,
    _sync_cuda,
    resolve_fork_step,
)
from .physics_dpo_horizon import (
    _dataset_for_horizon,
    _json_line,
    _load_condition_frame,
    _load_scene_split,
    _make_condition_record,
    _tensor_fingerprint,
    fingerprint_denoising_trace,
    select_profile_conditions,
)
from .physics_preference_reaudit import (
    _apply_holm,
    _bootstrap_context,
    _context_quality,
    _decide_context,
    _edge_id,
)
from .svd_conditioning_parity import _base_model_fingerprint


REVIEW_VERDICTS = frozenset({"same_scene", "different_composition", "invalid", "uncertain"})


class CandidateFallbackError(RuntimeError):
    """唯一 candidate fallback 的 provenance、生成或 review 门禁失败。"""


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise CandidateFallbackError(f"缺少 {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CandidateFallbackError(f"{label} 必须是 object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise CandidateFallbackError(f"缺少 {label}: {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise CandidateFallbackError(f"{label} row 必须是 object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_write_text(
        str(path),
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows),
    )


def _validated_upo_source(fallback: Any) -> dict[str, Any]:
    root = Path(str(fallback.upo_run))
    summary = _read_json(root / "summary.json", label="PA2-UPO summary")
    marker = root / "COMPLETE"
    if (
        str(summary.get("run_id")) != str(fallback.expected_upo_run_id)
        or str(summary.get("status")) != "blocked_candidate_yield"
        or str(summary.get("config_fingerprint")) != str(fallback.expected_upo_config_fingerprint)
        or not bool(summary.get("oracle_gates_pass"))
        or not isinstance(summary.get("checks"), Mapping)
        or not all(bool(value) for value in summary["checks"].values())
        or not marker.is_file()
        or marker.read_text(encoding="utf-8").strip() != sha256_json(summary)
        or bool(summary.get("uses_future_gt"))
    ):
        raise CandidateFallbackError("PA2-UPO v2 未精确解锁唯一 candidate fallback")
    prospective = summary.get("prospective_counts")
    if not isinstance(prospective, Mapping) or int(prospective.get("strict", -1)) >= int(
        fallback.trigger_strict_upper_exclusive
    ):
        raise CandidateFallbackError("PA2-UPO strict yield 未触发预注册 fallback")
    return {
        "path": str(root),
        "summary": summary,
        "summary_sha256": file_fingerprint(str(root / "summary.json")),
        "resolved_sha256": file_fingerprint(str(root / "resolved.yaml")),
        "complete": marker.read_text(encoding="utf-8").strip(),
    }


def _validate_fallback_config(cfg: Any) -> dict[str, Any]:
    branch = cfg.branch
    fallback = cfg.fallback
    if str(fallback.task_id) != "PA2-CAND-03D":
        raise CandidateFallbackError("fallback task_id 必须为 PA2-CAND-03D")
    if int(fallback.condition_count) != 8 or int(fallback.condition_offset) != 120:
        raise CandidateFallbackError("唯一 fallback 必须固定为 condition [120,128)")
    if int(branch.sibling_count) != 4 or int(branch.num_inference_steps) != 25:
        raise CandidateFallbackError("fallback 必须为 4 siblings / 25 steps")
    if not abs(float(branch.fork_fraction) - 0.4) <= 1.0e-12:
        raise CandidateFallbackError("fallback fork_fraction 必须为 0.4")
    if not abs(float(branch.strength_rho) - 0.04) <= 1.0e-12:
        raise CandidateFallbackError("fallback strength_rho 必须为 0.04")
    if str(branch.family) != "common_prefix" or bool(cfg.model.lora.enable):
        raise CandidateFallbackError("fallback 只允许 frozen Base common-prefix generation")
    if int(cfg.data.num_frames) != 14 or int(cfg.model.num_frames) != 14:
        raise CandidateFallbackError("fallback 必须为 14 frames")
    if str(cfg.model.generation.protocol) != "svd_official_v1":
        raise CandidateFallbackError("fallback 必须复用 official SVD generation")
    resolve_fork_step(25, 0.4)
    return _validated_upo_source(fallback)


def _select_new_conditions(
    split: Mapping[str, Any],
    *,
    partition: str,
    offset: int,
    count: int,
    required_start_index: int,
    used_scene_tokens: set[str],
) -> list[dict[str, Any]]:
    selected = select_profile_conditions(
        split,
        partition=partition,
        condition_count=offset + count,
        required_start_index=required_start_index,
    )[offset : offset + count]
    if len(selected) != count:
        raise CandidateFallbackError("冻结 scene 排序不足以满足 fallback offset/count")
    scenes = {str(row["scene_token"]) for row in selected}
    if len(scenes) != count or scenes & used_scene_tokens:
        raise CandidateFallbackError("fallback scene 与旧 120 conditions 重叠")
    return [dict(row) for row in selected]


def _review_prompt(run_id: str) -> str:
    return f"""# PA2-CAND-03D earlier-fork 结构盲审提示词

## 1. 评测目的与非目标

本轮只判断 fork=0.4 的 8 个新 condition 是否仍保持同一驾驶场景、主体身份和布局，决定
SVD common-prefix fallback 是否具有结构合法性。它**不评物理 winner**、不比较哪列运动更平滑、
不推断机器 oracle 对错，也不选择训练样本。

## 2. 盲法与禁止读取的信息

- 只观看 `{run_id}/panels/*.mp4`；每个 case 有 A–E 五列，包含同一 condition 的 Base 与四条 sibling，列顺序已固定随机化。
- 禁止读取 `review_cases.private.json`、`candidate_manifest.jsonl`、`oracle_graphs.jsonl`、
  `machine_summary.json`、`resolved.yaml`、trace、seed、metric 或 candidate ID。
- 不要根据列位置猜 Base；不要把“运动幅度更小”当作结构合法。

## 3. 素材范围与观看方式

- 共 8 个 case，每个 14 帧、7 FPS；先完整播放，再逐帧查看首帧、中段和末帧。
- 重点观察道路拓扑、车道线/护栏、主要车辆与行人身份、相机视角、物体相对布局。
- 色彩或轻微纹理差异本身不等于构图变化，但若造成主体消失、身份切换或几何破裂，应判 invalid。

## 4. Verdict 定义与优先级

按以下优先级每 case 只填一个 verdict：

1. `invalid`：任一列不可解码、黑屏/严重过曝、灾难性闪烁、主体身份崩溃、几何撕裂到无法判断同场景。
2. `different_composition`：素材可看，但至少一列发生道路布局、视角、主体身份/数量或关键相对位置的明显改变，已不是同一场景的合理未来分支。
3. `same_scene`：五列保持同一首帧条件、道路布局、视角和主要主体身份；后续轨迹/纹理可不同，但仍是同一场景的可接受未来。
4. `uncertain`：看完并逐帧检查后，仍无法在 `same_scene` 与前两种失败间稳定判断。

边界例：车辆轻微尺度漂移但身份/位置连续可判 `same_scene`；车辆突然变成另一车型、道路分叉/车道数量变化判
`different_composition`；严重融化或多帧不可辨认判 `invalid`。不要因某列运动较慢、加速度较低或更模糊就选它为 winner。

## 5. JSONL 填写格式

复制 `reviews.template.jsonl` 为 `reviews.jsonl`，保留 8 行与 case_id。每行：

```json
{{"case_id":"pa2-cand-review-00","verdict":"same_scene","failure_reasons":[],"reviewer":"你的名字","notes":"简短可审计说明"}}
```

`failure_reasons` 可选值：`layout_change`、`identity_change`、`camera_change`、`geometry_break`、
`flicker_or_exposure`、`unreadable`；`same_scene` 时应为空。不得填写 `pending`，不得新增/删除 case。

## 6. 聚合阈值

- 必须完成 8/8；
- `same_scene >= 7/8`；
- `different_composition + invalid = 0`；
- 最多允许 1 个 `uncertain`。

任何结构失败都拒绝 SVD sibling route；通过也只解锁冻结 oracle 的后续 yield/strict-precision 审计，
不会直接训练或自动转 preference label。

## 7. 完成后的精确命令

```bash
cd /root/autodl-tmp/motion_proj
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/motionproj
export PYTHONPATH=.
python -m motion_proj.diagnostics.physics_preference_candidate_fallback \\
  --config configs/diagnostics/physics_preference_candidate_fallback.yaml \\
  --aggregate-only
```

聚合只读取人工 JSONL 与已冻结 machine artifacts，不加载模型、不重算候选、不代填 verdict。
"""


def _write_review_materials(
    *,
    work_dir: Path,
    conditions: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    frames_by_candidate: Mapping[str, torch.Tensor],
    fps: int,
    seed: int,
) -> dict[str, Any]:
    panel_dir = work_dir / "panels"
    panel_dir.mkdir(exist_ok=False)
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for raw in candidates:
        row = dict(raw)
        by_condition.setdefault(str(row["condition_id"]), []).append(row)
    public_cases = []
    private_cases = []
    templates = []
    for index, condition in enumerate(sorted(conditions, key=lambda row: str(row["condition_id"]))):
        condition_id = str(condition["condition_id"])
        rows = by_condition[condition_id]
        base = [row for row in rows if str(row["candidate_role"]) == "base_guard"]
        siblings = [row for row in rows if str(row["candidate_role"]) == "sibling"]
        if len(base) != 1 or len(siblings) != 4:
            raise CandidateFallbackError("review condition 必须恰含 Base + 4 siblings")
        candidate_ids = [str(base[0]["candidate_id"]), *sorted(str(row["candidate_id"]) for row in siblings)]
        random.Random(seed + index).shuffle(candidate_ids)
        labels = [chr(ord("A") + offset) for offset in range(5)]
        panel = _make_panel([frames_by_candidate[candidate_id] for candidate_id in candidate_ids], labels)
        case_id = f"pa2-cand-review-{index:02d}"
        panel_path = panel_dir / f"{case_id}.mp4"
        write_video(panel, str(panel_path), fps=fps)
        public_cases.append({
            "case_id": case_id,
            "panel_path": str(panel_path.relative_to(work_dir)),
            "blind_columns": labels,
            "rubric": "五列是否保持同一驾驶场景布局、相机视角和主要主体身份？",
        })
        private_cases.append({
            "case_id": case_id,
            "condition_id": condition_id,
            "blind_mapping": dict(zip(labels, candidate_ids)),
        })
        templates.append({
            "case_id": case_id,
            "verdict": "pending",
            "failure_reasons": [],
            "reviewer": "",
            "notes": "",
        })
    atomic_write_json(str(work_dir / "review_cases.json"), public_cases)
    atomic_write_json(str(work_dir / "review_cases.private.json"), private_cases)
    _write_jsonl(work_dir / "reviews.template.jsonl", templates)
    prompt = _review_prompt(str(work_dir))
    atomic_write_text(str(work_dir / "REVIEW_PROMPT.md"), prompt)
    return {"case_count": len(public_cases), "panel_dir": str(panel_dir), "prompt": str(work_dir / "REVIEW_PROMPT.md")}


def _review_summary(work_dir: Path, review_cfg: Mapping[str, Any]) -> dict[str, Any]:
    public = json.loads((work_dir / "review_cases.json").read_text(encoding="utf-8"))
    if not isinstance(public, list):
        raise CandidateFallbackError("review_cases.json 必须是 list")
    expected = {str(row["case_id"]) for row in public}
    reviews = _read_jsonl(work_dir / "reviews.jsonl", label="human reviews")
    by_id = {}
    allowed_reasons = {
        "layout_change", "identity_change", "camera_change", "geometry_break",
        "flicker_or_exposure", "unreadable",
    }
    for row in reviews:
        case_id = str(row.get("case_id", ""))
        verdict = str(row.get("verdict", ""))
        if case_id not in expected or case_id in by_id:
            raise CandidateFallbackError(f"review case_id 未知或重复: {case_id}")
        if verdict not in REVIEW_VERDICTS:
            raise CandidateFallbackError(f"review verdict 非法: {verdict}")
        reasons = row.get("failure_reasons")
        if not isinstance(reasons, list) or any(str(reason) not in allowed_reasons for reason in reasons):
            raise CandidateFallbackError("failure_reasons 非法")
        if verdict == "same_scene" and reasons:
            raise CandidateFallbackError("same_scene 不得填写 failure_reasons")
        if not str(row.get("reviewer", "")).strip():
            raise CandidateFallbackError("reviewer 不能为空")
        by_id[case_id] = row
    same = sum(str(row["verdict"]) == "same_scene" for row in by_id.values())
    bad = sum(str(row["verdict"]) in {"different_composition", "invalid"} for row in by_id.values())
    uncertain = sum(str(row["verdict"]) == "uncertain" for row in by_id.values())
    complete = len(by_id) == int(review_cfg["required_cases"]) == len(expected)
    passed = bool(
        complete
        and same >= int(review_cfg["minimum_same_scene"])
        and bad <= int(review_cfg["maximum_bad_cases"])
        and uncertain <= int(review_cfg["maximum_uncertain"])
    )
    return {
        "required_cases": int(review_cfg["required_cases"]),
        "completed_cases": len(by_id),
        "same_scene": same,
        "different_or_invalid": bad,
        "uncertain": uncertain,
        "pass": passed,
        "status": "pass" if passed else "awaiting_reviews" if not complete else "rejected",
    }


def _clean_markers(work_dir: Path) -> None:
    for name in ("COMPLETE", "REJECTED", "FAILED", "awaiting_reviews", "MACHINE_COMPLETE"):
        path = work_dir / name
        if path.exists():
            path.unlink()


def aggregate_candidate_fallback_reviews(cfg: Any) -> dict[str, Any]:
    work_dir = Path(str(cfg.work_dir))
    machine = _read_json(work_dir / "machine_summary.json", label="machine summary")
    if str(machine.get("status")) != "awaiting_reviews" or not bool(machine.get("machine", {}).get("machine_pass")):
        raise CandidateFallbackError("只有 machine-pass awaiting_reviews run 可聚合")
    review = _review_summary(work_dir, cfg.fallback.review)
    if review["status"] == "pass":
        status = "done"
        next_gate = "PA2-CAND oracle yield decision"
    elif review["status"] == "rejected":
        status = "rejected"
        next_gate = "reject SVD common-prefix sibling route"
    else:
        status = "awaiting_reviews"
        next_gate = "complete 8-case human structure review"
    summary = machine | {"status": status, "human_review": review, "next_gate": next_gate}
    atomic_write_json(str(work_dir / "summary.json"), summary)
    _clean_markers(work_dir)
    marker = "COMPLETE" if status == "done" else "REJECTED" if status == "rejected" else "awaiting_reviews"
    atomic_write_text(str(work_dir / marker), sha256_json(summary) + "\n")
    if status == "awaiting_reviews":
        atomic_write_text(str(work_dir / "MACHINE_COMPLETE"), sha256_json(machine) + "\n")
    manifest = _read_json(work_dir / "manifest.json", label="manifest")
    manifest.update({"status": status, "ended_at": utc_now(), "exit_reason": next_gate})
    atomic_write_json(str(work_dir / "manifest.json"), manifest)
    return summary


def preflight_candidate_fallback(cfg: Any) -> dict[str, Any]:
    result = {
        "task_id": str(cfg.fallback.task_id), "status": "ready", "blockers": [],
        "uses_gpu": True, "uses_future_gt": False, "training": False,
    }
    try:
        upo = _validate_fallback_config(cfg)
        split, provenance = _load_scene_split(cfg.branch)
        old_conditions = _read_jsonl(Path(str(cfg.fallback.old_pair_run)) / "conditions.jsonl", label="old conditions")
        used_scenes = {str(row["scene_token"]) for row in old_conditions}
        selected = _select_new_conditions(
            split,
            partition=str(cfg.branch.condition_partition),
            offset=int(cfg.fallback.condition_offset),
            count=int(cfg.fallback.condition_count),
            required_start_index=int(cfg.branch.required_start_index),
            used_scene_tokens=used_scenes,
        )
        result.update({"upo": {key: value for key, value in upo.items() if key != "summary"}, "scene_split": provenance, "selected_conditions": selected})
        dataset = _dataset_for_horizon(cfg.data, num_frames=14)
        by_clip = {str(row["sample_id"]): dict(row) for row in dataset.clip_records}
        result["condition_frame_checks"] = []
        for row in selected:
            clip_id, token = str(row["clip_id"]), str(row["sample_tokens"][0])
            if clip_id not in by_clip or str(by_clip[clip_id]["sample_tokens"][0]) != token:
                raise CandidateFallbackError(f"fallback clip 不匹配 dataset: {clip_id}")
            frame = _load_condition_frame(dataset, token)
            result["condition_frame_checks"].append({"clip_id": clip_id, "sha256": _tensor_fingerprint(frame)})
        if not Path(str(cfg.fallback.review_prompt_path)).is_file():
            raise CandidateFallbackError("缺少完整人工评测提示词")
    except Exception as exc:
        result["status"] = "blocked"
        result["blockers"].append(repr(exc))
    return result


def run_candidate_fallback(cfg: Any) -> dict[str, Any]:
    upo = _validate_fallback_config(cfg)
    git = git_state(".")
    if git.get("dirty"):
        raise CandidateFallbackError("正式 PA2-CAND 拒绝 dirty worktree")
    work_dir = Path(str(cfg.work_dir))
    if work_dir.exists():
        raise FileExistsError(f"PA2-CAND run 已存在: {work_dir}")
    split, split_provenance = _load_scene_split(cfg.branch)
    old_conditions = _read_jsonl(Path(str(cfg.fallback.old_pair_run)) / "conditions.jsonl", label="old conditions")
    used_scenes = {str(row["scene_token"]) for row in old_conditions}
    selected = _select_new_conditions(
        split,
        partition=str(cfg.branch.condition_partition),
        offset=int(cfg.fallback.condition_offset),
        count=int(cfg.fallback.condition_count),
        required_start_index=int(cfg.branch.required_start_index),
        used_scene_tokens=used_scenes,
    )
    cfg_fp = config_fingerprint(cfg)
    work_dir.mkdir(parents=True, exist_ok=False)
    manifest = RunManifest(
        run_id=str(cfg.run_id), command=list(sys.argv), config_fingerprint=cfg_fp,
        cache_fingerprint="not-applicable:single-earlier-fork-fallback", seed=int(cfg.seed), git=git,
        environment=environment_fingerprint(), data_split=str(cfg.branch.condition_partition),
    )
    manifest_data = manifest.__dict__ | {
        "task_id": str(cfg.fallback.task_id), "status": "running", "training": False,
        "uses_future_gt": False, "candidate_generation": True,
        "single_fallback": {"condition_offset": 120, "condition_count": 8, "fork_fraction": 0.4, "rho": 0.04},
        "upo_source": {key: value for key, value in upo.items() if key != "summary"},
        "scene_split": split_provenance,
    }
    atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
    save_resolved_config(cfg, str(work_dir / "resolved.yaml"))
    metrics = JsonlMetrics(str(work_dir / "metrics.jsonl"))
    try:
        dataset = _dataset_for_horizon(cfg.data, num_frames=14)
        by_clip = {str(row["sample_id"]): dict(row) for row in dataset.clip_records}
        selected_with_frames = []
        for row in selected:
            token = str(row["sample_tokens"][0])
            frame = _load_condition_frame(dataset, token)
            selected_with_frames.append({**row, "condition_frame": frame, "condition_frame_sha256": _tensor_fingerprint(frame)})
        seed_everything(int(cfg.seed), deterministic=bool(cfg.train.deterministic))
        backbone = build_backbone(cfg.model, load=True, device=str(cfg.device))
        if not isinstance(backbone, SVDBackbone):
            raise CandidateFallbackError("fallback 当前只支持 SVDBackbone")
        backbone.unet.eval()
        backbone.vae.eval()
        backbone.image_encoder.eval()
        metadata = backbone.generation_protocol_metadata()
        base_fp = _base_model_fingerprint(str(cfg.model.pretrained))
        all_conditions = []
        all_candidates = []
        frames_by_candidate: dict[str, torch.Tensor] = {}
        generation_details = {}
        for index, row in enumerate(selected_with_frames):
            condition = _make_condition_record(
                selected=row,
                split=str(cfg.branch.condition_partition),
                camera=str(cfg.data.cameras[0]),
                num_frames=14,
                fps=int(metadata["fps_input"]),
                condition_frame_hash=str(row["condition_frame_sha256"]),
                scheduler_fingerprint=str(metadata["scheduler_config_fingerprint"]),
                base_model_fingerprint=base_fp,
                git_commit=str(git["commit"]),
                config_fingerprint_value=cfg_fp,
            )
            candidates, frames, _, detail = _generate_condition_group(
                backbone=backbone,
                condition=condition,
                condition_frame=row["condition_frame"],
                branch=cfg.branch,
                work_dir=work_dir,
                generation_seed=int(cfg.branch.generation_seed_start) + index,
                direction_seed=int(cfg.branch.direction_seed_start) + index,
                fps=int(metadata["fps_input"]),
                height=int(cfg.data.height),
                width=int(cfg.data.width),
                metrics=metrics,
            )
            detail.pop("independent_frames")
            detail.pop("independent_vae")
            all_conditions.append(condition)
            all_candidates.extend(candidates)
            frames_by_candidate.update(frames)
            generation_details[str(condition["condition_id"])] = detail
            _json_line(work_dir / "conditions.jsonl", condition)
            for candidate in candidates:
                _json_line(work_dir / "candidate_manifest.jsonl", candidate)
        indexed_conditions = validate_conditions(all_conditions, split)
        validate_candidates(all_candidates, indexed_conditions, exact_sibling_count=4)
        del backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            _sync_cuda(str(cfg.device))

        oracle_cfg = load_config(str(cfg.fallback.oracle_config))
        if config_fingerprint(oracle_cfg) != str(cfg.fallback.expected_upo_config_fingerprint):
            raise CandidateFallbackError("oracle config 已偏离 PA2-UPO v2 冻结 fingerprint")
        ropes = {component: float(upo["summary"]["measurement_ropes"][component]) for component in PRIMARY_COMPONENTS}
        threshold = float(upo["summary"]["calibrated_strict_threshold"])
        tracker = PairModeRAFTTracker(oracle_cfg.upo.tracker, device=str(cfg.device))
        by_condition: dict[str, list[dict[str, Any]]] = {}
        for candidate in all_candidates:
            by_condition.setdefault(str(candidate["condition_id"]), []).append(candidate)
        query_records = []
        track_records = []
        support_records = []
        background_records = []
        component_records = []
        interval_records = []
        graph_rows = []
        condition_audits = []
        for condition in all_conditions:
            condition_id = str(condition["condition_id"])
            rows = by_condition[condition_id]
            base = next(row for row in rows if str(row["candidate_role"]) == "base_guard")
            siblings = sorted(
                (row for row in rows if str(row["candidate_role"]) == "sibling"),
                key=lambda row: str(row["candidate_id"]),
            )
            base_id = str(base["candidate_id"])
            sibling_ids = [str(row["candidate_id"]) for row in siblings]
            query_set, observations = tracker.track_condition(
                base_candidate_id=base_id,
                base_frames=frames_by_candidate[base_id],
                sibling_frames={candidate_id: frames_by_candidate[candidate_id] for candidate_id in sibling_ids},
            )
            query_records.append(query_set.to_record(condition_id=condition_id))
            track_records.extend(observation.to_record(condition_id=condition_id) for observation in observations.values())
            quality = {candidate_id: video_quality_metrics(frames_by_candidate[candidate_id]) for candidate_id in [base_id, *sibling_ids]}
            contexts = []
            for candidate_a, candidate_b in __import__("itertools").combinations(sibling_ids, 2):
                edge = _edge_id(condition_id, candidate_a, candidate_b)
                supports = build_common_support(
                    query_set, observations[candidate_a], observations[candidate_b], oracle_cfg.upo.support,
                    window_starts=tuple(int(value) for value in oracle_cfg.upo.windows.starts),
                    window_length=int(oracle_cfg.upo.windows.length),
                )
                for support in supports:
                    evidence = compute_motion_component_evidence(
                        query_set, observations[candidate_a], observations[candidate_b], support,
                        oracle_cfg.upo.motion, image_hw=tuple(int(value) for value in frames_by_candidate[base_id].shape[-2:]),
                    )
                    context = _bootstrap_context(
                        condition_id=condition_id, edge_id=edge, support=support,
                        evidence=evidence, cfg=oracle_cfg,
                    )
                    context["quality"] = _context_quality(quality, support, oracle_cfg)
                    contexts.append(context)
                    support_records.append(support.to_record(condition_id=condition_id, edge_id=edge, query_set=query_set))
                    background_records.append(evidence.background_record(condition_id=condition_id, edge_id=edge, support=support))
                    component_records.append(evidence.component_record(condition_id=condition_id, edge_id=edge, support=support))
            _apply_holm(contexts, oracle_cfg)
            relations = []
            for context in contexts:
                relations.append(_decide_context(
                    context, ropes=ropes, threshold=threshold, cfg=oracle_cfg
                ))
                for component in PRIMARY_COMPONENTS:
                    interval_records.append({
                        "condition_id": condition_id,
                        "edge_id": context["edge_id"],
                        "candidate_a": context["support"].candidate_a,
                        "candidate_b": context["support"].candidate_b,
                        "start_frame": context["support"].start_frame,
                        "end_frame": context["support"].end_frame,
                        **context["intervals"][component],
                        "uses_future_gt": False,
                    })
            graph = build_condition_partial_order(
                condition_id, relations,
                minimum_tie_fraction=float(oracle_cfg.upo.relation.minimum_condition_tie_fraction),
            )
            graph["relations"] = relations
            graph_rows.append(graph)
            base_quality_pairs = [
                quality_comparability(quality[base_id], quality[candidate_id], oracle_cfg.upo.quality)
                for candidate_id in sibling_ids
            ]
            first_frame = {
                candidate_id: _first_frame_metrics(
                    frames_by_candidate[base_id], frames_by_candidate[candidate_id]
                ) for candidate_id in sibling_ids
            }
            generation = generation_details[condition_id]
            checks = {
                "base_guard_exact": bool(generation["base_guard_exact"]),
                "common_prefix_callback_verified": bool(generation["common_prefix_callback_verified"]),
                "perturbation_numerics": (
                    float(generation["actual_perturbation_rms_relative_gap"])
                    <= float(cfg.branch.thresholds.maximum_actual_perturbation_rms_relative_gap)
                    and float(generation["actual_perturbation_max_abs_mean"])
                    <= float(cfg.branch.thresholds.maximum_actual_perturbation_abs_mean)
                ),
                "first_frame": all(
                    float(row["rgb_rms"]) <= float(cfg.fallback.maximum_first_frame_rgb_rms)
                    for row in first_frame.values()
                ),
                "quality": all(bool(row["comparable"]) for row in base_quality_pairs),
                "query_protocol": bool(query_set.valid) and not bool(query_set.diagnostics["fallback_used"]),
                "no_cycle": graph["status"] != "invalid_cycle",
            }
            condition_audits.append({
                "condition_id": condition_id,
                "scene_id": condition["scene_id"],
                "checks": checks,
                "passed": all(checks.values()),
                "query_diagnostics": query_set.diagnostics,
                "first_frame": first_frame,
                "base_quality_comparability": base_quality_pairs,
                "oracle_graph_status": graph["status"],
            })
            for candidate_id in [base_id, *sibling_ids]:
                candidate = next(row for row in rows if str(row["candidate_id"]) == candidate_id)
                atomic_write_json(str(work_dir / str(candidate["score_path"])), {
                    "quality": quality[candidate_id], "oracle_protocol": "PA2-UPO-v2-frozen",
                    "uses_future_gt": False,
                })

        _write_jsonl(work_dir / "query_sets.jsonl", query_records)
        _write_jsonl(work_dir / "paired_tracks.jsonl", track_records)
        _write_jsonl(work_dir / "common_support.jsonl", support_records)
        _write_jsonl(work_dir / "background_fields.jsonl", background_records)
        _write_jsonl(work_dir / "component_differences.jsonl", component_records)
        _write_jsonl(work_dir / "bootstrap_intervals.jsonl", interval_records)
        _write_jsonl(work_dir / "oracle_graphs.jsonl", graph_rows)
        _write_jsonl(work_dir / "condition_audits.jsonl", condition_audits)
        graph_counts = {
            status: sum(row["status"] == status for row in graph_rows)
            for status in ("strict", "tie", "incomparable", "invalid_cycle", "invalid_component_conflict")
        }
        valid_conditions = sum(bool(row["passed"]) for row in condition_audits)
        machine_checks = {
            "eight_new_scene_disjoint_conditions": len(all_conditions) == 8,
            "minimum_legal_conditions": valid_conditions >= int(cfg.fallback.minimum_legal_conditions),
            "all_base_guards_exact": all(row["checks"]["base_guard_exact"] for row in condition_audits),
            "all_callback_and_perturbation_checks": all(
                row["checks"]["common_prefix_callback_verified"]
                and row["checks"]["perturbation_numerics"]
                for row in condition_audits
            ),
            "all_first_frames_valid": all(row["checks"]["first_frame"] for row in condition_audits),
            "no_quality_failure": all(row["checks"]["quality"] for row in condition_audits),
            "no_oracle_cycle": graph_counts["invalid_cycle"] == 0,
            "frozen_oracle_fingerprint": True,
        }
        machine_pass = all(machine_checks.values())
        review_materials = None
        if machine_pass:
            review_materials = _write_review_materials(
                work_dir=work_dir,
                conditions=all_conditions,
                candidates=all_candidates,
                frames_by_candidate=frames_by_candidate,
                fps=int(metadata["fps_input"]),
                seed=int(cfg.fallback.review.seed),
            )
            status = "awaiting_reviews"
            next_gate = "8-case human same-scene structure review"
        else:
            status = "rejected"
            next_gate = "reject SVD common-prefix sibling route"
        summary = {
            "status": status,
            "task_id": str(cfg.fallback.task_id),
            "run_id": str(cfg.run_id),
            "config_fingerprint": cfg_fp,
            "condition_count": len(all_conditions),
            "candidate_count": len(all_candidates),
            "valid_condition_count": valid_conditions,
            "oracle_graph_counts": graph_counts,
            "frozen_oracle": {
                "run_id": upo["summary"]["run_id"],
                "config_fingerprint": upo["summary"]["config_fingerprint"],
                "strict_threshold": threshold,
                "measurement_ropes": ropes,
            },
            "machine": {"machine_pass": machine_pass, "checks": machine_checks},
            "review_materials": review_materials,
            "next_gate": next_gate,
            "training": False,
            "uses_future_gt": False,
        }
        atomic_write_json(str(work_dir / "machine_summary.json"), summary)
        atomic_write_json(str(work_dir / "summary.json"), summary)
        _clean_markers(work_dir)
        marker = "awaiting_reviews" if status == "awaiting_reviews" else "REJECTED"
        atomic_write_text(str(work_dir / marker), sha256_json(summary) + "\n")
        if status == "awaiting_reviews":
            atomic_write_text(str(work_dir / "MACHINE_COMPLETE"), sha256_json(summary) + "\n")
        manifest_data.update({"status": status, "ended_at": utc_now(), "exit_reason": next_gate})
        atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
        return summary
    except Exception as exc:
        failure = {
            "status": "failed", "task_id": str(cfg.fallback.task_id), "run_id": str(cfg.run_id),
            "config_fingerprint": cfg_fp, "error": repr(exc), "training": False,
            "uses_future_gt": False,
        }
        atomic_write_json(str(work_dir / "summary.json"), failure)
        atomic_write_text(str(work_dir / "FAILED"), sha256_json(failure) + "\n")
        manifest_data.update({"status": "failed", "ended_at": utc_now(), "exit_reason": repr(exc)})
        atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="PA2-CAND unique earlier-fork fallback")
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if args.preflight and args.aggregate_only:
        parser.error("--preflight 与 --aggregate-only 不能同时使用")
    cfg = load_config(args.config)
    if args.preflight:
        result = preflight_candidate_fallback(cfg)
    elif args.aggregate_only:
        result = aggregate_candidate_fallback_reviews(cfg)
    else:
        result = run_candidate_fallback(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""PA2 append-only 扩量合并：复用首批 64 条，只为通过 machine gate 后生成 review。"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Mapping as MappingABC
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..config import config_fingerprint, load_config, save_resolved_config
from ..data.physics_dpo_schema import validate_candidates, validate_conditions, validate_preferences, validate_segments
from ..runtime.atomic import atomic_write_json, atomic_write_text
from ..runtime.experiment import RunManifest, utc_now
from ..runtime.fingerprint import environment_fingerprint, file_fingerprint, git_state, sha256_json
from .physics_dpo_branch import _load_scene_split
from .physics_dpo_horizon import _json_line
from .physics_dpo_pair import (
    CONSTRUCTORS,
    PairPilotError,
    _constructor_coverage,
    _read_json,
    _read_jsonl,
    _write_pair_review_materials,
    aggregate_physics_dpo_pair_reviews,
)
from ..preference.pair_scoring import DECISIVE_LABELS


class PairMergeError(RuntimeError):
    """PA2 source provenance、去重或扩量合并失败。"""


def _unique(rows: Sequence[Mapping[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        value = str(row.get(key, ""))
        if not value or value in result:
            raise PairMergeError(f"{label} {key} 缺失或重复: {value!r}")
        result[value] = row
    return result


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_write_text(
        str(path),
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )


def _validated_source(path: Path, *, expected_task: str, expected_status: str) -> dict[str, Any]:
    summary = _read_json(path / "summary.json", label=f"source summary {path}")
    manifest = _read_json(path / "manifest.json", label=f"source manifest {path}")
    marker = path / "COMPLETE"
    if (
        str(summary.get("task_id")) != expected_task
        or str(summary.get("status")) != expected_status
        or str(manifest.get("status")) != expected_status
        or not marker.is_file()
        or marker.read_text(encoding="utf-8").strip() != sha256_json(summary)
        or bool(summary.get("uses_future_gt"))
    ):
        raise PairMergeError(f"source run provenance/status 不符: {path}")
    return {
        "path": str(path), "summary": summary,
        "summary_sha256": file_fingerprint(str(path / "summary.json")),
        "manifest_sha256": file_fingerprint(str(path / "manifest.json")),
    }


def _load_sources(cfg: Any) -> list[dict[str, Any]]:
    specs = list(cfg.merge.sources)
    if len(specs) != 2:
        raise PairMergeError("PA2 merge 必须恰含 base + extension 两个 source")
    return [
        _validated_source(
            Path(str(spec.path)), expected_task=str(spec.expected_task), expected_status=str(spec.expected_status),
        )
        for spec in specs
    ]


class _VideoFrameStore(MappingABC[str, torch.Tensor]):
    """按 review 抽样惰性解码 source MP4，避免把 120-condition 全部视频载入内存。"""

    def __init__(self, paths: Mapping[str, Path]):
        self.paths = dict(paths)
        self.cache: dict[str, torch.Tensor] = {}

    def __getitem__(self, candidate_id: str) -> torch.Tensor:
        if candidate_id in self.cache:
            return self.cache[candidate_id]
        if candidate_id not in self.paths:
            raise KeyError(candidate_id)
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise PairMergeError("PA2 merge review 需要 OpenCV") from exc
        capture = cv2.VideoCapture(str(self.paths[candidate_id]))
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        capture.release()
        if len(frames) != 14:
            raise PairMergeError(f"candidate video 必须可解码 14 帧: {self.paths[candidate_id]}")
        tensor = torch.from_numpy(__import__("numpy").stack(frames)).permute(0, 3, 1, 2).float().div(127.5).sub(1.0)
        self.cache[candidate_id] = tensor
        return tensor

    def __iter__(self) -> Iterator[str]:
        return iter(self.paths)

    def __len__(self) -> int:
        return len(self.paths)


def _combined_artifacts(sources: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names = (
        "conditions.jsonl", "candidates.jsonl", "preferences.jsonl", "segments.jsonl",
        "constructor_candidates.jsonl", "constructor_pairs.jsonl", "constructor_segments.jsonl",
        "candidate_diagnostics.jsonl", "constructor_diagnostics.jsonl",
    )
    combined = {name: [] for name in names}
    candidate_paths: dict[str, Path] = {}
    source_index: list[dict[str, Any]] = []
    for source in sources:
        root = Path(str(source["path"]))
        loaded = {name: _read_jsonl(root / name) for name in names}
        for name, rows in loaded.items():
            combined[name].extend(rows)
        for row in loaded["candidates.jsonl"]:
            candidate_id = str(row["candidate_id"])
            path = root / str(row["rgb_video_path"])
            if candidate_id in candidate_paths or not path.is_file():
                raise PairMergeError(f"core candidate artifact 重复或缺失: {candidate_id}")
            candidate_paths[candidate_id] = path
            source_index.append({"candidate_id": candidate_id, "source_run": str(root), "video_path": str(path)})
        for row in loaded["constructor_candidates.jsonl"]:
            candidate_id = str(row["audit_candidate_id"])
            path = root / str(row["rgb_video_path"])
            if candidate_id in candidate_paths or not path.is_file():
                raise PairMergeError(f"audit candidate artifact 重复或缺失: {candidate_id}")
            candidate_paths[candidate_id] = path
            source_index.append({"candidate_id": candidate_id, "source_run": str(root), "video_path": str(path)})
    combined["candidate_paths"] = candidate_paths
    combined["source_index"] = source_index
    return combined


def preflight_physics_dpo_pair_merge(cfg: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"task_id": str(cfg.merge.task_id), "status": "ready", "uses_gpu": False, "blockers": []}
    try:
        sources = _load_sources(cfg)
        result["sources"] = [{key: value for key, value in source.items() if key != "summary"} for source in sources]
        result["source_condition_count"] = sum(int(source["summary"]["condition_count"]) for source in sources)
        if result["source_condition_count"] != int(cfg.merge.expected_condition_count):
            raise PairMergeError("source condition 总数不等于预注册扩量规模")
        scorers = {str(source["summary"].get("scorer_fingerprint")) for source in sources}
        if len(scorers) != 1:
            raise PairMergeError("source scorer fingerprint 不一致")
    except Exception as exc:
        result["status"] = "blocked"
        result["blockers"].append(repr(exc))
    return result


def run_physics_dpo_pair_merge(cfg: Any) -> dict[str, Any]:
    git = git_state(".")
    if git.get("dirty"):
        raise PairMergeError("PA2 merge 拒绝 dirty worktree")
    work_dir = Path(str(cfg.work_dir))
    if work_dir.exists():
        raise FileExistsError(f"PA2 merge run 已存在: {work_dir}")
    sources = _load_sources(cfg)
    source_summaries = [source["summary"] for source in sources]
    if len({str(row.get("scorer_fingerprint")) for row in source_summaries}) != 1:
        raise PairMergeError("PA2 source scorer fingerprint 不一致")
    if len({str(row.get("scene_split_fingerprint")) for row in source_summaries}) != 1:
        raise PairMergeError("PA2 source scene split fingerprint 不一致")
    if sum(int(row["condition_count"]) for row in source_summaries) != int(cfg.merge.expected_condition_count):
        raise PairMergeError("PA2 source condition count 不等于 expected")
    work_dir.mkdir(parents=True, exist_ok=False)
    cfg_fp = config_fingerprint(cfg)
    manifest = RunManifest(
        run_id=str(cfg.run_id), command=list(sys.argv), config_fingerprint=cfg_fp,
        cache_fingerprint="not-applicable:pa2-append-only-merge", seed=int(cfg.seed), git=git,
        environment=environment_fingerprint(), data_split=str(cfg.pair.condition_partition),
    )
    manifest_data = manifest.__dict__ | {
        "task_id": str(cfg.merge.task_id), "status": "running", "uses_future_gt": False,
        "training": False, "sources": [
            {key: value for key, value in source.items() if key != "summary"} for source in sources
        ],
    }
    atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
    save_resolved_config(cfg, str(work_dir / "resolved.yaml"))
    try:
        artifacts = _combined_artifacts(sources)
        conditions = list(artifacts["conditions.jsonl"])
        candidates = list(artifacts["candidates.jsonl"])
        preferences = list(artifacts["preferences.jsonl"])
        segments = list(artifacts["segments.jsonl"])
        constructor_candidates = list(artifacts["constructor_candidates.jsonl"])
        constructor_pairs = list(artifacts["constructor_pairs.jsonl"])
        constructor_segments = list(artifacts["constructor_segments.jsonl"])
        candidate_diagnostics = list(artifacts["candidate_diagnostics.jsonl"])
        constructor_diagnostics = list(artifacts["constructor_diagnostics.jsonl"])
        _unique(conditions, "condition_id", "condition")
        if len({str(row["scene_id"]) for row in conditions}) != len(conditions):
            raise PairMergeError("append-only source 存在 scene overlap")
        split, split_provenance = _load_scene_split(cfg.pair)
        indexed_conditions = validate_conditions(conditions, split)
        indexed_candidates = validate_candidates(candidates, indexed_conditions, exact_sibling_count=4)
        indexed_preferences = validate_preferences(preferences, indexed_conditions, indexed_candidates)
        validate_segments(segments, indexed_preferences, indexed_candidates)
        _unique(constructor_candidates, "audit_candidate_id", "audit candidate")
        _unique(constructor_pairs, "pair_id", "constructor pair")
        if len(conditions) != int(cfg.merge.expected_condition_count) or len(preferences) != len(conditions):
            raise PairMergeError("merged condition/preference 数不等于预注册规模")
        for name in (
            "conditions.jsonl", "candidates.jsonl", "preferences.jsonl", "segments.jsonl",
            "constructor_candidates.jsonl", "constructor_pairs.jsonl", "constructor_segments.jsonl",
            "candidate_diagnostics.jsonl", "constructor_diagnostics.jsonl",
        ):
            _write_jsonl(work_dir / name, artifacts[name])
        _write_jsonl(work_dir / "candidate_manifest.jsonl", candidates)
        _write_jsonl(work_dir / "source_index.jsonl", artifacts["source_index"])

        decisive = [row for row in preferences if str(row["global_label"]) in DECISIVE_LABELS]
        decisive_segment_conditions = {
            str(row["condition_id"]) for row in decisive
            if any(segment["pair_id"] == row["pair_id"] and segment["label"] in DECISIVE_LABELS for segment in segments)
        }
        constructor_summary = {
            constructor: {
                "pair_count": sum(row["constructor"] == constructor for row in constructor_pairs),
                "decisive_count": sum(row["constructor"] == constructor and row["global_label"] in DECISIVE_LABELS for row in constructor_pairs),
                "abstain_count": sum(row["constructor"] == constructor and row["global_label"] == "abstain" for row in constructor_pairs),
            }
            for constructor in CONSTRUCTORS
        }
        coverage = _constructor_coverage(constructor_summary, len(conditions))
        checks = {
            "validated_core_schema": True,
            "source_runs_disjoint": True,
            "minimum_valid_pairs": len(decisive) >= int(cfg.merge.minimum_valid_pairs),
            "minimum_non_tie_segment_conditions": len(decisive_segment_conditions) >= int(cfg.merge.minimum_valid_pairs),
            "three_constructor_comparison": bool(coverage["pass"]),
        }
        status = "awaiting_reviews" if all(checks.values()) else "blocked"
        review_materials = None
        if status == "awaiting_reviews":
            score_rows = [*candidate_diagnostics, *constructor_diagnostics]
            scores = {str(row["candidate_id"]): dict(row["punc_score"]) for row in score_rows}
            if len(scores) != len(artifacts["candidate_paths"]):
                raise PairMergeError("candidate score 与 artifact 数不一致")
            frames = _VideoFrameStore(artifacts["candidate_paths"])
            review_materials = _write_pair_review_materials(
                work_dir=work_dir, cfg=cfg, frames_by_candidate=frames,
                core_preferences=preferences, core_segments=segments,
                constructor_pairs=constructor_pairs, constructor_segments=constructor_segments,
                scores=scores, fps=int(cfg.merge.fps),
            )
        summary = {
            "status": status, "task_id": str(cfg.merge.task_id), "run_id": str(cfg.run_id),
            "config_fingerprint": cfg_fp, "condition_count": len(conditions),
            "valid_global_pairs": len(decisive), "non_tie_segment_conditions": len(decisive_segment_conditions),
            "constructor_summary": constructor_summary, "constructor_coverage": coverage,
            "machine": {"machine_pass": all(checks.values()), "checks": checks},
            "review_materials": review_materials,
            "scorer_fingerprint": source_summaries[0]["scorer_fingerprint"],
            "scene_split_fingerprint": split_provenance["split_fingerprint"],
            "sources": [{key: value for key, value in source.items() if key != "summary"} for source in sources],
            "next_gate": "PA2 human review" if status == "awaiting_reviews" else "PA2 expansion",
            "uses_future_gt": False,
        }
        atomic_write_json(str(work_dir / "machine_summary.json"), summary)
        atomic_write_json(str(work_dir / "summary.json"), summary)
        if status == "awaiting_reviews":
            atomic_write_text(str(work_dir / "MACHINE_COMPLETE"), sha256_json(summary) + "\n")
            atomic_write_text(str(work_dir / "awaiting_reviews"), sha256_json(summary) + "\n")
        else:
            atomic_write_text(str(work_dir / "COMPLETE"), sha256_json(summary) + "\n")
        manifest_data.update({"status": status, "ended_at": utc_now(), "exit_reason": "human_review_required" if status == "awaiting_reviews" else "machine_pair_gate"})
        atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
        return summary
    except Exception as exc:
        failure = {"status": "failed", "task_id": str(cfg.merge.task_id), "run_id": str(cfg.run_id),
                   "config_fingerprint": cfg_fp, "error": repr(exc), "uses_future_gt": False}
        atomic_write_json(str(work_dir / "summary.json"), failure)
        atomic_write_text(str(work_dir / "FAILED"), sha256_json(failure) + "\n")
        manifest_data.update({"status": "failed", "ended_at": utc_now(), "exit_reason": repr(exc)})
        atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="PA2 append-only expansion merge")
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if args.preflight and args.aggregate_only:
        parser.error("--preflight 与 --aggregate-only 不能同时使用")
    cfg = load_config(args.config)
    if args.preflight:
        result = preflight_physics_dpo_pair_merge(cfg)
    elif args.aggregate_only:
        result = aggregate_physics_dpo_pair_reviews(cfg)
    else:
        result = run_physics_dpo_pair_merge(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

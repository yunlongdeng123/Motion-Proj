#!/usr/bin/env python
"""从官方 nuScenes annotation 和 lightweight map 重构 K4 strict-v2 回归。"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.resim.canonical_hash import canonical_sha256
from motion_proj.resim.io_memory import memory_snapshot, trim_process_heap
from motion_proj.resim.nuscenes_trainval_tracks import (
    TrainvalAnnotationSource,
    build_scene_instances_info,
)
from motion_proj.runtime.atomic import atomic_write_json
from motion_proj.runtime.fingerprint import file_fingerprint
from resim.event_first_n1_cutin import (
    LaneIndexCache,
    _final_transition_records,
    _read_json_bounded,
)
from resim.event_first_n1_kinematic import _frame_times_s, _load_yaml, _track_and_matches
from scripts.replay_n1_cutin_k4_evidence import _expected_match, strict_from_frozen_evidence


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} 必须是 JSON object")
    return value


def _fixture_cases(fixture: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    expected = yaml.safe_load((fixture / "expected_strict_status.yaml").read_text(encoding="utf-8"))
    if not isinstance(expected, dict) or not isinstance(expected.get("cases"), dict):
        raise ValueError("K4 expected fixture 缺少 cases")
    manifest = _read_json(fixture / "audit_manifest_minimal.json")
    cases: dict[str, dict[str, Any]] = {}
    for path in sorted((fixture / "evidence").glob("K4-*.json")):
        expected_hash = manifest.get("evidence_sha256", {}).get(path.name)
        if expected_hash != file_fingerprint(str(path)):
            raise ValueError(f"K4 fixture evidence hash 不匹配: {path.name}")
        value = _read_json(path)
        if value.get("audit_id") != path.stem:
            raise ValueError(f"K4 audit_id 不匹配: {path.name}")
        cases[path.stem] = value
    if set(cases) != set(expected["cases"]):
        raise ValueError("K4 expected case 与 evidence 集合不一致")
    return cases, expected


def _semantic_hash(
    *,
    subject_token: str | None,
    source_token: str | None,
    target_token: str | None,
    crossing_frame: int | None,
    strict: Mapping[str, Any],
) -> str:
    receiver = dict(strict.get("receiver", {}))
    return canonical_sha256(
        {
            "subject_instance_token": subject_token,
            "source_token": source_token,
            "target_token": target_token,
            "crossing_frame": crossing_frame,
            "status": strict.get("status"),
            "primary_reason": strict.get("primary_reason"),
            "receiver_actor_id_by_frame": receiver.get("actor_id_by_frame", []),
            "uses_interpolated_physics": strict.get("uses_interpolated_physics"),
        }
    )


def _fixture_receiver_sequence(evidence: Mapping[str, Any]) -> list[int | None]:
    rows = list(dict(evidence.get("cutin", {})).get("per_frame", []))
    result: list[int | None] = []
    for row in rows:
        receiver = row.get("receiver") if isinstance(row, Mapping) else None
        actor = receiver.get("actor_id") if isinstance(receiver, Mapping) else None
        result.append(int(actor) if actor is not None else None)
    return result


def _match_record(
    records: list[dict[str, Any]], evidence: Mapping[str, Any], actor_tokens: Mapping[int, str]
) -> dict[str, Any] | None:
    subject_token = dict(evidence.get("roles", {})).get("SUBJECT")
    actor_ids = {actor_id for actor_id, token in actor_tokens.items() if token == subject_token}
    source_token = dict(evidence.get("source_run", {})).get("token")
    target_token = dict(evidence.get("target_run", {})).get("token")
    matches = [
        record
        for record in records
        if int(record["actor_id"]) in actor_ids
        and record["source_run"]["token"] == source_token
        and record["target_run"]["token"] == target_token
    ]
    if len(matches) > 1:
        raise RuntimeError(f"K4 event 匹配歧义: {evidence['audit_id']} ({len(matches)})")
    return matches[0] if matches else None


def _case_report(
    audit_id: str,
    evidence: Mapping[str, Any],
    expected: Mapping[str, Any],
    record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    expected_strict = strict_from_frozen_evidence(evidence)
    expected_sequence = _fixture_receiver_sequence(evidence)
    base = {
        "audit_id": audit_id,
        "fixture_event_id": evidence["event_id"],
        "scene_id": evidence["scene_id"],
        "human_verdict": expected["human_verdict"],
        "release_blocking": bool(expected.get("release_blocking", False)),
        "fixture_source_event_record_sha256": evidence.get("event_record_sha256"),
        "fixture_source_token": evidence["source_run"]["token"],
        "fixture_target_token": evidence["target_run"]["token"],
        "fixture_crossing_frame": int(evidence["crossing_frame"]),
        "fixture_receiver_actor_id_by_frame": expected_sequence,
        "expected_status": expected_strict["status"],
        "expected_primary_reason": expected_strict["primary_reason"],
        "expected_semantic_hash": _semantic_hash(
            subject_token=dict(evidence.get("roles", {})).get("SUBJECT"),
            source_token=evidence["source_run"]["token"],
            target_token=evidence["target_run"]["token"],
            crossing_frame=int(evidence["crossing_frame"]),
            strict=expected_strict,
        ),
    }
    if record is None:
        return {
            **base,
            "matched": False,
            "comparison": {"event_key": False, "raw_only": False},
            "passed": False,
            "failure": "event_not_reconstructed_from_raw_annotation_and_map",
        }
    strict = dict(record["strict"])
    actual_sequence = list(dict(strict.get("receiver", {})).get("actor_id_by_frame", []))
    expected_matches, expected_match_reason = _expected_match(expected, strict)
    expected_reason = expected_strict["primary_reason"]
    allowed_primary_reasons = set(expected.get("allowed_primary_reasons", []))
    primary_reason_matches = (
        strict["primary_reason"] == expected_reason
        or strict["primary_reason"] in allowed_primary_reasons
    )
    required_all_reasons = set(expected.get("required_all_reasons", []))
    all_reasons_match = required_all_reasons.issubset(set(strict.get("all_reasons", [])))
    # v1 fixture 与 v2 的窗口枚举策略不同：两者的 ID sequence 必须逐项输出并
    # 比较，但不能把“同一 switch 被更早/更晚 raw frame 捕获”误判为重放失败。
    # 语义 gate 由 required_all_reasons（如 K4-012 的 identity switch）负责。
    receiver_sequence_required = False
    receiver_sequence_matches = actual_sequence == expected_sequence
    semantic_comparable = (
        strict["status"] == expected_strict["status"]
        and strict["primary_reason"] == expected_strict["primary_reason"]
        and receiver_sequence_matches
    )
    actual_semantic = _semantic_hash(
        subject_token=record.get("subject_instance_token"),
        source_token=record["source_run"]["token"],
        target_token=record["target_run"]["token"],
        crossing_frame=int(record["crossing_frame"]),
        strict=strict,
    )
    comparison = {
        "event_key": (
            record["source_run"]["token"] == evidence["source_run"]["token"]
            and record["target_run"]["token"] == evidence["target_run"]["token"]
        ),
        "crossing_frame": int(record["crossing_frame"]) == int(evidence["crossing_frame"]),
        "status": expected_matches,
        "primary_reason": primary_reason_matches,
        "required_all_reasons": all_reasons_match,
        "receiver_actor_id_by_frame": receiver_sequence_matches,
        "receiver_sequence_required": receiver_sequence_required,
        "raw_only": strict.get("uses_interpolated_physics") is False,
        "semantic_record_hash": actual_semantic == base["expected_semantic_hash"],
        "semantic_record_hash_required": semantic_comparable,
    }
    required_keys = [
        "event_key",
        "crossing_frame",
        "status",
        "primary_reason",
        "required_all_reasons",
        "raw_only",
    ]
    if receiver_sequence_required:
        required_keys.append("receiver_actor_id_by_frame")
    if semantic_comparable:
        required_keys.append("semantic_record_hash")
    return {
        **base,
        "matched": True,
        "replay_event_id": record["event_id"],
        "replay_event_record_sha256": record["event_record_sha256"],
        "replay_status": strict["status"],
        "replay_primary_reason": strict["primary_reason"],
        "replay_all_reasons": strict.get("all_reasons", []),
        "replay_receiver_actor_id_by_frame": actual_sequence,
        "replay_subject_geometry": strict.get("subject_geometry"),
        "replay_subject_frames": strict.get("subject", {}).get("per_frame", []),
        "replay_receiver_per_frame": strict.get("receiver_per_frame", []),
        "replay_semantic_hash": actual_semantic,
        "comparison": comparison,
        "passed": all(comparison[key] for key in required_keys),
        "expected_match_reason": expected_match_reason,
    }


def replay(
    config_path: Path,
    fixture: Path,
    output_root: Path,
    *,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"output_root 已存在，拒绝覆盖: {output_root}")
    config = _load_yaml(config_path)
    if config.get("strict", {}).get("interpolation_can_pass_hard_gate") is not False:
        raise ValueError("K4 scene replay 要求 strict interpolation hard gate=false")
    cases, expected = _fixture_cases(fixture)
    output_root.mkdir(parents=True)
    source = TrainvalAnnotationSource(Path(config["dataset_root"]))
    scene_names = sorted({case["scene_id"] for case in cases.values()})
    if cache_root is None:
        cache_dir = output_root / "scene-cache"
        meta = build_scene_instances_info(
            Path(config["dataset_root"]),
            scene_names,
            int(config["interpolate_n"]),
            cache_dir,
            retain=False,
            source=source,
        )
        cache_reused = False
    else:
        cache_dir = cache_root.resolve()
        meta = {}
        for scene_name in scene_names:
            scene = source.resolve_scene(scene_name)
            cache_path = cache_dir / scene_name / "instances" / "instances_info.json"
            if not cache_path.is_file():
                raise FileNotFoundError(f"replay cache 缺少 scene: {cache_path}")
            meta[scene_name] = {
                "scene_token": scene["token"],
                "map_name": source.map_name_by_scene[scene["token"]],
                "cache_path": str(cache_path),
            }
        cache_reused = True
    lane_indices = LaneIndexCache(Path(config["dataset_root"]), config["map_matching"])
    scene_memory: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    matched_records: list[dict[str, Any]] = []
    cases_by_scene: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for audit_id, evidence in cases.items():
        cases_by_scene.setdefault(evidence["scene_id"], []).append((audit_id, evidence))
    for scene_name in scene_names:
        entry = meta[scene_name]
        instances_info = _read_json_bounded(Path(entry["cache_path"]))
        scene = source.resolve_scene(scene_name)
        tracks, matches, actor_tokens, _ = _track_and_matches(
            scene_name,
            instances_info,
            lane_indices[entry["map_name"]],
            config,
        )
        records, _summary = _final_transition_records(
            scene_name,
            lane_indices[entry["map_name"]],
            tracks,
            matches,
            actor_tokens,
            _frame_times_s(source, scene, int(config["interpolate_n"])),
            config,
            entry["map_name"],
        )
        for audit_id, evidence in sorted(cases_by_scene[scene_name]):
            record = _match_record(records, evidence, actor_tokens)
            rows.append(_case_report(audit_id, evidence, expected["cases"][audit_id], record))
            if record is not None:
                matched_records.append({"fixture_audit_id": audit_id, **record})
        scene_memory.append({"scene_id": scene_name, "memory": memory_snapshot()})
        del instances_info, tracks, matches, actor_tokens, records
        trim_process_heap()
        gc.collect()
    rows.sort(key=lambda row: row["audit_id"])
    output_records = output_root / "scene_replay_cases.jsonl"
    with output_records.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    raw_records_path = output_root / "replay_matched_records.jsonl"
    with raw_records_path.open("w", encoding="utf-8") as handle:
        for row in sorted(matched_records, key=lambda value: value["fixture_audit_id"]):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    human_fp = [row for row in rows if row["human_verdict"] == "FALSE_POSITIVE"]
    blocking = [row for row in rows if row["release_blocking"]]
    report = {
        "schema_version": "n1-cutin-k4-scene-replay-v1",
        "purpose": "raw_annotation_and_map_regression_not_prospective_precision",
        "config_fingerprint": file_fingerprint(str(config_path)),
        "fixture": str(fixture.resolve()),
        "case_count": len(rows),
        "scene_count": len(scene_names),
        "cache_reused": cache_reused,
        "all_cases_passed": all(row["passed"] for row in rows),
        "human_false_positive_pass_count": sum(
            row.get("replay_status") == "PASS" for row in human_fp
        ),
        "blocking_tp_passed": all(
            row.get("replay_status") == "PASS" and row["passed"] for row in blocking
        ),
        "uses_interpolated_physics": False,
        "n2_authorized": False,
        "scene_memory": scene_memory,
        "cases": rows,
    }
    report["passed"] = bool(
        report["all_cases_passed"]
        and report["human_false_positive_pass_count"] == 0
        and report["blocking_tp_passed"]
    )
    atomic_write_json(str(output_root / "K4_SCENE_REPLAY.json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/resim/event_first_n1_cutin_final_v1.yaml"),
    )
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    report = replay(args.config, args.fixture, args.output_root, cache_root=args.cache_root)
    print(json.dumps({key: report[key] for key in ("passed", "case_count", "scene_count", "blocking_tp_passed", "human_false_positive_pass_count")}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

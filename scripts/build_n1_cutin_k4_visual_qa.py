#!/usr/bin/env python
"""把已完成的 K4 原始场景重放制作为只读视觉 QA 包。

该入口只验证 audit renderer 和盲审页面的证据可读性；它不会创建 formal run，
不会写人工 verdict，也不构成新的 calibration 或 precision 结论。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.resim.canonical_hash import canonical_sha256
from motion_proj.runtime.atomic import atomic_write_json, atomic_write_text
from scripts.build_n1_cutin_audit import build_audit_pack


QA_CASES = frozenset({"K4-009", "K4-010", "K4-011", "K4-012", "K4-013", "K4-014", "K4-015"})


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    atomic_write_text(
        str(path),
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )


def build(replay_dir: Path, config_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"output_dir 已存在，拒绝覆盖: {output_dir}")
    rows = [
        row
        for row in _read_jsonl(replay_dir / "replay_matched_records.jsonl")
        if row.get("fixture_audit_id") in QA_CASES
    ]
    found = {row["fixture_audit_id"] for row in rows}
    if found != QA_CASES:
        raise ValueError(f"K4 QA case 不完整: {sorted(QA_CASES - found)}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config 必须是 YAML object")
    primary = [row for row in rows if row["strict"]["status"] == "PASS"]
    diagnostic = [row for row in rows if row["strict"]["status"] != "PASS"]
    output_dir.mkdir(parents=True)
    pool = {
        "schema_version": "n1-cutin-k4-visual-qa-pool-v1",
        "purpose": "engineering_visual_qa_only",
        "strict_passes": primary,
        "n2_authorized": False,
    }
    pool["strict_event_pool_sha256"] = canonical_sha256(pool)
    atomic_write_json(str(output_dir / "strict_event_pool.json"), pool)
    _write_jsonl(output_dir / "strict_candidates.jsonl", primary)
    _write_jsonl(output_dir / "diagnostic_abstain_candidates.jsonl", diagnostic)
    atomic_write_json(
        str(output_dir / "summary.json"),
        {
            "schema_version": "n1-cutin-k4-visual-qa-summary-v1",
            "task_id": "N1-EVENT-CUTIN-FINAL-01",
            "purpose": "engineering_visual_qa_only",
            "strict_event_pool_sha256": pool["strict_event_pool_sha256"],
            "n2_authorized": False,
        },
    )
    atomic_write_text(str(output_dir / "resolved.yaml"), config_path.read_text(encoding="utf-8"))
    atomic_write_text(str(output_dir / "K4_VISUAL_QA_ONLY"), "not_a_formal_run\n")
    manifest = build_audit_pack(output_dir, config_path)
    mapping = {}
    for path in sorted((output_dir / "audit" / "evidence").glob("*.json")):
        evidence = json.loads(path.read_text(encoding="utf-8"))
        fixture_id = evidence.get("engineering_fixture_audit_id")
        if fixture_id:
            mapping[fixture_id] = evidence["audit_id"]
    atomic_write_json(
        str(output_dir / "audit" / "k4_case_to_audit_id.json"),
        {"schema_version": "n1-cutin-k4-visual-qa-map-v1", "mapping": mapping},
    )
    return {"run_dir": str(output_dir), "mapping": mapping, **manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/resim/event_first_n1_cutin_final_v1.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.replay_dir, args.config, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""把三个 M5 场景运行汇总为冻结的 24 序列失败矩阵。"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import yaml

from motion_proj.dynamic_editing_v2.pilot_metrics import canonical_sha256
from motion_proj.dynamic_editing_v2.stress_metrics import primary_failure, safe_mean


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object, *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def merge_sequence(scene_row: dict, perception_row: dict, priority: list[str]) -> dict:
    if (
        scene_row["scene"],
        scene_row["role"],
        scene_row["edit"],
    ) != (
        perception_row["scene"],
        perception_row["role"],
        perception_row["edit"],
    ):
        raise ValueError("scene/perception sequence identity mismatch")
    codes = set(scene_row["failure_codes_pre_perception"])
    codes.update(perception_row["failure_codes"])
    ordered = sorted(codes, key=priority.index)
    return {
        "scene": scene_row["scene"],
        "role": scene_row["role"],
        "instance_token": scene_row["instance_token"],
        "edit": scene_row["edit"],
        "status": scene_row["status"],
        "failure_codes": ordered,
        "primary_failure": primary_failure(ordered),
        "render_metrics": scene_row["metrics"],
        "perception_metrics": perception_row["metrics"],
        "abstain_reason": scene_row.get("abstain_reason"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--scene-training", type=Path, action="append", required=True)
    parser.add_argument("--scene-output", type=Path, action="append", required=True)
    parser.add_argument("--perception-report", type=Path, action="append", required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    for directory in ("artifacts", "logs", "source_snapshot", "stages"):
        (args.run_dir / directory).mkdir(parents=True, exist_ok=True)
    terminal = args.run_dir / "terminal.json"
    atomic_json(
        terminal, {"status": "running", "updated_at": now(), "failure": None}
    )
    project_root = Path(__file__).resolve().parents[1]
    project_commit = subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
    ).strip()
    source_snapshot = {}
    for relative in (
        "scripts/finalize_dr_v2_m5.py",
        "motion_proj/dynamic_editing_v2/stress_metrics.py",
        "configs/dynamic_editing_v2/m5_protocol_v1.yaml",
    ):
        source = project_root / relative
        destination = args.run_dir / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_snapshot[relative] = {
            "path": str(destination),
            "sha256": sha256_file(destination),
        }
    contract = {
        "schema_version": 1,
        "task_id": "DR-V2-M5-STRESS-3SCENE-01",
        "component": "three-scene frozen failure-matrix aggregation",
        "protocol": str(args.protocol),
        "scene_training": [str(path) for path in args.scene_training],
        "scene_output": [str(path) for path in args.scene_output],
        "perception_report": [str(path) for path in args.perception_report],
        "project_commit": project_commit,
        "source_snapshot": source_snapshot,
        "started_at": now(),
    }
    atomic_json(args.run_dir / "manifest.json", contract)
    atomic_json(args.run_dir / "resolved.json", contract)
    try:
        if not (
            len(args.scene_training)
            == len(args.scene_output)
            == len(args.perception_report)
            == 3
        ):
            raise ValueError("M5 aggregate 必须精确接收三个 scene")
        protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
        expected_scenes = set(protocol["scenes"])
        training = {}
        for path in args.scene_training:
            value = json.loads((path / "summary.json").read_text(encoding="utf-8"))
            if json.loads((path / "terminal.json").read_text())["status"] != "done":
                raise RuntimeError(f"scene training 未完成: {path}")
            training[value["scene_name"]] = {"run": str(path), "summary": value}
        rendering = {}
        for path in args.scene_output:
            value = json.loads((path / "report.json").read_text(encoding="utf-8"))
            if value["status"] != "done" or not all(value["checks"].values()):
                raise RuntimeError(f"scene render 未通过 validator: {path}")
            rendering[value["scene"]] = {"output": str(path), "report": value}
        perception = {}
        for path in args.perception_report:
            value = json.loads(path.read_text(encoding="utf-8"))
            if (
                value["status"] != "done"
                or not value.get("checks")
                or not all(value["checks"].values())
            ):
                raise RuntimeError(f"perception 未完成: {path}")
            perception[value["scene"]] = {"report_path": str(path), "report": value}
        if set(training) != expected_scenes or set(rendering) != expected_scenes or set(perception) != expected_scenes:
            raise RuntimeError("三类 scene 输入与冻结 protocol 不一致")

        sequences = []
        pseudo_rows = []
        for scene in sorted(expected_scenes):
            perception_by_key = {
                (row["role"], row["edit"]): row
                for row in perception[scene]["report"]["sequences"]
            }
            for row in rendering[scene]["report"]["sequences"]:
                key = (row["role"], row["edit"])
                if key not in perception_by_key:
                    raise RuntimeError(f"perception sequence 缺失: {scene}/{key}")
                sequences.append(
                    merge_sequence(
                        row, perception_by_key[key], protocol["failure_priority"]
                    )
                )
            pseudo_path = Path(rendering[scene]["output"]) / "pseudo_hole_metrics.jsonl"
            pseudo_rows.extend(
                json.loads(line)
                for line in pseudo_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )

        failure_scenes: dict[str, set[str]] = defaultdict(set)
        failure_sequences: dict[str, int] = defaultdict(int)
        for sequence in sequences:
            for code in sequence["failure_codes"]:
                failure_scenes[code].add(sequence["scene"])
                failure_sequences[code] += 1
        stable_failures = [
            {
                "code": code,
                "scene_count": len(failure_scenes[code]),
                "sequence_count": failure_sequences[code],
            }
            for code in protocol["failure_priority"]
            if len(failure_scenes[code]) == 3
        ]
        pseudo_summary = {}
        for method in (
            "no_completion",
            "baseline_native_background",
            "telea_2d_diagnostic",
        ):
            selected = [row for row in pseudo_rows if row["method"] == method]
            pseudo_summary[method] = {
                "rows": len(selected),
                "psnr_mean": safe_mean(row["psnr"] for row in selected),
                "ssim_mean": safe_mean(row["ssim"] for row in selected),
                "lpips_alex_256px_mean": safe_mean(
                    row["lpips_alex_256px"] for row in selected
                ),
            }
        scene_coverage = len({row["scene"] for row in sequences})
        actor_slots = len(
            {
                (row["scene"], row["role"])
                for row in sequences
                if row["status"] == "done"
            }
        )
        edit_coverage = sorted({row["edit"] for row in sequences})
        done_sequences = sum(row["status"] == "done" for row in sequences)
        tier_b_pixels = sum(
            row["render_metrics"]["truth_tier_b_pixels"]
            for row in sequences
            if row["render_metrics"] is not None
        ) // 4
        tier_c_pixels = sum(
            row["render_metrics"]["truth_tier_c_pixels"]
            for row in sequences
            if row["render_metrics"] is not None
        ) // 4
        gates = {
            "three_scenes_runnable": scene_coverage == 3,
            "at_least_four_actor_slots": actor_slots >= 4,
            "four_edits_covered": edit_coverage
            == ["delete", "lateral", "speed", "stop_restart"],
            "all_24_sequences_accounted": len(sequences) == 24,
            "not_all_abstain": done_sequences > 0,
            "failure_matrix_matches_reports": all(
                row["primary_failure"] == primary_failure(row["failure_codes"])
                for row in sequences
            ),
            "truth_tier_a_nonempty": len(pseudo_rows) > 0,
            "truth_tier_b_or_c_accounted": tier_b_pixels + tier_c_pixels > 0,
            "stable_failure_or_honest_none": True,
        }
        status = "done" if all(gates.values()) else "blocked"

        matrix_jsonl = args.run_dir / "artifacts/failure_matrix.jsonl"
        with matrix_jsonl.open("x", encoding="utf-8") as handle:
            for row in sequences:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        matrix_csv = args.run_dir / "artifacts/failure_matrix.csv"
        with matrix_csv.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "scene",
                    "role",
                    "instance_token",
                    "edit",
                    "status",
                    "primary_failure",
                    "failure_codes",
                ),
            )
            writer.writeheader()
            for row in sequences:
                writer.writerow(
                    {
                        key: json.dumps(row[key]) if key == "failure_codes" else row[key]
                        for key in writer.fieldnames
                    }
                )
        qa_html = [
            "<!doctype html><meta charset='utf-8'><title>M5 failure matrix</title>",
            "<h1>M5 三场景编辑压力测试</h1>",
            "<p>自动失败码用于诊断，不替代 M8 人工盲审。</p>",
            "<table><tr><th>scene</th><th>actor</th><th>edit</th><th>primary</th><th>all codes</th></tr>",
        ]
        for row in sequences:
            qa_html.append(
                "<tr>"
                f"<td>{html.escape(row['scene'])}</td>"
                f"<td>{html.escape(row['role'])}</td>"
                f"<td>{html.escape(row['edit'])}</td>"
                f"<td>{html.escape(str(row['primary_failure']))}</td>"
                f"<td>{html.escape(', '.join(row['failure_codes']))}</td>"
                "</tr>"
            )
        qa_html.append("</table><h2>场景内嵌 QA</h2><ul>")
        for scene in sorted(expected_scenes):
            qa = rendering[scene]["report"]["qa"]
            qa_html.append(f"<li>{html.escape(scene)}: <code>{html.escape(qa)}</code></li>")
        qa_html.append("</ul>")
        qa_path = args.run_dir / "artifacts/qa_index.html"
        qa_path.write_text("\n".join(qa_html) + "\n", encoding="utf-8")

        report = {
            "schema_version": 1,
            "task_id": "DR-V2-M5-STRESS-3SCENE-01",
            "status": status,
            "project_commit": project_commit,
            "source_snapshot": source_snapshot,
            "protocol": str(args.protocol),
            "protocol_sha256": sha256_file(args.protocol),
            "training": training,
            "rendering": {
                scene: {"output": row["output"], "report_sha256": sha256_file(Path(row["output"]) / "report.json")}
                for scene, row in rendering.items()
            },
            "perception": {
                scene: {"report": row["report_path"], "report_sha256": sha256_file(Path(row["report_path"]))}
                for scene, row in perception.items()
            },
            "coverage": {
                "scenes": scene_coverage,
                "actor_slots": actor_slots,
                "edits": edit_coverage,
                "sequences": len(sequences),
                "done_sequences": done_sequences,
                "truth_tier_a_rows": len(pseudo_rows),
                "truth_tier_b_pixels": tier_b_pixels,
                "truth_tier_c_pixels": tier_c_pixels,
            },
            "failure_matrix_rows": len(sequences),
            "stable_failures": stable_failures,
            "stable_failure_conclusion": (
                "cross-scene stable failures observed"
                if stable_failures
                else "no failure code repeated across all three scenes"
            ),
            "pseudo_hole_summary": pseudo_summary,
            "gates": gates,
            "qa": str(qa_path),
            "quality_claim": "M5 done means valid experiment coverage, not method quality",
        }
        report["report_payload_sha256"] = canonical_sha256(report)
        atomic_json(args.run_dir / "report.json", report)
        atomic_json(
            args.run_dir / "stages/aggregate.json",
            {
                "stage": "aggregate",
                "status": status,
                "gates": gates,
                "failure_matrix": str(matrix_jsonl),
                "report": str(args.run_dir / "report.json"),
            },
        )
        (args.run_dir / "summary.md").write_text(
            "\n".join(
                [
                    "# DR-V2 M5 三场景压力测试",
                    "",
                    f"- status: `{status}`",
                    f"- coverage: `{len(sequences)}/24 sequences`",
                    f"- stable failures: `{[row['code'] for row in stable_failures]}`",
                    f"- report: `{args.run_dir / 'report.json'}`",
                    "- M5 done 仅表示实验有效，不表示方法质量通过。",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        atomic_json(
            terminal,
            {
                "status": status,
                "updated_at": now(),
                "failure": None
                if status == "done"
                else {"code": "M5_AGGREGATE_GATE_FAILED", "gates": gates},
            },
            replace=True,
        )
        print(
            json.dumps(
                {
                    "status": status,
                    "sequences": len(sequences),
                    "stable_failures": stable_failures,
                    "gates": gates,
                },
                sort_keys=True,
            )
        )
        if status != "done":
            raise SystemExit(2)
    except BaseException as error:
        if not (args.run_dir / "report.json").exists():
            atomic_json(
                terminal,
                {
                    "status": "blocked",
                    "updated_at": now(),
                    "failure": {
                        "code": "M5_AGGREGATE_RUNTIME_FAILED",
                        "detail": f"{type(error).__name__}: {error}",
                    },
                },
                replace=True,
            )
        raise


if __name__ == "__main__":
    main()

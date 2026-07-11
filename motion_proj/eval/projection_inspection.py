"""P1 投影 target 人工检查协议：导出检查包并聚合 reviews。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf

from ..auditor import MotionAuditor
from ..config import config_fingerprint, load_config, save_resolved_config
from ..data import NuScenesFutureVideoDataset
from ..projector import DynamicsProjector
from ..runtime.atomic import atomic_write_json, atomic_write_text
from ..runtime.experiment import ExperimentRegistry, RunManifest, utc_now
from ..runtime.fingerprint import environment_fingerprint, file_fingerprint, git_state, sha256_json
from ..utils.io import write_video
from ..utils.viz import make_comparison_panel
from .synthetic_corrupt_nuscenes import evaluate_synthetic_case

PROTOCOL_VERSION = "projection-target-manual-v1"
VALID_VERDICTS = {"reasonable", "not_reasonable", "borderline"}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _load_case_rows(run_dir: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for path in sorted((run_dir / "cases").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows[int(payload["case_index"])] = payload
    return rows


def summarize_reviews(
    case_rows: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    if not case_rows:
        raise ValueError("没有可汇总的检查 case")
    by_case = {str(row["case_id"]): row for row in case_rows}
    review_map: dict[str, dict[str, Any]] = {}
    for review in reviews:
        case_id = str(review.get("case_id", ""))
        verdict = str(review.get("verdict", ""))
        if case_id and verdict in VALID_VERDICTS:
            review_map[case_id] = review

    reviewed = []
    pending = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(case_rows, key=lambda item: int(item["case_index"])):
        case_id = str(row["case_id"])
        review = review_map.get(case_id)
        if review is None:
            pending.append(case_id)
            continue
        verdict = str(review["verdict"])
        grouped[verdict].append({**row, "review": review})
        if verdict != "borderline":
            reviewed.append((case_id, verdict))

    reasonable = sum(verdict == "reasonable" for _, verdict in reviewed)
    not_reasonable = sum(verdict == "not_reasonable" for _, verdict in reviewed)
    denominator = len(reviewed)
    reasonable_rate = reasonable / denominator if denominator else None
    threshold = float(settings["minimum_reasonable_rate"])
    all_reviewed = len(pending) == 0
    accepted = (
        all_reviewed
        and denominator > 0
        and reasonable_rate is not None
        and reasonable_rate >= threshold
    )
    return {
        "protocol": PROTOCOL_VERSION,
        "cases": len(case_rows),
        "reviewed_cases": denominator,
        "pending_cases": len(pending),
        "borderline_cases": len(grouped["borderline"]),
        "reasonable_cases": reasonable,
        "not_reasonable_cases": not_reasonable,
        "reasonable_rate": reasonable_rate,
        "energy_decreased_rate": sum(bool(row["energy_decreased"]) for row in case_rows) / len(case_rows),
        "mean_eligible_fraction": sum(float(row["eligible_fraction"]) for row in case_rows) / len(case_rows),
        "acceptance": {
            "minimum_reasonable_rate": threshold,
            "all_cases_reviewed": all_reviewed,
            "accepted": accepted,
            "pending_case_ids": pending,
        },
        "by_verdict": {
            verdict: {
                "cases": len(items),
                "mean_energy_decreased_rate": (
                    sum(bool(item["energy_decreased"]) for item in items) / len(items)
                    if items else None
                ),
            }
            for verdict, items in sorted(grouped.items())
        },
        "by_source": _group_by(case_rows, "source"),
        "by_corruption": _group_by(case_rows, "corruption"),
    }


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "unknown"))].append(row)
    return {
        name: {
            "cases": len(items),
            "energy_decreased_rate": sum(bool(item["energy_decreased"]) for item in items) / len(items),
            "mean_eligible_fraction": sum(float(item["eligible_fraction"]) for item in items) / len(items),
        }
        for name, items in sorted(grouped.items())
    }


def _write_case_artifacts(run_dir: Path, payload: dict[str, Any], fps: int) -> dict[str, Any]:
    case_id = str(payload["case_id"])
    case_index = int(payload["case_index"])
    panel = make_comparison_panel(payload["y_corrupted"], payload["x_dagger"], payload["mask"])
    panel_path = run_dir / "panels" / f"{case_id}.mp4"
    write_video(panel, str(panel_path), fps=fps)

    metadata = {key: value for key, value in payload.items() if key not in {"y_corrupted", "x_dagger", "mask"}}
    metadata["panel_path"] = str(panel_path)
    case_path = run_dir / "cases" / f"{case_id}.json"
    atomic_write_json(str(case_path), metadata)
    return metadata


def _write_review_template(run_dir: Path, case_rows: list[dict[str, Any]]) -> None:
    template_path = run_dir / "reviews.template.jsonl"
    if template_path.exists():
        return
    lines = []
    for row in sorted(case_rows, key=lambda item: int(item["case_index"])):
        lines.append(
            json.dumps(
                {
                    "case_id": row["case_id"],
                    "verdict": "reasonable",
                    "reviewer": "human",
                    "notes": "",
                },
                ensure_ascii=False,
            )
        )
    template_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_readme(run_dir: Path) -> None:
    readme = run_dir / "INSPECTION_README.md"
    if readme.exists():
        return
    readme.write_text(
        """# P1 投影 Target 人工检查包

## 如何查看
- 每个 case 对应 `panels/<case_id>.mp4`，布局为 `[corrupted y | x_dagger | reliability mask]`。
- 元数据位于 `cases/<case_id>.json`，包含 corruption、能量与 eligible fraction。

## 判定 rubric
- `reasonable`：相对左侧 corrupted y，右侧 x_dagger 的静态漂移/目标轨迹/闪烁伪影更少，且没有明显新破损。
- `not_reasonable`：x_dagger 未改善或引入新的明显伪影。
- `borderline`：改善与退化并存，无法明确归类；不计入 70% 分子分母。

## 提交 review
1. 复制 `reviews.template.jsonl` 为 `reviews.jsonl`。
2. 逐条填写 `verdict` 与必要 `notes`。
3. 重新运行 aggregate：
   `python -m motion_proj.eval.projection_inspection --config <config> --run-id <run_id> --aggregate-only`

正式验收门槛：`minimum_reasonable_rate >= 0.70`，且全部 case 均已 review。
""",
        encoding="utf-8",
    )


def export_cases(cfg: Any, run_dir: Path, settings: dict[str, Any]) -> list[dict[str, Any]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "panels").mkdir(exist_ok=True)
    (run_dir / "cases").mkdir(exist_ok=True)

    completed = _load_case_rows(run_dir)
    num_cases = int(settings["num_cases"])
    missing = [index for index in range(num_cases) if index not in completed]
    if not missing:
        return [completed[index] for index in range(num_cases)]

    dataset = NuScenesFutureVideoDataset(cfg.data)
    auditor = MotionAuditor(device=str(cfg.device), enable_depth=True)
    projector = DynamicsProjector(smooth_lambda=float(settings["smooth_lambda"]))
    fps = int(settings.get("panel_fps", 4))
    seed = int(cfg.seed)

    for case_index in missing:
        payload = evaluate_synthetic_case(case_index, dataset, auditor, projector, settings, seed)
        completed[case_index] = _write_case_artifacts(run_dir, payload, fps)

    rows = [completed[index] for index in range(num_cases)]
    _write_review_template(run_dir, rows)
    _write_readme(run_dir)
    return rows


def run_experiment(
    cfg: Any,
    run_id: str | None = None,
    aggregate_only: bool = False,
) -> tuple[Path, dict[str, Any]]:
    settings = OmegaConf.to_container(cfg.experiment, resolve=True)
    assert isinstance(settings, dict)
    git = git_state(".")
    if git.get("dirty"):
        raise RuntimeError("正式 P1 检查包导出拒绝在 dirty worktree 上运行")

    cfg_fingerprint = config_fingerprint(cfg)
    experiment_fingerprint = sha256_json(
        {"protocol": PROTOCOL_VERSION, "config": cfg_fingerprint, "git_commit": git["commit"]}
    )
    if run_id is None:
        run_id = (
            f"p1-projection-manual{int(settings['num_cases'])}-s{int(cfg.seed)}-"
            f"{str(git['commit'])[:8]}-{cfg_fingerprint[:8]}"
        )

    root = Path(str(cfg.work_dir))
    run_dir = root / run_id
    complete_path = run_dir / "COMPLETE"
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.json"
    if (
        not aggregate_only
        and complete_path.is_file()
        and complete_path.read_text(encoding="utf-8").strip() == experiment_fingerprint
        and summary_path.is_file()
    ):
        return run_dir, json.loads(summary_path.read_text(encoding="utf-8"))

    resolved_path = run_dir / "resolved.yaml"
    resolved_text = OmegaConf.to_yaml(cfg, resolve=True)
    if resolved_path.exists() and resolved_path.read_text(encoding="utf-8") != resolved_text:
        raise RuntimeError(f"run 目录已有不同配置: {run_dir}")
    if not resolved_path.exists():
        save_resolved_config(cfg, str(resolved_path))

    export_git_commit = str(git["commit"])
    export_experiment_fingerprint = experiment_fingerprint
    manifest_payload: dict[str, Any] | None = None
    if aggregate_only:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"aggregate-only 缺少原始 manifest: {manifest_path}")
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        export_git_commit = str(manifest_payload["git"]["commit"])
        export_experiment_fingerprint = str(manifest_payload["config_fingerprint"])

    registry = ExperimentRegistry(str(root / "experiments.sqlite3"))
    known = {row["run_id"]: row for row in registry.list()}
    if run_id not in known:
        registry.register(run_id, "running", export_experiment_fingerprint, str(run_dir))
    else:
        registry.update(run_id, "running", exit_reason="resume")

    if not aggregate_only:
        manifest = RunManifest(
            run_id=run_id,
            command=list(sys.argv),
            config_fingerprint=experiment_fingerprint,
            cache_fingerprint=f"not-applicable:{PROTOCOL_VERSION}",
            seed=int(cfg.seed),
            git=git,
            environment=environment_fingerprint(),
            data_split=f"{cfg.data.version}:{','.join(cfg.data.cameras)}:synthetic-{int(settings['num_cases'])}",
        )
        manifest.save(str(manifest_path))
        case_rows = export_cases(cfg, run_dir, settings)
    else:
        case_rows = [completed for _, completed in sorted(_load_case_rows(run_dir).items())]

    reviews_path = run_dir / "reviews.jsonl"
    reviews = _load_jsonl(reviews_path)
    completed_at = utc_now()
    summary = summarize_reviews(case_rows, reviews, settings)
    summary.update(
        {
            "run_id": run_id,
            "task_id": str(settings["task_id"]),
            "seed": int(cfg.seed),
            "git_commit": export_git_commit,
            "config_fingerprint": cfg_fingerprint,
            "experiment_fingerprint": export_experiment_fingerprint,
            "completed_at": completed_at,
            "review_fingerprint": file_fingerprint(str(reviews_path)) if reviews_path.is_file() else None,
            "review_aggregation": {
                "git_commit": str(git["commit"]),
                "aggregated_at": completed_at,
            },
        }
    )
    atomic_write_json(str(summary_path), summary)

    if not aggregate_only:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_obj = RunManifest(**manifest)
        manifest_obj.status = "completed"
        manifest_obj.ended_at = utc_now()
        if summary["acceptance"]["all_cases_reviewed"]:
            manifest_obj.exit_reason = (
                "acceptance_passed" if summary["acceptance"]["accepted"] else "acceptance_failed"
            )
        else:
            manifest_obj.exit_reason = "awaiting_reviews"
        manifest_obj.save(str(manifest_path))
        atomic_write_text(str(complete_path), experiment_fingerprint + "\n")

    if summary["acceptance"]["all_cases_reviewed"]:
        exit_reason = "acceptance_passed" if summary["acceptance"]["accepted"] else "acceptance_failed"
    else:
        exit_reason = "awaiting_reviews"
    if aggregate_only:
        assert manifest_payload is not None
        manifest_payload["exit_reason"] = exit_reason
        manifest_payload["status"] = "completed"
        atomic_write_json(str(manifest_path), manifest_payload)
    registry.update(run_id, "completed", exit_reason=exit_reason, summary=summary)
    return run_dir, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    run_dir, summary = run_experiment(cfg, args.run_id, aggregate_only=args.aggregate_only)
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

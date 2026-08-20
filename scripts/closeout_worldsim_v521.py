#!/usr/bin/env python3
"""P10：汇总 V5.2.1 最终 artifact、报告与 Go/NoGo。"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from motion_proj.worldsim_v521.census import sha256_file
from motion_proj.worldsim_v521.protocol import atomic_json, atomic_jsonl, atomic_text, inventory_files


P1_RUN = Path("/root/autodl-tmp/runs/worldsim_v521/20260820T084604Z__p1-base-asset-census-s0-r001")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def require(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"closeout prerequisite 缺失：{path}")


def class_stats(rows: list[dict[str, Any]], base: str) -> dict[str, Any]:
    selected = [row for row in rows if row["base"] == base and row["entity_kind"] == "view"]
    counts = Counter(label for row in selected for label in row["failure_class"] if label != "B-MIXED")
    return {
        "cases": len(selected), "class_counts": dict(sorted(counts.items())),
        "scenes": sorted({row["scene"] for row in selected}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-run", required=True, type=Path)
    parser.add_argument("--confirmation-run", required=True, type=Path)
    parser.add_argument("--closeout-run", required=True, type=Path)
    parser.add_argument("--docs-root", type=Path, default=Path("docs"))
    args = parser.parse_args()
    census = args.census_run.resolve()
    confirmation = args.confirmation_run.resolve()
    closeout = args.closeout_run.resolve()
    prerequisites = [
        census / "summary.json", census / "P3_SUMMARY.json", census / "P4_SUMMARY.json",
        census / "P5_LOCALIZATION.json", census / "P6_P8_SUMMARY.json",
        confirmation / "P9_SUMMARY.json", confirmation / "P9_CONFIRMATION_VERDICT.json",
    ]
    for path in prerequisites:
        require(path)
    if json.loads((census / "summary.json").read_text(encoding="utf-8"))["outcome"] != "p2_gate_pass":
        raise RuntimeError("P2 未通过")
    if json.loads((confirmation / "P9_SUMMARY.json").read_text(encoding="utf-8"))["outcome"] != "p9_confirmation_complete":
        raise RuntimeError("P9 未通过")
    closeout.mkdir(parents=True, exist_ok=False)
    mapping = {
        P1_RUN / "BASE_ASSET_REGISTRY.json": closeout / "BASE_ASSET_REGISTRY.json",
        P1_RUN / "DISCOVERY_CONFIRMATION_FREEZE.json": closeout / "DISCOVERY_CONFIRMATION_FREEZE.json",
        P1_RUN / "MATCHED_FRAME_REGISTRY.jsonl": closeout / "MATCHED_FRAME_REGISTRY.jsonl",
        census / "BASE_CENSUS_METRICS.jsonl": closeout / "BASE_CENSUS_METRICS.jsonl",
        census / "ACTOR_CENSUS_METRICS.jsonl": closeout / "ACTOR_CENSUS_METRICS.jsonl",
        census / "TEMPORAL_CENSUS_METRICS.jsonl": closeout / "TEMPORAL_CENSUS_METRICS.jsonl",
        census / "BADCASE_TAXONOMY_FREEZE.yaml": closeout / "BADCASE_TAXONOMY_FREEZE.yaml",
        census / "BADCASE_LEADERBOARDS.json": closeout / "BADCASE_LEADERBOARDS.json",
        census / "BADCASE_SUMMARY.json": closeout / "BADCASE_SUMMARY_DISCOVERY.json",
        census / "M1_REAUDIT.json": closeout / "M1_REAUDIT.json",
        census / "M2_REAUDIT.json": closeout / "M2_REAUDIT.json",
        census / "M3_REAUDIT.json": closeout / "M3_REAUDIT.json",
        confirmation / "CONFIRMATION_BASE_CENSUS_METRICS.jsonl": closeout / "CONFIRMATION_BASE_CENSUS_METRICS.jsonl",
        confirmation / "CONFIRMATION_ACTOR_CENSUS_METRICS.jsonl": closeout / "CONFIRMATION_ACTOR_CENSUS_METRICS.jsonl",
        confirmation / "CONFIRMATION_TEMPORAL_CENSUS_METRICS.jsonl": closeout / "CONFIRMATION_TEMPORAL_CENSUS_METRICS.jsonl",
        confirmation / "P9_CONFIRMATION_VERDICT.json": closeout / "P9_CONFIRMATION_VERDICT.json",
    }
    for source, destination in mapping.items():
        copy(source, destination)
    discovery_registry = read_jsonl(census / "BADCASE_REGISTRY.jsonl")
    confirmation_registry = read_jsonl(confirmation / "CONFIRMATION_BADCASE_REGISTRY.jsonl")
    combined_registry = sorted(discovery_registry + confirmation_registry, key=lambda row: (row["split_role"], row["case_id"]))
    atomic_jsonl(closeout / "BADCASE_REGISTRY.jsonl", combined_registry)
    panels = read_jsonl(census / "PANEL_REGISTRY.jsonl") + read_jsonl(confirmation / "CONFIRMATION_PANEL_REGISTRY.jsonl")
    atomic_jsonl(closeout / "PANEL_REGISTRY.jsonl", panels)
    p2 = json.loads((census / "summary.json").read_text(encoding="utf-8"))
    p5 = json.loads((census / "P5_LOCALIZATION.json").read_text(encoding="utf-8"))
    p9 = json.loads((confirmation / "P9_SUMMARY.json").read_text(encoding="utf-8"))
    verdicts = json.loads((confirmation / "P9_CONFIRMATION_VERDICT.json").read_text(encoding="utf-8"))
    m1 = json.loads((census / "M1_REAUDIT.json").read_text(encoding="utf-8"))
    m2 = json.loads((census / "M2_REAUDIT.json").read_text(encoding="utf-8"))
    m3 = json.loads((census / "M3_REAUDIT.json").read_text(encoding="utf-8"))
    decision_matrix = [
        {
            "question": "sparse observation", "evidence": "V5.1 scene-1087 historical direct evidence",
            "prevalence": "undefined_no_exact_base_overlap", "cross_scene_stability": "insufficient",
            "independent_from_base_rgb": "unproven", "module": "M1", "v52_priority": "evidence_alignment_first",
        },
        {
            "question": "identity persistence", "evidence": "0471/0379 recall low and persistence=0 historical",
            "prevalence": "undefined_no_identity_denominator", "cross_scene_stability": "historical_two_scene_signal",
            "independent_from_base_rgb": "unproven", "module": "M1", "v52_priority": "evidence_alignment_first",
        },
        {
            "question": "dynamic boundary", "evidence": "frozen B-BOUNDARY registry + Confirmation verdict",
            "prevalence": {base: class_stats(combined_registry, base)["class_counts"].get("B-BOUNDARY", 0) for base in ("adgs", "streetgs")},
            "cross_scene_stability": {key: value for key, value in verdicts["class_verdicts"].items() if key.endswith("|B-BOUNDARY")},
            "independent_from_base_rgb": "partially_separated_by_axis_not_causal", "module": "Base/M1/M2",
            "v52_priority": "conditioned_on_confirmed_direction",
        },
        {
            "question": "disocclusion hole", "evidence": "no audited visibility transition",
            "prevalence": "undefined", "cross_scene_stability": "undefined",
            "independent_from_base_rgb": "unproven", "module": "M2", "v52_priority": "not_method_ready",
        },
        {
            "question": "geometry error", "evidence": "base depth semantics not comparable",
            "prevalence": "undefined", "cross_scene_stability": "undefined",
            "independent_from_base_rgb": "unproven", "module": "Base/M2", "v52_priority": "instrumentation_required",
        },
        {
            "question": "temporal inconsistency", "evidence": "unwarped temporal proxy only; V4 M3 separately confirmed",
            "prevalence": "undefined", "cross_scene_stability": "undefined",
            "independent_from_base_rgb": "unproven", "module": "M3", "v52_priority": "keep_pending",
        },
    ]
    summary = {
        "schema": "worldsim_v521_closeout_summary_v1",
        "task_id": "WS-V521-P10-CLOSEOUT-01",
        "status": "done", "outcome": "v521_base_badcase_basis_frozen",
        "coverage_terminal": "complete_full",
        "discovery_views_per_base": p2["discovery_views"],
        "confirmation_views_per_base": p9["base_rows"] // 2,
        "discovery_registry": {base: class_stats(discovery_registry, base) for base in ("adgs", "streetgs")},
        "confirmation_registry": {base: class_stats(confirmation_registry, base) for base in ("adgs", "streetgs")},
        "m1_conclusion": m1["conclusion"], "m2_conclusion": m2["conclusion"], "m3_conclusion": m3["conclusion"],
        "decision_matrix": decision_matrix,
        "fresh_validation_test_kitti_quality_read": False,
        "stage_h_executed": False, "bki_executed": False,
        "algorithm_candidates_submitted": 0,
        "badcase_basis_frozen": True,
        "problem_definition": "RGB global/actor/boundary measured; geometry/occlusion/identity/true-temporal remain explicit evidence gaps",
        "ready_for_v522_algorithm_design": False,
        "next_stage": "V5.2.2 exact base/M1 overlap + actor identity/visibility/depth/correspondence evidence alignment",
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    atomic_json(closeout / "BADCASE_SUMMARY.json", summary)
    atomic_json(closeout / "RESEARCH_DECISION_MATRIX.json", {"ordered_review_not_scalar": True, "rows": decision_matrix})
    report = f"""# WorldSim V5.2.1 Base Badcase Report

## 结论

两套 exact 基座均完成同帧 census，coverage terminal=`complete_full`。Discovery 每基座 `{p2['discovery_views']}` views，
Confirmation 每基座 `{p9['base_rows'] // 2}` views；没有读取 fresh validation/test/KITTI，Stage H/BKI 未执行，也没有算法 candidate。

可合法冻结的 failure 轴为 `GLOBAL_RGB / ACTOR_RGB / BOUNDARY`；geometry 因两基座 depth 语义不可比而 undefined，
occlusion 因缺 visibility transition annotation 而 undefined，temporal 只有 unwarped proxy，identity/observability 缺 exact denominator。
这些 undefined 是 census 结论，不以 proxy 填补。

## 分母与分类

- AD-GS Discovery：`{class_stats(discovery_registry, 'adgs')}`
- StreetGS Discovery：`{class_stats(discovery_registry, 'streetgs')}`
- AD-GS Confirmation：`{class_stats(confirmation_registry, 'adgs')}`
- StreetGS Confirmation：`{class_stats(confirmation_registry, 'streetgs')}`

P3 以每场 q10 后跨场等权聚合冻结阈值；各轴独立排名，没有综合 scalar。Confirmation 未重拟合阈值、未改 K、未改判词。
class-level verdict 见 `P9_CONFIRMATION_VERDICT.json`。

## Localization 边界

P5 使用完整 census denominator；actor area、actor/static 与 boundary/actor residual ratio 可解释，distance、speed、LiDAR support、
visibility、occlusion transition 全部如实缺测。Spearman 只作相关性，不写因果。
"""
    review = f"""# WorldSim V5.2.1 M1/M2/M3 Review

## M1 — `{m1['conclusion']}`

新 base census 六场与 frozen M1 development/validation 场景 exact overlap=`{m1['exact_overlap_scenes']}`，不足预注册的 2 个独立场景，
Q1–Q4 因而全部 undefined。scene-1087 的 mean observed views=`0.028232`、zero-observation Gaussians=`904,933` 仍是直接历史证据，
但当前不能回答其 base RGB 是否同时失败；不得据此晋级 TrackBayes/TubeBayes。

## M2 — `{m2['conclusion']}`

保留 V4 `154 requests / 214 candidates`、`83 accepted / 71 abstain` 与 full-denominator geometry MAE `+3.3908096237 m` caveat，
以及 V5 geometry-first 无 absolute-safe candidate 的 rejection。没有 exact request→新 case mapping，故不改 router、不搜 threshold。

## M3 — `{m3['conclusion']}`

V4 temporal delta 的 validation/test confirmation 保持有效；V5 constraint-projection rejection 不倒写 V4。新 census 仅有 unwarped proxy，
不能命名 `B-TEMPORAL`，也无法证明 V4 delta 与真实 base temporal badcase 的 exact overlap，因此保持 pending。
"""
    closeout_doc = f"""# WorldSim V5.2.1 Closeout

## Terminal

- badcase basis frozen：yes
- coverage：`complete_full`
- problem definition：RGB global/actor/boundary 已冻结；geometry/occlusion/identity/true-temporal 是明确 evidence gaps
- M1/M2/M3：`{m1['conclusion']} / {m2['conclusion']} / {m3['conclusion']}`
- algorithm candidate：`0`
- fresh validation/test/KITTI：`unread`
- Stage H/BKI：`not executed`

## Go / NoGo

当前对直接进入 V5.2.2 算法结构设计为 **NoGo**。下一步应先建立 base RGB 与 M1 ownership 的 exact same-view overlap，
并冻结 actor identity、visibility/occlusion、可比 depth 与 correspondence evidence；完成前不得再次更换传播核或图扩散方式。

最终机器证据位于 `{closeout}`。
"""
    docs = args.docs_root.resolve()
    atomic_text(docs / "WORLDSIM_V5_2_1_BADCASE_REPORT.md", report)
    atomic_text(docs / "WORLDSIM_V5_2_1_M123_REVIEW.md", review)
    atomic_text(docs / "WORLDSIM_V5_2_1_CLOSEOUT.md", closeout_doc)
    atomic_json(closeout / "run_manifest.json", {"schema": "worldsim_v521_closeout_manifest_v1", "artifacts": inventory_files(closeout)})
    atomic_json(closeout / "status.json", {"status": "done", "outcome": "v521_base_badcase_basis_frozen"})


if __name__ == "__main__":
    main()

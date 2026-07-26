#!/usr/bin/env python
"""final cut-in audit worker：生成逐帧图、信号、盲审页和可填写 JSONL。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.resim.canonical_hash import canonical_sha256
from motion_proj.runtime.atomic import atomic_write_json, atomic_write_text
from motion_proj.runtime.fingerprint import file_fingerprint


COMPONENT_FIELDS = (
    "subject_maneuver_verdict",
    "receiver_corridor_verdict",
    "receiver_relation_verdict",
    "temporal_persistence_verdict",
)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number} 不是 JSON object")
        rows.append(value)
    return rows


def _deterministic_key(record: dict) -> str:
    return hashlib.sha256(record["event_id"].encode("utf-8")).hexdigest()


def _select_primary(records: list[dict], audit: dict) -> list[dict]:
    ordered = sorted(records, key=lambda row: (_deterministic_key(row), row["event_id"]))
    maximum = int(audit["primary_max_count"])
    if len(ordered) <= maximum:
        return ordered
    first_by_scene = []
    seen = set()
    for row in ordered:
        if row["scene_id"] not in seen:
            first_by_scene.append(row)
            seen.add(row["scene_id"])
    target = int(audit["primary_target_count"])
    selected = first_by_scene[:target]
    if len(selected) < target:
        selected_ids = {row["event_id"] for row in selected}
        selected.extend(
            row for row in ordered if row["event_id"] not in selected_ids
        )
        selected = selected[:target]
    return selected


def _draw_box(ax, state: dict, color: str, label: str) -> None:
    xy = state.get("world_xy")
    dimensions = state.get("dimensions_lwh")
    if not isinstance(xy, list) or len(xy) < 2 or not isinstance(dimensions, list) or len(dimensions) < 2:
        return
    length, width = float(dimensions[0]), float(dimensions[1])
    x, y = float(xy[0]), float(xy[1])
    yaw = float(state.get("yaw_rad", 0.0))
    patch = Rectangle(
        (x - length / 2.0, y - width / 2.0),
        length,
        width,
        linewidth=1.6,
        edgecolor=color,
        facecolor="none",
        transform=Affine2D().rotate_around(x, y, yaw) + ax.transData,
    )
    ax.add_patch(patch)
    ax.text(x, y, label, color=color, fontsize=8, ha="center", va="center")


def _render_topdown(event: dict, audit_id: str, output_dir: Path) -> list[str]:
    strict = event["strict"]
    subject_frames = strict["subject"].get("per_frame", [])
    receiver_by_frame = {
        int(row["frame"]): row.get("nearest_rear")
        for row in strict.get("receiver_per_frame", [])
    }
    corridor = strict.get("corridor") or {}
    subject_geometry = strict.get("subject_geometry") or {}
    centerline = np.asarray(
        corridor.get("centerline_xy", subject_geometry.get("target_centerline_xy", [])),
        dtype=float,
    )
    source_centerline = np.asarray(subject_geometry.get("source_centerline_xy", []), dtype=float)
    subject_path = np.asarray(
        [row["world_xy"][:2] for row in subject_frames if isinstance(row.get("world_xy"), list)],
        dtype=float,
    )
    receiver_path = np.asarray(
        [
            row["nearest_rear"]["world_xy"][:2]
            for row in strict.get("receiver_per_frame", [])
            if isinstance(row.get("nearest_rear"), dict)
            and isinstance(row["nearest_rear"].get("world_xy"), list)
        ],
        dtype=float,
    )
    subject_id = strict["subject"].get("actor_id")
    paths = []
    for subject in subject_frames:
        frame = int(subject["frame"])
        fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=140)
        if source_centerline.ndim == 2 and len(source_centerline) > 1:
            ax.plot(source_centerline[:, 0], source_centerline[:, 1], color="#d97706", linestyle="--", linewidth=1.2, label="source lane")
        if centerline.ndim == 2 and len(centerline) > 1:
            ax.plot(centerline[:, 0], centerline[:, 1], color="#2b6cb0", linewidth=1.3, label="target corridor")
            arrow_index = max(0, min(len(centerline) - 2, len(centerline) // 2))
            direction = centerline[arrow_index + 1, :2] - centerline[arrow_index, :2]
            ax.arrow(
                centerline[arrow_index, 0],
                centerline[arrow_index, 1],
                float(direction[0]),
                float(direction[1]),
                color="#2b6cb0",
                width=0.08,
                head_width=0.8,
                length_includes_head=True,
            )
        if subject_path.ndim == 2 and len(subject_path) > 1:
            ax.plot(subject_path[:, 0], subject_path[:, 1], color="#d62728", alpha=0.35, linestyle=":", linewidth=1.2, label="subject raw path")
        if receiver_path.ndim == 2 and len(receiver_path) > 1:
            ax.plot(receiver_path[:, 0], receiver_path[:, 1], color="#2ca02c", alpha=0.35, linestyle=":", linewidth=1.2, label="receiver raw path")
        _draw_box(ax, subject, "#d62728", f"SUBJECT {subject_id if subject_id is not None else '?'}")
        receiver = receiver_by_frame.get(frame)
        if receiver is not None:
            receiver_id = receiver.get("actor_id", "?")
            _draw_box(ax, receiver, "#2ca02c", f"RECEIVER {receiver_id}")
            gap = receiver.get("bumper_gap_m")
            gap_text = f"{float(gap):.2f}" if gap is not None else "n/a"
            ax.text(
                0.02,
                0.02,
                f"raw_2hz  frame={frame}  rear={receiver.get('actor_id')} rank={receiver.get('nearest_rear_rank')} gap={gap_text}m",
                transform=ax.transAxes,
                fontsize=8,
                va="bottom",
            )
        else:
            ax.text(0.02, 0.02, f"raw_2hz  frame={frame}  no rear receiver observed", transform=ax.transAxes, fontsize=8, va="bottom")
        points = []
        if centerline.ndim == 2 and len(centerline):
            points.extend(centerline[:, :2].tolist())
        for state in (subject, receiver):
            if isinstance(state, dict) and isinstance(state.get("world_xy"), list):
                points.append(state["world_xy"][:2])
        for path in (subject_path, receiver_path):
            if path.ndim == 2 and len(path):
                points.extend(path[:, :2].tolist())
        if points:
            array = np.asarray(points, dtype=float)
            margin = 10.0
            ax.set_xlim(float(array[:, 0].min() - margin), float(array[:, 0].max() + margin))
            ax.set_ylim(float(array[:, 1].min() - margin), float(array[:, 1].max() + margin))
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{audit_id} raw 2 Hz frame {frame}")
        ax.set_xlabel("global x (m)")
        ax.set_ylabel("global y (m)")
        ax.grid(alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=7)
        path = output_dir / f"{audit_id}_f{frame}.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path.name))
    return paths


def _render_signals(event: dict, audit_id: str, output_dir: Path) -> str:
    strict = event["strict"]
    subject = strict["subject"].get("per_frame", [])
    receiver = strict.get("receiver", {})
    frames = [int(row["frame"]) for row in subject]
    d = [float(row.get("target_d_m", np.nan)) for row in subject]
    heading = [float(row.get("target_heading_error_deg", np.nan)) for row in subject]
    inside = [1.0 if row.get("box_inside_target_band") else 0.0 for row in subject]
    gap = [np.nan if value is None else float(value) for value in receiver.get("gap_m_by_frame", [])]
    speed = [np.nan if value is None else float(value) for value in receiver.get("longitudinal_speed_mps_by_frame", [])]
    identity = [np.nan if value is None else float(value) for value in receiver.get("actor_id_by_frame", [])]
    rank = [np.nan if value is None else float(value) for value in receiver.get("nearest_rear_rank_by_frame", [])]
    fig, axes = plt.subplots(4, 1, figsize=(9.2, 8.0), dpi=140, sharex=True)
    axes[0].plot(frames, d, marker="o", label="subject signed d_target")
    axes[0].step(frames, inside, where="mid", label="box inside", color="#2ca02c")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].text(0.99, 0.04, "hard evidence: raw 2 Hz", transform=axes[0].transAxes, ha="right", va="bottom", fontsize=7)
    axes[0].set_ylabel("d / inside")
    axes[1].plot(frames, heading, marker="o", color="#9467bd")
    axes[1].set_ylabel("heading deg")
    axes[2].plot(frames[: len(gap)], gap, marker="o", label="bumper gap")
    axes[2].plot(frames[: len(speed)], speed, marker="x", label="receiver v_long")
    axes[2].legend(loc="best", fontsize=8)
    axes[2].set_ylabel("m / mps")
    identity_frames = frames[: len(identity)]
    axes[3].step(identity_frames, identity, where="mid", color="#64748b", alpha=0.65, label="nearest rear actor")
    palette = plt.get_cmap("tab10")
    non_null_ids = sorted({int(value) for value in identity if not np.isnan(value)})
    for index, actor_id in enumerate(non_null_ids):
        positions = [position for position, value in enumerate(identity) if not np.isnan(value) and int(value) == actor_id]
        axes[3].scatter(
            [identity_frames[position] for position in positions],
            [identity[position] for position in positions],
            color=palette(index % 10),
            s=28,
            zorder=3,
            label=f"actor {actor_id}",
        )
    for position in range(1, len(identity)):
        before, after = identity[position - 1], identity[position]
        if not np.isnan(before) and not np.isnan(after) and int(before) != int(after):
            frame = identity_frames[position]
            axes[3].axvline(frame, color="#dc2626", linestyle="--", linewidth=1.2)
            axes[3].annotate(
                f"ID {int(before)}→{int(after)}",
                (frame, float(after)),
                xytext=(4, 8),
                textcoords="offset points",
                color="#b91c1c",
                fontsize=8,
            )
    axes[3].step(frames[: len(rank)], rank, where="mid", label="rear rank")
    axes[3].legend(loc="best", fontsize=8)
    axes[3].set_ylabel("actor / rank")
    axes[3].set_xlabel("raw frame index")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle(f"{audit_id} raw 2 Hz signals")
    path = output_dir / f"{audit_id}.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path.name


def _review_prompt(run_dir: Path) -> str:
    return f"""# N1 final receiver-centric cut-in 人工审核说明

## 目的与边界

本审核只判断 strict machine PASS 是否符合 receiver-centric cut-in 定义；它不评估事故风险、TTC 高低、模型能力或 N2。真实事件必须满足：SUBJECT 在原始 nuScenes 2 Hz keyframe 中原先处于相邻、近似平行 source lane，随后中心与定向车身持续进入 target corridor；同一动态、同向 RECEIVER 在进入前后始终是 SUBJECT 后方最近车辆，且中间无 target-corridor 车辆。

`receiver_branch_merge` 不属于本轮 machine-positive taxonomy。原始 2 Hz 是 hard evidence；10 Hz 插值仅可用于显示，不能补齐任何 component。

## 页面与盲法

- 盲审入口：`audit/index.html`。它不显示 machine status、reason、K4 标签或旧 reviewer notes。
- 工程入口：`audit/debug_index.html`。人工完成前请勿用它裁决。
- 每个 case 含逐 raw-frame topdown、corridor-aligned `(s,d)` 信号曲线和 evidence JSON。相机仅在原始文件存在且双角色可辨认时作为辅助；缺失时页面会明确标记，不能靠肉眼猜身份。

## 四个 component 与 overall verdict

1. `subject_maneuver_verdict`：SUBJECT 是否确实从相邻平行 lane 横向进入，而非正常转弯、路线续接、token jitter 或身份错配。
2. `receiver_corridor_verdict`：target corridor 是否正确，RECEIVER 是否在进入前已属于 target stream。
3. `receiver_relation_verdict`：同一 RECEIVER 是否在全 raw window 处于 SUBJECT 后方最近 rank=1，gap/path 是否清晰。
4. `temporal_persistence_verdict`：pre/post raw support、进入后稳定和 identity 是否足够。

每项填 `VALID`、`INVALID` 或 `UNCERTAIN`。overall 规则：

- `TRUE_POSITIVE`：四项皆 `VALID`，`failure_codes=[]`；
- `FALSE_POSITIVE`：至少一项 `INVALID`，且至少一个 failure code；
- `UNCERTAIN`：至少一项 `UNCERTAIN`、没有 `INVALID`，并说明证据限制。

可用 failure code：`SUBJECT_IDENTITY_MISMATCH`、`SUBJECT_NO_LATERAL_MANEUVER`、`ROUTE_CONTINUATION`、`NORMAL_TURN`、`MAP_MATCH_JITTER`、`INTERPOLATION_ONLY`、`WRONG_BRANCH`、`OPPOSITE_OR_CROSS_TRAFFIC`、`RECEIVER_INVALID`、`RECEIVER_ON_SOURCE_STREAM`、`GAP_INVALID`、`PATH_NOT_CLEAR`、`IDENTITY_NOT_PERSISTENT`、`INSUFFICIENT_VISUAL_EVIDENCE`、`OTHER`。

## JSONL 写法和完整性

只编辑 `audit/review_working.jsonl` 每行的四个 component、`overall_verdict`、`failure_codes`、`reviewer` 和 `notes`。不得改 audit_id、tier、hash、顺序、证据引用或其他字段；不要重排、删行或新增行。`primary_pass` 进入 precision；`diagnostic_abstain` 只诊断 coverage，绝不进入 primary precision。

## 完成后

运行：

```bash
cd {PROJECT_ROOT}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate motionproj
PYTHONPATH=. python scripts/validate_n1_cutin_review.py \\
  --run-dir {run_dir} \\
  --review-file {run_dir}/audit/review_working.jsonl
```

validator 只聚合且不写 verdict、不启动 N2。最终 primary gate 为：正式 pass 至少 15 determinate、8 TP、5 TP scenes、precision≥0.80、Wilson lower≥0.50、UNCERTAIN≤0.15；若审核完整但仅满足 sparse 条件，则只能为 `usable_but_sparse`。无论结果，N2 都不会自动启动。
"""


def _html_rows(items: list[dict], *, debug: bool) -> str:
    rows = []
    for item in items:
        evidence = item["evidence"]
        links = " ".join(
            f'<a href="topdown/{path}">{path}</a>' for path in item["topdown_files"]
        )
        debug_text = ""
        if debug:
            strict = evidence["strict"]
            debug_text = f"<td>{strict['status']}</td><td>{strict['primary_reason']}</td>"
        visible_tier = item["tier"] if debug else {
            "primary_pass": "primary",
            "diagnostic_abstain": "diagnostic",
        }.get(item["tier"], "review")
        rows.append(
            "<tr>"
            f"<td>{item['audit_id']}</td><td>{visible_tier}</td><td>{links}</td>"
            f"<td><a href=\"signals/{item['signal_file']}\">signals</a></td>"
            f"<td><a href=\"evidence/{item['audit_id']}.json\">evidence</a></td>{debug_text}</tr>"
        )
    return "\n".join(rows)


def _write_html(audit_dir: Path, items: list[dict], *, debug: bool) -> None:
    headers = "<th>audit id</th><th>tier</th><th>raw topdown</th><th>signals</th><th>evidence</th>"
    if debug:
        headers += "<th>machine status</th><th>primary reason</th>"
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>N1 final cut-in audit</title>
<style>body{{font-family:Arial,sans-serif;margin:24px}}table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:7px;vertical-align:top}}th{{background:#eee}}a{{word-break:break-all}}</style>
</head><body><h1>{'Debug' if debug else 'Blind'} N1 final receiver-centric cut-in audit</h1>
<p>{'工程页：显示 machine status/reason，仅用于实现诊断。' if debug else '盲审页：不展示任何机器裁决、K4 标签或历史 notes。'}</p>
<table><thead><tr>{headers}</tr></thead><tbody>{_html_rows(items, debug=debug)}</tbody></table></body></html>"""
    atomic_write_text(str(audit_dir / ("debug_index.html" if debug else "index.html")), page)


def build_audit_pack(run_dir: Path, config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("配置必须是 YAML object")
    pool = json.loads((run_dir / "strict_event_pool.json").read_text(encoding="utf-8"))
    passes = _read_jsonl(run_dir / "strict_candidates.jsonl")
    diagnostics_path = run_dir / "diagnostic_abstain_candidates.jsonl"
    diagnostics = _read_jsonl(diagnostics_path) if diagnostics_path.is_file() else []
    primary = _select_primary(passes, config["audit"])
    diagnostic = sorted(diagnostics, key=lambda row: (_deterministic_key(row), row["event_id"]))[: int(config["audit"]["abstain_diagnostic_max_count"])]
    selected = [("primary_pass", row) for row in primary] + [("diagnostic_abstain", row) for row in diagnostic]
    audit_dir = run_dir / "audit"
    evidence_dir = audit_dir / "evidence"
    topdown_dir = audit_dir / "topdown"
    signal_dir = audit_dir / "signals"
    camera_dir = audit_dir / "camera"
    for directory in (evidence_dir, topdown_dir, signal_dir, camera_dir):
        directory.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        str(camera_dir / "README.md"),
        "本 final audit 未将缺失相机资产当作 gate；每条 evidence 显式标记 camera evidence unavailable。\n",
    )
    rendered = []
    template_rows = []
    for index, (tier, event) in enumerate(selected, 1):
        prefix = "P" if tier == "primary_pass" else "A"
        audit_id = f"F1-{prefix}-{index:03d}"
        topdown_files = _render_topdown(event, audit_id, topdown_dir)
        signal_file = _render_signals(event, audit_id, signal_dir)
        evidence = {
            "schema_version": "n1-receiver-cutin-final-audit-evidence-v2",
            "audit_id": audit_id,
            "review_tier": tier,
            "event_id": event["event_id"],
            "event_record_sha256": event["event_record_sha256"],
            "scene_id": event["scene_id"],
            "engineering_fixture_audit_id": event.get("fixture_audit_id"),
            "roles": {
                "SUBJECT": event.get("subject_instance_token"),
                "RECEIVER": event.get("receiver_instance_token"),
            },
            "strict": event["strict"],
            "camera_evidence": {
                "available": False,
                "reason": "CAMERA_METADATA_NOT_EMBEDDED_IN_FINAL_MINING_RECORD",
                "warning": "相机缺失或角色不可辨认；不得靠肉眼猜身份。",
            },
            "topdown_files": topdown_files,
            "signal_file": signal_file,
        }
        evidence_path = evidence_dir / f"{audit_id}.json"
        atomic_write_json(str(evidence_path), evidence)
        topdown_hashes = {
            name: file_fingerprint(str(topdown_dir / name)) for name in topdown_files
        }
        template = {
            "audit_id": audit_id,
            "review_tier": tier,
            "event_id": event["event_id"],
            "evidence_sha256": file_fingerprint(str(evidence_path)),
            "topdown_sha256": topdown_hashes,
            "signal_sha256": file_fingerprint(str(signal_dir / signal_file)),
            "subject_maneuver_verdict": "",
            "receiver_corridor_verdict": "",
            "receiver_relation_verdict": "",
            "temporal_persistence_verdict": "",
            "overall_verdict": "",
            "failure_codes": [],
            "reviewer": "",
            "notes": "",
        }
        template_rows.append(template)
        rendered.append(
            {
                "audit_id": audit_id,
                "tier": tier,
                "evidence": evidence,
                "topdown_files": topdown_files,
                "signal_file": signal_file,
            }
        )
    template_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in template_rows
    )
    atomic_write_text(str(audit_dir / "review_template.jsonl"), template_text)
    atomic_write_text(str(audit_dir / "review_working.jsonl"), template_text)
    atomic_write_text(str(audit_dir / "HUMAN_REVIEW_PROMPT.md"), _review_prompt(run_dir))
    _write_html(audit_dir, rendered, debug=False)
    _write_html(audit_dir, rendered, debug=True)
    immutable_files = [
        "review_template.jsonl",
        "HUMAN_REVIEW_PROMPT.md",
        "index.html",
        "debug_index.html",
        "camera/README.md",
    ]
    immutable_files += [f"evidence/{item['audit_id']}.json" for item in rendered]
    immutable_files += [f"signals/{item['signal_file']}" for item in rendered]
    immutable_files += [
        f"topdown/{name}" for item in rendered for name in item["topdown_files"]
    ]
    hashes = {relative: file_fingerprint(str(audit_dir / relative)) for relative in sorted(immutable_files)}
    manifest = {
        "schema_version": "n1-receiver-cutin-human-audit-pack-v2",
        "run_dir": str(run_dir),
        "strict_event_pool_sha256": pool["strict_event_pool_sha256"],
        "candidate_population_count": len(passes),
        "primary_audit_item_count": len(primary),
        "diagnostic_abstain_item_count": len(diagnostic),
        "audit_item_count": len(template_rows),
        "selection": "deterministic_sha256",
        "immutable_file_hashes": hashes,
        "immutable_artifact_set_sha256": canonical_sha256(hashes),
        "mutable_review_file": "review_working.jsonl",
        "human_verdict_filled": False,
        "n2_authorized": False,
    }
    atomic_write_json(str(audit_dir / "audit_manifest.json"), manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_audit_pack(args.run_dir, args.config)
    print(json.dumps({"run_dir": str(args.run_dir), **manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()

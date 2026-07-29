#!/usr/bin/env python3
"""等待 M4 串行任务结束，审计并聚合六个 AD-GS 官方场景。"""

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT = Path("/root/autodl-tmp/motion_proj")
TASK_ID = "DR-M4-ADGS-6SCENE-01"
EXPECTED_PROJECT_COMMIT = "d90226cbba3854fe67cf32e6cb6be323a106e778"
EXPECTED_UPSTREAM_COMMIT = "9a208512e49c8ddbaa20387921d9648adcd21cb4"
EXPECTED_DATA_MANIFEST_SHA256 = (
    "64c68972a25834757168cd8fdc11c64b134b6ae0d9206a9ebde4064891c16092"
)
EXPECTED_PATCH_SHA256 = (
    "49b4c06ecec6c30f1e80b5abf4d46970920f9d71952acbda273774d9b5b34f48"
)
M3_SCENE_0230_RUN = Path(
    "/root/autodl-tmp/runs/dynamic_recon/DR-M3-ADGS-0230-01/"
    "20260727T195611__scene0230__s0-r3"
)
SCENES = [
    "scene-0230",
    "scene-0242",
    "scene-0255",
    "scene-0295",
    "scene-0518",
    "scene-0749",
]
GATES = {
    "PSNR": {"direction": "ge", "threshold": 30.56},
    "SSIM": {"direction": "ge", "threshold": 0.915},
    "LPIPS(VGG)": {"direction": "le", "threshold": 0.184},
}
METRIC_KEYS = ["PSNR", "SSIM", "LPIPS(VGG)", "LPIPS(ALEX)", "FPS"]


def now():
    return dt.datetime.now(
        dt.timezone(dt.timedelta(hours=8))
    ).isoformat()


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return sha256_bytes(encoded)


def atomic_text(path, payload):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(payload)
    os.replace(str(tmp), str(path))


def atomic_json(path, payload):
    atomic_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )


def append_event(run_dir, event, **payload):
    row = {"timestamp": now(), "event": event, **payload}
    with (run_dir / "events.jsonl").open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)


def load_json(path):
    if not path.is_file():
        raise RuntimeError("缺少必需文件: {}".format(path))
    return json.loads(path.read_text())


def process_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def initialize(run_dir, current_run, sequence_state_dir, poll_seconds):
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("聚合 run 目录非空，禁止覆盖: {}".format(run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "source_snapshot").mkdir()

    script = Path(__file__).resolve()
    snapshot = run_dir / "source_snapshot/finalize_dr_m4.py"
    shutil.copy2(str(script), str(snapshot))
    git_status = subprocess.check_output(
        ["git", "-C", str(PROJECT), "status", "--short"],
        universal_newlines=True,
    )
    project_commit = subprocess.check_output(
        ["git", "-C", str(PROJECT), "rev-parse", "HEAD"],
        universal_newlines=True,
    ).strip()
    if project_commit != EXPECTED_PROJECT_COMMIT:
        raise RuntimeError(
            "聚合启动时 project commit 已变化: {} != {}".format(
                project_commit, EXPECTED_PROJECT_COMMIT
            )
        )
    resolved = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "instance_id": run_dir.name,
        "protocol": "AD-GS official six-scene arithmetic mean",
        "scenes": SCENES,
        "frames": [10, 69],
        "sensors_in_upstream_order": [
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
        ],
        "resolution": [900, 1600],
        "seed": 0,
        "iterations": 60000,
        "metric_source": "official model_60000/results.json",
        "gates": GATES,
        "scene_0230_source_run": str(M3_SCENE_0230_RUN),
        "scene_0242_source_run": str(current_run),
        "remaining_sequence_state_dir": str(sequence_state_dir),
        "poll_seconds": poll_seconds,
    }
    config_fingerprint = canonical_sha256(resolved)
    resolved["config_fingerprint"] = config_fingerprint
    atomic_json(run_dir / "resolved.yaml", resolved)
    manifest = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "instance_id": run_dir.name,
        "started_at": now(),
        "project_commit": project_commit,
        "project_git_status_sha256": sha256_bytes(git_status.encode()),
        "config_fingerprint": config_fingerprint,
        "data_manifest_sha256": EXPECTED_DATA_MANIFEST_SHA256,
        "upstream_commit": EXPECTED_UPSTREAM_COMMIT,
        "compatibility_patch_sha256": EXPECTED_PATCH_SHA256,
        "seed": 0,
        "source_snapshot": {
            "path": str(snapshot),
            "bytes": snapshot.stat().st_size,
            "sha256": sha256_file(snapshot),
        },
    }
    atomic_json(run_dir / "manifest.json", manifest)
    atomic_json(
        run_dir / "terminal.json",
        {"status": "running", "updated_at": now(), "failure": None},
    )
    append_event(
        run_dir,
        "aggregate_wait_started",
        sequence_state_dir=str(sequence_state_dir),
    )


def wait_for_sequence(run_dir, sequence_state_dir, poll_seconds):
    state_path = sequence_state_dir / "state.json"
    pid_path = sequence_state_dir / "sequencer.pid"
    while True:
        state = load_json(state_path)
        status = state.get("status")
        if status == "done":
            append_event(run_dir, "sequence_done_observed")
            return state
        if status == "blocked":
            raise RuntimeError(
                "M4 sequencer blocked: {}".format(state.get("failure"))
            )
        if status != "running":
            raise RuntimeError("M4 sequencer 非法状态: {}".format(status))
        if not pid_path.is_file():
            raise RuntimeError("M4 sequencer 缺少 sequencer.pid")
        pid = int(pid_path.read_text().strip())
        if not process_alive(pid):
            time.sleep(min(5, poll_seconds))
            state = load_json(state_path)
            if state.get("status") == "done":
                append_event(run_dir, "sequence_done_observed")
                return state
            raise RuntimeError(
                "M4 sequencer 进程已消失，state 仍为 {}".format(
                    state.get("status")
                )
            )
        time.sleep(poll_seconds)


def require_equal(actual, expected, label):
    if actual != expected:
        raise RuntimeError(
            "{} 不匹配: {!r} != {!r}".format(label, actual, expected)
        )


def render_counts(model_dir, split):
    root = model_dir / split / "ours_60000"
    return {
        "renders": len(list((root / "renders").glob("*.png"))),
        "ground_truth": len(list((root / "gt").glob("*.png"))),
    }


def load_official_metrics(results_path):
    payload = load_json(results_path)
    if len(payload) != 1:
        raise RuntimeError(
            "{} 顶层结果项不是 1 个: {}".format(results_path, list(payload))
        )
    key, metrics = next(iter(payload.items()))
    if key != "ours_60000":
        raise RuntimeError(
            "{} 结果 key 不是 ours_60000: {}".format(results_path, key)
        )
    normalized = {}
    for metric in METRIC_KEYS:
        if metric not in metrics:
            raise RuntimeError("{} 缺少指标 {}".format(results_path, metric))
        value = float(metrics[metric])
        if not math.isfinite(value):
            raise RuntimeError(
                "{} 的 {} 非 finite".format(results_path, metric)
            )
        normalized[metric] = value
    return normalized


def verify_scene(scene, source_run):
    manifest = load_json(source_run / "manifest.json")
    resolved = load_json(source_run / "resolved.yaml")
    terminal = load_json(source_run / "terminal.json")
    metrics_contract = load_json(source_run / "metrics.json")
    require_equal(terminal.get("status"), "done", scene + " terminal")
    require_equal(metrics_contract.get("status"), "done", scene + " metrics")
    require_equal(resolved.get("scene"), scene, scene + " resolved.scene")
    require_equal(resolved.get("frames"), [10, 69], scene + " frames")
    require_equal(
        resolved.get("sensors_in_upstream_order"),
        ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"],
        scene + " sensors",
    )
    require_equal(resolved.get("resolution"), [900, 1600], scene + " resolution")
    require_equal(resolved.get("seed"), 0, scene + " seed")
    require_equal(
        resolved.get("upstream_commit"),
        EXPECTED_UPSTREAM_COMMIT,
        scene + " upstream commit",
    )
    require_equal(
        resolved.get("compatibility_patch_sha256"),
        EXPECTED_PATCH_SHA256,
        scene + " compatibility patch",
    )
    require_equal(manifest.get("seed"), 0, scene + " manifest seed")
    require_equal(
        manifest.get("data_manifest_sha256"),
        EXPECTED_DATA_MANIFEST_SHA256,
        scene + " data manifest",
    )
    if scene == "scene-0230":
        require_equal(
            manifest.get("task_id"),
            "DR-M3-ADGS-0230-01",
            scene + " source task",
        )
        audit = load_json(source_run / "m3_final_audit.json")
        require_equal(audit.get("verdict"), "pass", scene + " M3 audit")
    else:
        require_equal(manifest.get("task_id"), TASK_ID, scene + " source task")
        require_equal(
            manifest.get("project_commit"),
            EXPECTED_PROJECT_COMMIT,
            scene + " project commit",
        )

    stage_rows = {
        row.get("stage"): row for row in metrics_contract.get("stages", [])
    }
    for stage_name in ["train_60000", "render_60000"]:
        stage = stage_rows.get(stage_name)
        if not stage:
            raise RuntimeError("{} 缺少 stage {}".format(scene, stage_name))
        require_equal(stage.get("status"), "done", scene + " " + stage_name)
        require_equal(stage.get("return_code"), 0, scene + " " + stage_name)
        resource = stage.get("resource", {})
        require_equal(resource.get("oom_delta"), 0, scene + " oom_delta")
        require_equal(
            resource.get("oom_kill_delta"), 0, scene + " oom_kill_delta"
        )
        if resource.get("minimum_disk_free_bytes", 0) < 20 * 1024 ** 3:
            raise RuntimeError("{} stage 磁盘余量低于 20 GiB".format(scene))

    launcher_rc = source_run / "launcher.rc"
    if launcher_rc.is_file():
        require_equal(
            int(launcher_rc.read_text().strip()), 0, scene + " launcher.rc"
        )

    model_dir = source_run / "model_60000"
    results_path = model_dir / "results.json"
    result_train_path = model_dir / "results-train.json"
    official = load_official_metrics(results_path)
    train_metrics = load_official_metrics(result_train_path)
    counts = {
        "test": render_counts(model_dir, "test"),
        "train": render_counts(model_dir, "train"),
    }
    require_equal(
        counts["test"], {"renders": 42, "ground_truth": 42},
        scene + " test render count",
    )
    require_equal(
        counts["train"], {"renders": 138, "ground_truth": 138},
        scene + " train render count",
    )

    checkpoint_dir = model_dir / "point_cloud/iteration_60000"
    checkpoints = {}
    for name in ["point_cloud.ply", "deform.pth", "env.pth"]:
        path = checkpoint_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("{} checkpoint 缺失或为空: {}".format(scene, path))
        checkpoints[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
        }
    checkpoints["point_cloud.ply"]["sha256"] = sha256_file(
        checkpoint_dir / "point_cloud.ply"
    )
    return {
        "scene": scene,
        "source_run": str(source_run),
        "source_task_id": manifest["task_id"],
        "source_instance_id": manifest["instance_id"],
        "config_fingerprint": manifest["config_fingerprint"],
        "official_test_metrics": official,
        "official_train_metrics": train_metrics,
        "render_counts": counts,
        "results": {
            "path": str(results_path),
            "bytes": results_path.stat().st_size,
            "sha256": sha256_file(results_path),
        },
        "checkpoint": checkpoints,
        "resource": {
            stage_name: stage_rows[stage_name]["resource"]
            for stage_name in ["train_60000", "render_60000"]
        },
    }


def collect_sources(current_run, sequence_state):
    sources = {
        "scene-0230": M3_SCENE_0230_RUN,
        "scene-0242": current_run,
    }
    sequence_scenes = sequence_state.get("scenes", {})
    for scene in SCENES[2:]:
        entry = sequence_scenes.get(scene)
        if not entry:
            raise RuntimeError("sequencer state 缺少 {}".format(scene))
        require_equal(entry.get("status"), "done", scene + " sequence status")
        require_equal(entry.get("return_code"), 0, scene + " sequence rc")
        sources[scene] = Path(entry["run_dir"])
    return sources


def aggregate(scene_rows):
    means = {}
    for metric in METRIC_KEYS:
        means[metric] = sum(
            row["official_test_metrics"][metric] for row in scene_rows
        ) / len(scene_rows)
    gates = {}
    for metric, spec in GATES.items():
        value = means[metric]
        passed = (
            value >= spec["threshold"]
            if spec["direction"] == "ge"
            else value <= spec["threshold"]
        )
        gates[metric] = {**spec, "value": value, "passed": passed}
    worst = {
        "PSNR": min(
            scene_rows, key=lambda row: row["official_test_metrics"]["PSNR"]
        ),
        "SSIM": min(
            scene_rows, key=lambda row: row["official_test_metrics"]["SSIM"]
        ),
        "LPIPS(VGG)": max(
            scene_rows,
            key=lambda row: row["official_test_metrics"]["LPIPS(VGG)"],
        ),
    }
    worst_values = {
        metric: {
            "scene": row["scene"],
            "value": row["official_test_metrics"][metric],
        }
        for metric, row in worst.items()
    }
    return {
        "scene_count": len(scene_rows),
        "expected_scene_count": len(SCENES),
        "coverage": len(scene_rows) / len(SCENES),
        "missing_scenes": sorted(set(SCENES) - {row["scene"] for row in scene_rows}),
        "official_test_mean": means,
        "worst_scene": worst_values,
        "gates": gates,
        "all_gates_passed": all(row["passed"] for row in gates.values()),
    }


def write_artifacts(run_dir):
    artifact_paths = []
    for name in [
        "manifest.json",
        "resolved.yaml",
        "metrics.json",
        "metrics.jsonl",
        "summary.json",
        "summary.md",
        "terminal.json",
        "events.jsonl",
    ]:
        artifact_paths.append(run_dir / name)
    artifact_paths.extend(sorted((run_dir / "source_snapshot").glob("*")))
    artifacts = []
    for path in artifact_paths:
        if path.is_file():
            artifacts.append({
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    atomic_json(run_dir / "artifacts.json", {"artifacts": artifacts})


def write_success_or_gate_failure(run_dir, scene_rows, aggregate_row):
    status = "done" if aggregate_row["all_gates_passed"] else "blocked"
    failure = None
    if status == "blocked":
        failed = [
            metric for metric, row in aggregate_row["gates"].items()
            if not row["passed"]
        ]
        failure = {
            "type": "NumericalGateFailure",
            "message": "六场景工程复现带宽未全部通过",
            "failed_metrics": failed,
        }
    metrics = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": status,
        "metric_protocol": "official test, arithmetic mean over six scenes",
        "scenes": scene_rows,
        "aggregate": aggregate_row,
        "failure": failure,
    }
    atomic_json(run_dir / "metrics.json", metrics)
    with (run_dir / "metrics.jsonl").open("w") as handle:
        for row in scene_rows:
            handle.write(json.dumps(
                {"type": "scene", **row}, ensure_ascii=False
            ) + "\n")
        handle.write(json.dumps(
            {"type": "aggregate", **aggregate_row}, ensure_ascii=False
        ) + "\n")
    summary = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "instance_id": run_dir.name,
        "status": status,
        "completed_at": now(),
        "scene_count": aggregate_row["scene_count"],
        "coverage": aggregate_row["coverage"],
        "official_test_mean": aggregate_row["official_test_mean"],
        "worst_scene": aggregate_row["worst_scene"],
        "gates": aggregate_row["gates"],
        "all_gates_passed": aggregate_row["all_gates_passed"],
        "failure": failure,
        "next_action": (
            "M4 已通过，可进入 M5 DGGT upstream smoke"
            if status == "done"
            else "M4 blocked；只允许检查冻结资产/协议/指标链路并做至多一次有根因重跑"
        ),
    }
    atomic_json(run_dir / "summary.json", summary)
    mean = aggregate_row["official_test_mean"]
    summary_md = """# DR-M4 AD-GS 六场景聚合

- 状态：`{status}`
- 覆盖率：`{count}/6 ({coverage:.1%})`
- PSNR：`{psnr:.6f}`（门槛 `>= 30.56`）
- SSIM：`{ssim:.6f}`（门槛 `>= 0.915`）
- LPIPS(VGG)：`{lpips:.6f}`（门槛 `<= 0.184`）
- 三项同时通过：`{passed}`
- 下一步：{next_action}

本结果严格取六个 official `model_60000/results.json` 的算术均值；100/1,000-step 工程画像不进入聚合。
""".format(
        status=status,
        count=aggregate_row["scene_count"],
        coverage=aggregate_row["coverage"],
        psnr=mean["PSNR"],
        ssim=mean["SSIM"],
        lpips=mean["LPIPS(VGG)"],
        passed=str(aggregate_row["all_gates_passed"]).lower(),
        next_action=summary["next_action"],
    )
    atomic_text(run_dir / "summary.md", summary_md)
    atomic_json(
        run_dir / "terminal.json",
        {"status": status, "updated_at": now(), "failure": failure},
    )
    append_event(
        run_dir,
        "aggregate_finished",
        status=status,
        all_gates_passed=aggregate_row["all_gates_passed"],
    )
    write_artifacts(run_dir)
    return status


def write_blocked(run_dir, exc):
    failure = {
        "type": type(exc).__name__,
        "message": str(exc),
    }
    metrics = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "blocked",
        "scenes": [],
        "aggregate": None,
        "failure": failure,
    }
    atomic_json(run_dir / "metrics.json", metrics)
    atomic_text(
        run_dir / "metrics.jsonl",
        json.dumps({"type": "failure", **failure}, ensure_ascii=False) + "\n",
    )
    summary = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "instance_id": run_dir.name,
        "status": "blocked",
        "completed_at": now(),
        "failure": failure,
        "next_action": "M4 blocked；禁止进入 M5，先检查 sequencer/run contract",
    }
    atomic_json(run_dir / "summary.json", summary)
    atomic_text(
        run_dir / "summary.md",
        "# DR-M4 AD-GS 六场景聚合\n\n"
        "- 状态：`blocked`\n"
        "- 失败：`{}: {}`\n"
        "- 下一步：禁止进入 M5，先检查 sequencer/run contract。\n".format(
            type(exc).__name__, exc
        ),
    )
    atomic_json(
        run_dir / "terminal.json",
        {"status": "blocked", "updated_at": now(), "failure": failure},
    )
    append_event(run_dir, "aggregate_blocked", failure=failure)
    write_artifacts(run_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--current-run", required=True)
    parser.add_argument("--sequence-state-dir", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    current_run = Path(args.current_run)
    sequence_state_dir = Path(args.sequence_state_dir)
    poll_seconds = max(10, args.poll_seconds)
    initialized = False
    try:
        initialize(
            run_dir, current_run, sequence_state_dir, poll_seconds
        )
        initialized = True
        sequence_state = wait_for_sequence(
            run_dir, sequence_state_dir, poll_seconds
        )
        sources = collect_sources(current_run, sequence_state)
        scene_rows = []
        for scene in SCENES:
            append_event(
                run_dir, "scene_audit_started",
                scene=scene, source_run=str(sources[scene])
            )
            row = verify_scene(scene, sources[scene])
            scene_rows.append(row)
            append_event(
                run_dir, "scene_audit_done",
                scene=scene, metrics=row["official_test_metrics"]
            )
        aggregate_row = aggregate(scene_rows)
        status = write_success_or_gate_failure(
            run_dir, scene_rows, aggregate_row
        )
        if status != "done":
            return 2
        return 0
    except Exception as exc:
        if initialized:
            write_blocked(run_dir, exc)
        else:
            print("{}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

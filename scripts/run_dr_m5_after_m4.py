#!/usr/bin/env python3
"""等待 M4 聚合通过后，严格一次性启动 M5 DGGT runner。"""

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


PROJECT = Path("/root/autodl-tmp/motion_proj")
M5_RUNNER = PROJECT / "scripts/run_dr_m5_dggt.py"
M5_TASK_ID = "DR-M5-DGGT-NUSC-01"
EXPECTED_PROJECT_COMMIT = "d90226cbba3854fe67cf32e6cb6be323a106e778"
EXPECTED_M5_RUNNER_SHA256 = (
    "3be81eef40d2062b9a8000ed086a5d9fbbb99e81e7aa25d3345dc90b4c07f445"
)


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


def atomic_json(path, payload):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(str(tmp), str(path))


def append_event(state_dir, event, **payload):
    row = {"timestamp": now(), "event": event, **payload}
    with (state_dir / "events.jsonl").open("a") as handle:
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


def verify_frozen_source():
    project_commit = subprocess.check_output(
        ["git", "-C", str(PROJECT), "rev-parse", "HEAD"],
        universal_newlines=True,
    ).strip()
    if project_commit != EXPECTED_PROJECT_COMMIT:
        raise RuntimeError(
            "project commit 已变化: {} != {}".format(
                project_commit, EXPECTED_PROJECT_COMMIT
            )
        )
    runner_sha256 = sha256_file(M5_RUNNER)
    if runner_sha256 != EXPECTED_M5_RUNNER_SHA256:
        raise RuntimeError(
            "M5 runner SHA-256 已变化: {} != {}".format(
                runner_sha256, EXPECTED_M5_RUNNER_SHA256
            )
        )


def wait_m4(state_dir, m4_run, finalizer_pid_file, poll_seconds):
    append_event(state_dir, "wait_m4", m4_aggregate_run=str(m4_run))
    while True:
        terminal = load_json(m4_run / "terminal.json")
        status = terminal.get("status")
        if status == "done":
            launcher_rc = m4_run / "launcher.rc"
            if not launcher_rc.is_file():
                time.sleep(poll_seconds)
                continue
            if int(launcher_rc.read_text().strip()) != 0:
                raise RuntimeError("M4 aggregate launcher.rc 非 0")
            summary = load_json(m4_run / "summary.json")
            if not summary.get("all_gates_passed"):
                raise RuntimeError("M4 terminal done 但 all_gates_passed 非 true")
            append_event(state_dir, "m4_done")
            return
        if status == "blocked":
            raise RuntimeError(
                "M4 aggregate blocked，禁止启动 M5: {}".format(
                    terminal.get("failure")
                )
            )
        if status != "running":
            raise RuntimeError("M4 aggregate 非法状态: {}".format(status))
        if not finalizer_pid_file.is_file():
            raise RuntimeError("缺少 M4 finalizer pid 文件")
        pid = int(finalizer_pid_file.read_text().strip())
        if not process_alive(pid):
            time.sleep(min(5, poll_seconds))
            terminal = load_json(m4_run / "terminal.json")
            if terminal.get("status") == "running":
                raise RuntimeError("M4 finalizer 已消失且 terminal 仍 running")
        time.sleep(poll_seconds)


def launch_m5(state_dir, state, m4_run):
    if state.get("m5_run_dir"):
        raise RuntimeError("controller state 已登记 M5 run，禁止重复启动")
    instance_id = "{}__native-nusc-s0-wm3090".format(
        dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    )
    run_dir = (
        Path("/root/autodl-tmp/runs/dynamic_recon")
        / M5_TASK_ID
        / instance_id
    )
    state["m5_run_dir"] = str(run_dir)
    state["m5_started_at"] = now()
    atomic_json(state_dir / "state.json", state)
    append_event(state_dir, "m5_start", run_dir=str(run_dir))
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        "/root/autodl-tmp/envs/motionproj/bin/python",
        str(M5_RUNNER),
        "--run-dir",
        str(run_dir),
        "--m4-aggregate-run",
        str(m4_run),
    ]
    with (state_dir / "m5.stdout.log").open("wb") as stdout:
        with (state_dir / "m5.stderr.log").open("wb") as stderr:
            result = subprocess.run(
                command,
                cwd=str(PROJECT),
                env=env,
                stdout=stdout,
                stderr=stderr,
            )
    terminal = load_json(run_dir / "terminal.json")
    state["m5_return_code"] = result.returncode
    state["m5_terminal"] = terminal
    state["updated_at"] = now()
    atomic_json(state_dir / "state.json", state)
    if result.returncode != 0 or terminal.get("status") != "done":
        raise RuntimeError(
            "M5 未完成: rc={} terminal={}".format(
                result.returncode, terminal
            )
        )
    append_event(state_dir, "m5_done", run_dir=str(run_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--m4-aggregate-run", required=True)
    parser.add_argument("--finalizer-pid-file", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    state_dir = Path(args.state_dir)
    m4_run = Path(args.m4_aggregate_run)
    finalizer_pid_file = Path(args.finalizer_pid_file)
    state_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": 1,
        "status": "running",
        "started_at": now(),
        "m4_aggregate_run": str(m4_run),
        "expected_project_commit": EXPECTED_PROJECT_COMMIT,
        "expected_m5_runner_sha256": EXPECTED_M5_RUNNER_SHA256,
    }
    atomic_json(state_dir / "state.json", state)
    try:
        verify_frozen_source()
        wait_m4(
            state_dir,
            m4_run,
            finalizer_pid_file,
            max(10, args.poll_seconds),
        )
        verify_frozen_source()
        launch_m5(state_dir, state, m4_run)
    except Exception as exc:
        state["status"] = "blocked"
        state["failure"] = "{}: {}".format(type(exc).__name__, exc)
        state["updated_at"] = now()
        atomic_json(state_dir / "state.json", state)
        append_event(
            state_dir, "controller_blocked", failure=state["failure"]
        )
        raise
    state["status"] = "done"
    state["finished_at"] = now()
    atomic_json(state_dir / "state.json", state)
    append_event(state_dir, "controller_done")


if __name__ == "__main__":
    main()

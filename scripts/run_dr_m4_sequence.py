#!/usr/bin/env python3
"""在 scene-0242 通过后严格串行执行 M4 其余官方场景。"""

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


PROJECT = Path("/root/autodl-tmp/motion_proj")
RUNNER = PROJECT / "scripts/run_dr_adgs_scene.py"
PYTHON = "/root/autodl-tmp/envs/motionproj/bin/python"
TASK_ID = "DR-M4-ADGS-6SCENE-01"
EXPECTED_PROJECT_COMMIT = "d90226cbba3854fe67cf32e6cb6be323a106e778"
EXPECTED_RUNNER_SHA256 = (
    "3fe0b746ff442085d3b0b40bb64a30c8fe2f05fae90842541af37872de150653"
)
REMAINING_SCENES = [
    "scene-0255",
    "scene-0295",
    "scene-0518",
    "scene-0749",
]


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


def read_terminal(run_dir):
    path = run_dir / "terminal.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


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
    runner_sha256 = sha256_file(RUNNER)
    if runner_sha256 != EXPECTED_RUNNER_SHA256:
        raise RuntimeError(
            "runner SHA-256 已变化: {} != {}".format(
                runner_sha256, EXPECTED_RUNNER_SHA256
            )
        )


def wait_current_run(state_dir, current_run, interval):
    append_event(state_dir, "wait_current", run_dir=str(current_run))
    while True:
        terminal = read_terminal(current_run)
        if terminal is None or terminal.get("status") == "running":
            launcher_rc = current_run / "launcher.rc"
            if launcher_rc.is_file():
                rc = int(launcher_rc.read_text().strip())
                raise RuntimeError(
                    "scene-0242 launcher 已退出但终态未完成: rc={}".format(
                        rc
                    )
                )
            launcher_pid = current_run / "launcher.pid"
            if launcher_pid.is_file():
                pid = int(launcher_pid.read_text().strip())
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    raise RuntimeError(
                        "scene-0242 launcher 已消失且没有完成终态"
                    )
            time.sleep(interval)
            continue
        if terminal.get("status") != "done":
            raise RuntimeError(
                "scene-0242 未完成，sequencer 停止: {}".format(terminal)
            )
        launcher_rc = current_run / "launcher.rc"
        if not launcher_rc.is_file():
            time.sleep(interval)
            continue
        if int(launcher_rc.read_text().strip()) != 0:
            raise RuntimeError("scene-0242 launcher.rc 非 0")
        append_event(state_dir, "current_done", run_dir=str(current_run))
        return


def load_state(state_path, current_run):
    if state_path.is_file():
        return json.loads(state_path.read_text())
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "started_at": now(),
        "current_source_run": str(current_run),
        "expected_project_commit": EXPECTED_PROJECT_COMMIT,
        "expected_runner_sha256": EXPECTED_RUNNER_SHA256,
        "scenes": {},
        "status": "running",
    }


def run_scene(state_dir, state, scene):
    scene_state = state["scenes"].get(scene)
    if scene_state:
        run_dir = Path(scene_state["run_dir"])
        terminal = read_terminal(run_dir)
        if terminal and terminal.get("status") == "done":
            append_event(
                state_dir, "scene_already_done", scene=scene,
                run_dir=str(run_dir)
            )
            return
        raise RuntimeError(
            "{} 已有非 done 实例，禁止自动覆盖或重跑: {}".format(
                scene, run_dir
            )
        )

    instance_id = "{}__{}__s0-wm3090".format(
        dt.datetime.now().strftime("%Y%m%dT%H%M%S"),
        scene.replace("scene-", "scene"),
    )
    run_dir = (
        Path("/root/autodl-tmp/runs/dynamic_recon")
        / TASK_ID / instance_id
    )
    state["scenes"][scene] = {
        "run_dir": str(run_dir),
        "status": "running",
        "started_at": now(),
    }
    atomic_json(state_dir / "state.json", state)
    append_event(
        state_dir, "scene_start", scene=scene, run_dir=str(run_dir)
    )

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        PYTHON,
        str(RUNNER),
        "--run-dir",
        str(run_dir),
        "--scene",
        scene,
        "--task-id",
        TASK_ID,
        "--through",
        "train60000",
    ]
    with (state_dir / "{}.stdout.log".format(scene)).open("wb") as stdout:
        with (state_dir / "{}.stderr.log".format(scene)).open("wb") as stderr:
            result = subprocess.run(
                command,
                cwd=str(PROJECT),
                env=env,
                stdout=stdout,
                stderr=stderr,
            )
    terminal = read_terminal(run_dir)
    if result.returncode != 0 or not terminal:
        raise RuntimeError(
            "{} runner 失败: rc={} terminal={}".format(
                scene, result.returncode, terminal
            )
        )
    if terminal.get("status") != "done":
        raise RuntimeError(
            "{} 未达到 done: {}".format(scene, terminal)
        )
    state["scenes"][scene].update({
        "status": "done",
        "finished_at": now(),
        "return_code": result.returncode,
    })
    atomic_json(state_dir / "state.json", state)
    append_event(
        state_dir, "scene_done", scene=scene, run_dir=str(run_dir)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--current-run", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    state_dir = Path(args.state_dir)
    current_run = Path(args.current_run)
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "state.json"
    state = load_state(state_path, current_run)
    atomic_json(state_path, state)

    try:
        verify_frozen_source()
        wait_current_run(
            state_dir, current_run, max(10, args.poll_seconds)
        )
        for scene in REMAINING_SCENES:
            verify_frozen_source()
            run_scene(state_dir, state, scene)
    except Exception as exc:
        state["status"] = "blocked"
        state["failure"] = "{}: {}".format(type(exc).__name__, exc)
        state["updated_at"] = now()
        atomic_json(state_path, state)
        append_event(
            state_dir, "sequence_blocked", failure=state["failure"]
        )
        raise

    state["status"] = "done"
    state["finished_at"] = now()
    atomic_json(state_path, state)
    append_event(state_dir, "sequence_done")


if __name__ == "__main__":
    main()

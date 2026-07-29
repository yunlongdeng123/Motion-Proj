#!/usr/bin/env python3
"""等待 M5 终态后依次执行 M6 压力审计与 M7 novelty 裁决。"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


PROJECT = Path("/root/autodl-tmp/motion_proj")
PYTHON = "/root/autodl-tmp/envs/motionproj/bin/python"
ADGS_PYTHON = "/root/autodl-tmp/envs/adgs/bin/python"
RUN_ROOT = Path("/root/autodl-tmp/runs/dynamic_recon")
M6_RUNNER = PROJECT / "scripts/run_dr_m6_stress.py"
M7_RUNNER = PROJECT / "scripts/finalize_dr_m7_novelty.py"
M5_COMMON_RUNNER = PROJECT / "scripts/finalize_dr_m5_common_diagnostic.py"
PSEUDO_TRACKS = PROJECT / "motion_proj/dynamic_recon/pseudo_tracks.py"
# 启动前由主 agent 用远端 sha256sum 冻结。
EXPECTED_SOURCE_SHA256 = {
    "finalize_dr_m5_common_diagnostic.py": "44909502e4690a624e4971f3828caadf1162e778cf39cdf2b6a8d2c6cdce0c0b",
    "run_dr_m6_stress.py": "2d5bffea42ab106458f9a7755f73af5d19f60ca490a79f920f6cc838f528fac6",
    "finalize_dr_m7_novelty.py": "b9c0c2e7c2137905ea945b535e409c76d1ac701376d5deafdfc7c65a47f8147a",
    "pseudo_tracks.py": "1d0c6083e4c79cc4d0df8ed583537546e638b2d28c242907517d1cf3bf5c3535",
}


def now() -> str:
    return dt.datetime.now(
        dt.timezone(dt.timedelta(hours=8))
    ).isoformat()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    os.replace(str(temporary), str(path))


def append_event(state_dir: Path, event: str, **payload) -> None:
    row = {"timestamp": now(), "event": event, **payload}
    with (state_dir / "events.jsonl").open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)


def load_json(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def verify_sources() -> None:
    paths = [M5_COMMON_RUNNER, M6_RUNNER, M7_RUNNER, PSEUDO_TRACKS]
    for path in paths:
        expected = EXPECTED_SOURCE_SHA256[path.name]
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"source hash 已变化: {path.name} {actual} != {expected}"
            )


def wait_m5(state_dir: Path, m5_run: Path, poll_seconds: int) -> dict:
    append_event(state_dir, "wait_m5", m5_run=str(m5_run))
    while True:
        terminal = load_json(m5_run / "terminal.json")
        if terminal is None or terminal.get("status") == "running":
            time.sleep(poll_seconds)
            continue
        if terminal.get("status") not in {"done", "blocked"}:
            raise RuntimeError(f"M5 非法终态: {terminal}")
        if terminal.get("status") == "blocked" and not terminal.get("failure"):
            raise RuntimeError("M5 blocked 但没有 failure")
        append_event(
            state_dir,
            "m5_terminal_observed",
            status=terminal["status"],
            failure=terminal.get("failure"),
        )
        return terminal


def new_run_dir(task_id: str, suffix: str) -> Path:
    instance_id = f"{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}__{suffix}"
    return RUN_ROOT / task_id / instance_id


def execute(
    state_dir: Path,
    state: dict,
    name: str,
    command: list[str],
    run_dir: Path,
) -> None:
    state[name] = {
        "status": "running",
        "run_dir": str(run_dir),
        "started_at": now(),
        "command": command,
    }
    atomic_json(state_dir / "state.json", state)
    append_event(state_dir, f"{name}_start", run_dir=str(run_dir))
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    with (state_dir / f"{name}.stdout.log").open("wb") as stdout:
        with (state_dir / f"{name}.stderr.log").open("wb") as stderr:
            result = subprocess.run(
                command,
                cwd=str(PROJECT),
                env=environment,
                stdout=stdout,
                stderr=stderr,
            )
    terminal = load_json(run_dir / "terminal.json")
    if result.returncode != 0 or terminal is None:
        raise RuntimeError(
            f"{name} 失败: rc={result.returncode}, terminal={terminal}"
        )
    if terminal.get("status") != "done":
        raise RuntimeError(f"{name} 未到 done: {terminal}")
    state[name].update(
        {
            "status": "done",
            "finished_at": now(),
            "return_code": result.returncode,
        }
    )
    atomic_json(state_dir / "state.json", state)
    append_event(state_dir, f"{name}_done", run_dir=str(run_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--m5-run", required=True)
    parser.add_argument("--poll-seconds", type=int, default=15)
    args = parser.parse_args()
    state_dir = Path(args.state_dir)
    m5_run = Path(args.m5_run)
    if state_dir.exists() and any(state_dir.iterdir()):
        raise RuntimeError(f"controller state dir 非空: {state_dir}")
    state_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": 1,
        "status": "running",
        "started_at": now(),
        "m5_run": str(m5_run),
        "expected_source_sha256": EXPECTED_SOURCE_SHA256,
    }
    atomic_json(state_dir / "state.json", state)
    try:
        verify_sources()
        m5_terminal = wait_m5(
            state_dir, m5_run, max(10, args.poll_seconds)
        )
        state["m5_terminal"] = m5_terminal
        verify_sources()
        m5_common_run = None
        if m5_terminal["status"] == "done":
            m5_common_run = new_run_dir(
                "DR-M5-DGGT-NUSC-01", "common-observation-s0-wm3090"
            )
            execute(
                state_dir,
                state,
                "m5_common",
                [
                    ADGS_PYTHON,
                    str(M5_COMMON_RUNNER),
                    "--run-dir",
                    str(m5_common_run),
                    "--m5-run",
                    str(m5_run),
                ],
                m5_common_run,
            )
        else:
            state["m5_common"] = {
                "status": "not_run_upstream_blocked",
                "run_dir": None,
                "reason": "M5 native blocked with preserved failure evidence",
            }
            atomic_json(state_dir / "state.json", state)
            append_event(
                state_dir,
                "m5_common_not_run",
                reason=state["m5_common"]["reason"],
            )
        verify_sources()
        m6_run = new_run_dir(
            "DR-M6-STRESS-01", "identity-audit-s0-wm3090"
        )
        m6_command = [
            PYTHON,
            str(M6_RUNNER),
            "--run-dir",
            str(m6_run),
            "--m5-run",
            str(m5_run),
        ]
        if m5_common_run is not None:
            m6_command.extend(["--m5-common-run", str(m5_common_run)])
        execute(
            state_dir,
            state,
            "m6",
            m6_command,
            m6_run,
        )
        verify_sources()
        m7_run = new_run_dir(
            "DR-M7-HYPOTHESIS-01", "novelty-audit-s0-wm3090"
        )
        execute(
            state_dir,
            state,
            "m7",
            [
                PYTHON,
                str(M7_RUNNER),
                "--run-dir",
                str(m7_run),
                "--m6-run",
                str(m6_run),
            ],
            m7_run,
        )
    except Exception as exc:
        state["status"] = "blocked"
        state["failure"] = f"{type(exc).__name__}: {exc}"
        state["updated_at"] = now()
        atomic_json(state_dir / "state.json", state)
        append_event(state_dir, "controller_blocked", failure=state["failure"])
        raise
    state["status"] = "done"
    state["finished_at"] = now()
    atomic_json(state_dir / "state.json", state)
    append_event(state_dir, "controller_done")


if __name__ == "__main__":
    main()

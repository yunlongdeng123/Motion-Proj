#!/usr/bin/env python3
"""Validate and reuse a completed M3 checkpoint after post-render guard stop."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import torch


SOURCE_SUCCESS_STAGES = (
    "train_profile100",
    "native_actor_mapping_probe",
    "train_profile1000",
)
EXPECTED_FAILURE_CODE = "M3_FORMAL_RENDER_CGROUP_MEMORY_GUARD"
EXPECTED_STEP = 30_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def require_status(path: Path, statuses: set[str]) -> dict:
    payload = load_object(path)
    if payload.get("status") not in statuses:
        raise RuntimeError(f"unexpected stage status at {path}: {payload.get('status')}")
    return payload


def atomic_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refuse to overwrite stage artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_recovery_contract(
    *, formal: dict, terminal: dict, checkpoint: Path, checkpoint_step: int
) -> None:
    failure = terminal.get("failure") or {}
    if formal.get("status") != "blocked":
        raise RuntimeError("source formal stage must remain blocked")
    if formal.get("stop_reason") != "memory.current/memory.max >= 0.90 twice":
        raise RuntimeError(f"unexpected formal stop reason: {formal.get('stop_reason')}")
    if failure.get("code") != EXPECTED_FAILURE_CODE:
        raise RuntimeError(f"unexpected terminal failure code: {failure.get('code')}")
    if checkpoint_step != EXPECTED_STEP:
        raise RuntimeError(
            f"formal checkpoint step mismatch: {checkpoint_step} != {EXPECTED_STEP}"
        )
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise RuntimeError(f"formal checkpoint is missing or empty: {checkpoint}")
    if int(formal.get("checkpoint_bytes", -1)) != checkpoint.stat().st_size:
        raise RuntimeError("formal checkpoint byte count changed")
    events = terminal.get("failure", {}).get("detail", "")
    if "oom0" not in events or "oom-kill0" not in events:
        raise RuntimeError("source terminal does not preserve the no-OOM evidence")


def reusable_stage(name: str, source_path: Path, source: dict) -> dict:
    return {
        **source,
        "stage": name,
        "status": "done",
        "return_code": 0,
        "reuse_mode": "validated_immutable_source_stage",
        "source_stage": str(source_path.resolve()),
        "source_stage_sha256": sha256_file(source_path),
        "source_status": source.get("status"),
        "verified_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    args = parser.parse_args()

    for name in ("raw_prepare", "preprocess", "sky_masks"):
        require_status(args.run_dir / "stages" / f"{name}.json", {"done"})

    source_rows: dict[str, tuple[Path, dict]] = {}
    for name in SOURCE_SUCCESS_STAGES:
        path = args.source_run / "stages" / f"{name}.json"
        source_rows[name] = (path, require_status(path, {"available", "done"}))

    formal_path = args.source_run / "stages" / "train_formal.json"
    formal = require_status(formal_path, {"blocked"})
    terminal_path = args.source_run / "terminal.json"
    terminal = require_status(terminal_path, {"blocked"})
    checkpoint = Path(str(formal.get("checkpoint", "")))
    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    checkpoint_step = int(checkpoint_payload.get("step", -1))
    del checkpoint_payload
    validate_recovery_contract(
        formal=formal,
        terminal=terminal,
        checkpoint=checkpoint,
        checkpoint_step=checkpoint_step,
    )

    for name, (source_path, source) in source_rows.items():
        atomic_json(
            args.run_dir / "stages" / f"{name}.json",
            reusable_stage(name, source_path, source),
        )

    verified_at = dt.datetime.now(dt.timezone.utc).astimezone().isoformat()
    checkpoint_sha256 = sha256_file(checkpoint)
    recovery = {
        "stage": "formal_checkpoint_recovery",
        "status": "done",
        "return_code": 0,
        "recovery_mode": "completed_step30000_checkpoint_after_post_render_guard",
        "source_run": str(args.source_run.resolve()),
        "source_formal_stage": str(formal_path.resolve()),
        "source_formal_stage_sha256": sha256_file(formal_path),
        "source_terminal": str(terminal_path.resolve()),
        "source_terminal_sha256": sha256_file(terminal_path),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_step": checkpoint_step,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": checkpoint_sha256,
        "training_completed": True,
        "source_upstream_return_code": formal.get("return_code"),
        "source_upstream_full_render_completed": False,
        "source_upstream_full_render_frames": 577,
        "source_guard_stop_reason": formal.get("stop_reason"),
        "oom": 0,
        "oom_kill": 0,
        "verified_at": verified_at,
    }
    atomic_json(
        args.run_dir / "stages" / "formal_checkpoint_recovery.json", recovery
    )
    formal_reuse = {
        **recovery,
        "stage": "train_formal",
        "status": "done",
        "checkpoint_step": checkpoint_step,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": checkpoint_sha256,
        "semantic_scope": "training/checkpoint only; upstream accumulated full render remains blocked in source run",
    }
    atomic_json(args.run_dir / "stages" / "train_formal.json", formal_reuse)
    print(
        json.dumps(
            {
                "status": "done",
                "checkpoint": str(checkpoint),
                "checkpoint_step": checkpoint_step,
                "checkpoint_sha256": checkpoint_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

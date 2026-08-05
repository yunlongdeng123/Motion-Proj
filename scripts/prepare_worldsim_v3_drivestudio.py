#!/usr/bin/env python
"""Create or verify the patched DriveStudio worktree used by WorldSim V3."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


EXPECTED_UPSTREAM_COMMIT = "e59bda4fa681f829dbb1d65f0de582b0f633c450"
EXPECTED_STATUS = "M datasets/driving_dataset.py"


def run(*command: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_patched_tree(destination: Path, patch: Path) -> dict[str, str]:
    head = run("git", "rev-parse", "HEAD", cwd=destination)
    if head != EXPECTED_UPSTREAM_COMMIT:
        raise RuntimeError(f"DriveStudio commit drift: {head}")
    status = run("git", "status", "--short", cwd=destination)
    if status != EXPECTED_STATUS:
        raise RuntimeError(f"unexpected patched worktree status: {status!r}")
    run("git", "diff", "--check", cwd=destination)
    subprocess.run(
        ["git", "apply", "--reverse", "--check", str(patch)],
        cwd=destination,
        check=True,
    )
    source = (destination / "datasets/driving_dataset.py").read_text(
        encoding="utf-8"
    )
    required = (
        "from motion_proj.worldsim_v3.drivestudio_compat import (",
        "points, colors = concatenate_paired_lidar_chunks(",
    )
    if not all(marker in source for marker in required):
        raise RuntimeError("DriveStudio compatibility markers are missing")
    return {
        "destination": str(destination),
        "upstream_commit": head,
        "patch": str(patch),
        "patch_sha256": sha256_file(patch),
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/root/autodl-tmp/third_party/drivestudio"),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(
            "/root/autodl-tmp/third_party/drivestudio-worldsim-v3-r2"
        ),
    )
    parser.add_argument(
        "--patch",
        type=Path,
        default=Path(
            "/root/autodl-tmp/motion_proj/compatibility/"
            "DriveStudio-2026-08-05.patch"
        ),
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if not args.patch.is_file():
        raise FileNotFoundError(args.patch)
    if args.verify_only:
        if not args.destination.is_dir():
            raise FileNotFoundError(args.destination)
    else:
        if args.destination.exists():
            raise FileExistsError(
                f"refusing to overwrite existing destination: {args.destination}"
            )
        source_head = run("git", "rev-parse", "HEAD", cwd=args.source)
        if source_head != EXPECTED_UPSTREAM_COMMIT:
            raise RuntimeError(f"source DriveStudio commit drift: {source_head}")
        source_status = run("git", "status", "--short", cwd=args.source)
        if source_status:
            raise RuntimeError(f"source DriveStudio is dirty: {source_status!r}")
        args.destination.parent.mkdir(parents=True, exist_ok=True)
        run(
            "git",
            "clone",
            "--shared",
            "--no-checkout",
            str(args.source),
            str(args.destination),
        )
        run(
            "git",
            "checkout",
            "--detach",
            EXPECTED_UPSTREAM_COMMIT,
            cwd=args.destination,
        )
        subprocess.run(
            ["git", "apply", "--check", str(args.patch)],
            cwd=args.destination,
            check=True,
        )
        subprocess.run(
            ["git", "apply", str(args.patch)],
            cwd=args.destination,
            check=True,
        )

    print(
        json.dumps(
            verify_patched_tree(args.destination, args.patch),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

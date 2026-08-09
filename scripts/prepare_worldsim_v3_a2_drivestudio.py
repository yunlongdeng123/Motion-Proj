#!/usr/bin/env python
"""创建或核验带 A2 ancestry instrumentation 的 DriveStudio 工作树。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


EXPECTED_UPSTREAM_COMMIT = "e59bda4fa681f829dbb1d65f0de582b0f633c450"
EXPECTED_STATUS = {
    "M datasets/driving_dataset.py",
    "M models/gaussians/vanilla.py",
    "M models/nodes/rigid.py",
    "M models/trainers/scene_graph.py",
    "M models/trainers/single.py",
}


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


def verify_patched_tree(
    destination: Path,
    compatibility_patch: Path,
    instrumentation_patch: Path,
) -> dict[str, object]:
    head = run("git", "rev-parse", "HEAD", cwd=destination)
    if head != EXPECTED_UPSTREAM_COMMIT:
        raise RuntimeError(f"DriveStudio commit drift: {head}")
    status = {
        line.strip()
        for line in run("git", "status", "--short", cwd=destination).splitlines()
        if line.strip()
    }
    if status != EXPECTED_STATUS:
        raise RuntimeError(f"unexpected patched worktree status: {sorted(status)!r}")
    run("git", "diff", "--check", cwd=destination)
    subprocess.run(
        [
            "git",
            "apply",
            "--unidiff-zero",
            "--ignore-space-change",
            "--reverse",
            "--check",
            str(instrumentation_patch),
        ],
        cwd=destination,
        check=True,
    )

    required_markers = {
        "datasets/driving_dataset.py": (
            "seed_metadata: Dict[str, Tensor] = None",
            '"metadata": filtered_metadata',
        ),
        "models/gaussians/vanilla.py": (
            "GaussianAncestryLedger",
            'state_dict["worldsim_a2_ancestry"]',
            "record_a2_diagnostics",
        ),
        "models/nodes/rigid.py": (
            "InitSource.LIDAR",
            "append_external_clones",
        ),
        "models/trainers/scene_graph.py": (
            "InitSource.RANDOM_NEAR",
            "InitSource.RANDOM_FAR",
        ),
        "models/trainers/single.py": (
            "init_sources=sampled_sources",
        ),
    }
    for relative, markers in required_markers.items():
        source = (destination / relative).read_text(encoding="utf-8")
        if not all(marker in source for marker in markers):
            raise RuntimeError(f"A2 instrumentation markers are missing: {relative}")

    return {
        "destination": str(destination),
        "upstream_commit": head,
        "compatibility_patch": str(compatibility_patch),
        "compatibility_patch_sha256": sha256_file(compatibility_patch),
        "instrumentation_patch": str(instrumentation_patch),
        "instrumentation_patch_sha256": sha256_file(instrumentation_patch),
        "status": sorted(status),
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
            "/root/autodl-tmp/third_party/drivestudio-worldsim-v3-a2-r5"
        ),
    )
    parser.add_argument(
        "--compatibility-patch",
        type=Path,
        default=Path(
            "/root/autodl-tmp/motion_proj/compatibility/"
            "DriveStudio-2026-08-05.patch"
        ),
    )
    parser.add_argument(
        "--instrumentation-patch",
        type=Path,
        default=Path(
            "/root/autodl-tmp/motion_proj/compatibility/"
            "DriveStudio-WorldSim-A2-ancestry-v1.patch"
        ),
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    for patch in (args.compatibility_patch, args.instrumentation_patch):
        if not patch.is_file():
            raise FileNotFoundError(patch)
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
        for patch in (args.compatibility_patch,):
            subprocess.run(
                ["git", "apply", "--check", str(patch)],
                cwd=args.destination,
                check=True,
            )
            subprocess.run(
                ["git", "apply", str(patch)],
                cwd=args.destination,
                check=True,
            )
        subprocess.run(
            [
                "git",
                "apply",
                "--unidiff-zero",
                "--ignore-space-change",
                "--check",
                str(args.instrumentation_patch),
            ],
            cwd=args.destination,
            check=True,
        )
        subprocess.run(
            [
                "git",
                "apply",
                "--unidiff-zero",
                "--ignore-space-change",
                str(args.instrumentation_patch),
            ],
            cwd=args.destination,
            check=True,
        )

    print(
        json.dumps(
            verify_patched_tree(
                args.destination,
                args.compatibility_patch,
                args.instrumentation_patch,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

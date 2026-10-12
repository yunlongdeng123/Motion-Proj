#!/usr/bin/env python3
"""Remove only untracked Python bytecode created by our baseline import probes."""
from pathlib import Path
import subprocess

root = Path("/root/autodl-tmp/drivingeditor_repro/DrivingEditor").resolve()
entries = subprocess.check_output(["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"]).decode().split("\0")
removed = []
for relative in entries:
    if not relative:
        continue
    target = root / relative
    if target.suffix == ".pyc" and "__pycache__" in target.parts and not target.is_symlink():
        assert target.resolve().is_relative_to(root)
        target.unlink()
        removed.append(relative)
print(f"Removed {len(removed)} untracked runtime bytecode files; tracked files untouched")
subprocess.run(["git", "-C", str(root), "status", "--short"], check=True)

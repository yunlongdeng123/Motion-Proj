"""生成并检查 Git 源码 ZIP；用于本地和 CI，大小上限为十进制 100 MB。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile


def check(ref: str, output: Path, max_bytes: int) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "archive", "--format=zip", f"--output={output.resolve()}", ref], check=True)
    with zipfile.ZipFile(output) as archive:
        result = {
            "ref": ref, "zip_bytes": output.stat().st_size,
            "uncompressed_bytes": sum(x.file_size for x in archive.infolist()),
            "entries": len(archive.infolist()), "limit_bytes": max_bytes,
        }
    result["passed"] = result["zip_bytes"] <= max_bytes
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-bytes", type=int, default=100_000_000)
    args = parser.parse_args()
    if args.output:
        result = check(args.ref, args.output, args.max_bytes)
    else:
        with tempfile.TemporaryDirectory(prefix="motionproj-archive-") as directory:
            result = check(args.ref, Path(directory) / "source.zip", args.max_bytes)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit("源码 ZIP 超过上限，拒绝交付。")


if __name__ == "__main__":
    main()

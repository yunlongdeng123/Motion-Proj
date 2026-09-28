"""Extract and decode exactly the RGB files in the frozen DELETE audit.

Existing per-scene LiDAR/RGB archive manifests nominate likely shards. If a
log crosses a shard boundary, the requested JPEG is sought in both shards.
Only the 1,802 selected files are materialized; old data is untouched.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image


MANIFESTS = [
    "/root/autodl-tmp/data/worldsim_v67/*member_shards*.json",
    "/root/autodl-tmp/data/worldsim_v5/manifests/*member_shards*.json",
    "/root/autodl-tmp/data/dynamic_editing_v2/manifests/*member_shards*.json",
]


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def normalize_shard(value):
    value = str(value)
    if value.isdigit():
        return f"{int(value):02d}"
    return value.split("trainval", 1)[1][:2]


def extract_one(spec):
    shard, archive, members_file, output = spec
    t0 = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([
        "tar", "-xzf", str(archive), "-C", str(output),
        "--no-same-owner", "--skip-old-files", "-T", str(members_file),
    ], text=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    stderr = proc.stderr.strip().splitlines()
    return {"shard": shard, "archive": str(archive), "requested": len(members_file.read_text().splitlines()),
            "exit_code": proc.returncode, "seconds": round(time.monotonic() - t0, 2),
            "stderr_tail": stderr[-4:]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--expected", type=Path, required=True)
    p.add_argument("--pub", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()
    result_path = args.root / "pub_rgb_check.json"
    assert not result_path.exists()
    expected = json.loads(args.expected.read_text())
    required = {f["filename"] for c in expected["clips"] for f in c["frames"]}
    assert len(required) == expected["unique_rgb_file_count"]
    for name in required:
        path = Path(name)
        assert (name.startswith("samples/CAM_") or name.startswith("sweeps/CAM_")) and path.suffix.lower() == ".jpg"
        assert not path.is_absolute() and ".." not in path.parts
    by_prefix = defaultdict(set)
    for pattern in MANIFESTS:
        for manifest in glob.glob(pattern):
            rows = json.loads(Path(manifest).read_text())
            for name, shard in rows.items():
                by_prefix[Path(name).name.split("__", 1)[0]].add(normalize_shard(shard))
    groups = defaultdict(list)
    unmapped = []
    for name in sorted(required):
        shards = by_prefix.get(Path(name).name.split("__", 1)[0], set())
        if not shards:
            unmapped.append(name)
        for shard in sorted(shards):
            groups[shard].append(name)
    assert not unmapped, f"No shard clue for {len(unmapped)} selected JPEGs"
    plan = {"task_id": expected["task_id"], "run_id": expected["run_id"],
            "required_unique_rgb": len(required),
            "requested_by_shard": {k: len(v) for k, v in sorted(groups.items())},
            "ambiguous_shard_files": sum(len(by_prefix[Path(name).name.split("__", 1)[0]]) > 1 for name in required),
            "mapping_basis": "existing per-scene public-archive manifests with matching source log prefix; actual extraction and decode determine availability"}
    write(args.root / "pub_rgb_extract_plan.json", plan)
    print("RGB_PLAN", plan["requested_by_shard"], flush=True)
    member_dir = args.root / "pub_member_lists"
    member_dir.mkdir(exist_ok=True)
    specs = []
    for shard, names in sorted(groups.items()):
        list_path = member_dir / f"{shard}.txt"
        list_path.write_text("\n".join(names) + "\n")
        archive = args.pub / f"v1.0-trainval{shard}_blobs.tgz"
        assert archive.is_file()
        specs.append((shard, archive, list_path, args.root / "pub_by_shard" / shard))
    progress_path = args.root / "pub_rgb_extract_progress.json"
    scans = json.loads(progress_path.read_text())["finished"] if progress_path.exists() else []
    completed_shards = {row["shard"] for row in scans}
    specs = [spec for spec in specs if spec[0] not in completed_shards]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(extract_one, spec) for spec in specs]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            scans.append(row)
            write(progress_path, {"finished": scans, "shards_total": len(groups)})
            print("RGB_SHARD", row["shard"], "SECONDS", row["seconds"],
                  "EXIT", row["exit_code"], flush=True)

    decoded = []
    missing = []
    corrupt = []
    duplicates = []
    rgb_root = args.root / "rgb"
    for name in sorted(required):
        places = [args.root / "pub_by_shard" / shard / name for shard in groups if
                  (args.root / "pub_by_shard" / shard / name).is_file()]
        if not places:
            missing.append(name)
            continue
        if len(places) != 1:
            duplicates.append({"filename": name, "shards": [str(p.relative_to(args.root)) for p in places]})
            continue
        source = places[0]
        try:
            with Image.open(source) as image:
                image.verify()
            with Image.open(source) as image:
                width, height = image.size
                assert (width, height) == (1600, 900)
        except Exception as exc:
            corrupt.append({"filename": name, "reason": repr(exc)})
            continue
        alias = rgb_root / name
        alias.parent.mkdir(parents=True, exist_ok=True)
        if not alias.exists():
            alias.symlink_to(source)
        decoded.append({"filename": name, "source_shard": source.relative_to(args.root).parts[1],
                        "width": width, "height": height, "bytes": source.stat().st_size})
    result = {"task_id": expected["task_id"], "run_id": expected["run_id"],
              "public_archive_root": str(args.pub), "required_unique_rgb": len(required),
              "extracted_and_decoded": len(decoded), "missing": missing, "corrupt": corrupt,
              "duplicates": duplicates, "all_selected_rgb_available": not (missing or corrupt or duplicates),
              "scan_records": sorted(scans, key=lambda x: x["shard"]),
              "timing_gaps_gt65ms": expected["missing_or_large_gap"],
              "files": decoded, "human_verdict": None}
    write(result_path, result)
    print("PUB_RGB_CHECK", len(decoded), "MISSING", len(missing),
          "CORRUPT", len(corrupt), "DUPLICATE", len(duplicates), flush=True)
    if not result["all_selected_rgb_available"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

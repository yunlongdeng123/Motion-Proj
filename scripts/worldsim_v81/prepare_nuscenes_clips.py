"""从官方 nuScenes 元数据构建连续 25 帧 CAM_FRONT P0 原图片段。"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from bisect import bisect_left
import json
import os
from pathlib import Path
import re
import statistics
import tarfile
import threading

import ijson
from PIL import Image
from nuscenes.utils.splits import create_splits_scenes


META_DEFAULT = Path("/root/autodl-tmp/runs/worldsim_v77/WS-V77-DELETE-AUDIT-20260928/r1/metadata/v1.0-trainval")
ARCHIVE_DEFAULT = Path("/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval")
DATA_DEFAULT = Path("/root/autodl-tmp/data/worldsim_v81/nuscenes_p0")
RUN_DEFAULT = Path("/root/autodl-tmp/runs/worldsim_v81/WS-V81-SEEN-TO-SCENE-20261009/r1/data")
RUNS_V77 = Path("/root/autodl-tmp/runs/worldsim_v77")
FRAMES = 25


def load_json(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def camera_records(metadata_dir):
    records = {}
    with (metadata_dir / "sample_data.json").open("rb") as stream:
        for row in ijson.items(stream, "item"):
            name = row["filename"]
            if name.startswith(("samples/CAM_FRONT/", "sweeps/CAM_FRONT/")) and "__CAM_FRONT__" in name and name.endswith(".jpg"):
                records[row["token"]] = {
                    key: row[key] for key in ("token", "sample_token", "timestamp", "filename",
                                              "prev", "next", "is_key_frame", "height", "width")
                }
    return records


def cache_files(root=RUNS_V77):
    """收集原名文件；实际进入清单前仍须逐帧解码并核对原尺寸。"""
    mapping = {}
    for directory, _dirs, _files in os.walk(root):
        directory_path = Path(directory)
        if directory_path.name != "CAM_FRONT" or directory_path.parent.name not in {"samples", "sweeps"}:
            continue
        if not any(part in {"rgb", "by_shard", "pub_by_shard", "extra_inputs"}
                   for part in directory_path.parts):
            continue
        kind = directory_path.parent.name
        for entry in os.scandir(directory):
            if not entry.name.endswith(".jpg") or "__CAM_FRONT__" not in entry.name:
                continue
            if not entry.is_file(follow_symlinks=True):
                continue
            relative = f"{kind}/CAM_FRONT/{entry.name}"
            mapping.setdefault(relative, str(directory_path / entry.name))
    return mapping


def source_prefix(filename):
    return Path(filename).name.split("__", 1)[0]


def source_stamp(filename):
    match = re.search(r"__(\d{16})(?:\.|$)", Path(filename).name)
    if not match:
        raise ValueError("filename lacks nuScenes timestamp")
    return int(match.group(1))


def normalize_archive(value):
    return f"v1.0-trainval{int(value):02d}_blobs.tgz" if value.isdigit() else value


def shard_index():
    exact = {}
    prefixes = defaultdict(list)
    bases = (Path("/root/autodl-tmp/data/worldsim_v67"),
             Path("/root/autodl-tmp/data/dynamic_editing_v2/manifests"))
    for base in bases:
        for path in base.glob("*member_shards.json"):
            for member, archive in load_json(path).items():
                archive = normalize_archive(archive)
                exact[member] = archive
                try:
                    prefixes[source_prefix(member)].append((source_stamp(member), archive))
                except (IndexError, ValueError):
                    pass
    for values in prefixes.values():
        values.sort()
    return exact, prefixes


def infer_archive(filename, exact, prefixes):
    if filename in exact:
        return exact[filename]
    values = prefixes.get(source_prefix(filename))
    if not values:
        return None
    stamp = source_stamp(filename)
    index = bisect_left(values, (stamp, ""))
    nearby = values[max(0, index - 2):min(len(values), index + 2)]
    return min(nearby, key=lambda pair: abs(pair[0] - stamp))[1]


def build_candidate(start, records, sample_scene):
    first_scene = sample_scene.get(start["sample_token"])
    if first_scene is None:
        return None
    chain = [start]
    for _ in range(FRAMES - 1):
        nxt = records.get(chain[-1]["next"])
        if nxt is None or nxt["prev"] != chain[-1]["token"]:
            return None
        if sample_scene.get(nxt["sample_token"]) != first_scene:
            return None
        chain.append(nxt)
    stamps = [row["timestamp"] for row in chain]
    intervals = [b - a for a, b in zip(stamps, stamps[1:])]
    median_us = statistics.median(intervals)
    if min(intervals) <= 0 or max(intervals) >= 150_000 or not 60_000 <= median_us <= 110_000:
        return None
    if sum(not row["is_key_frame"] for row in chain) <= FRAMES // 2:
        return None
    if any(row["width"] != 1600 or row["height"] != 900 for row in chain):
        return None
    return {"scene_token": first_scene, "frames": chain, "timestamps_us": stamps,
            "median_dt_us": median_us, "mean_dt_us": statistics.mean(intervals),
            "max_dt_us": max(intervals)}


def candidates(records, sample_scene, scenes, cache):
    split_names = create_splits_scenes()
    split_by_name = {name: split for split in ("train", "val") for name in split_names[split]}
    scene_by_token = {row["token"]: row for row in scenes}
    best = {}
    for record in records.values():
        scene_token = sample_scene.get(record["sample_token"])
        scene = scene_by_token.get(scene_token)
        if not scene or scene["name"] not in split_by_name:
            continue
        candidate = build_candidate(record, records, sample_scene)
        if candidate is None:
            continue
        candidate["scene_name"] = scene["name"]
        candidate["split"] = split_by_name[scene["name"]]
        candidate["cached"] = sum(row["filename"] in cache for row in candidate["frames"])
        previous = best.get(scene_token)
        if previous is None or (candidate["cached"], -candidate["max_dt_us"]) > (previous["cached"], -previous["max_dt_us"]):
            best[scene_token] = candidate
    return sorted(best.values(), key=lambda row: (-row["cached"], row["max_dt_us"], row["scene_name"]))


def verify_rgb(path):
    try:
        with Image.open(path) as image:
            if image.size != (1600, 900) or image.mode != "RGB":
                return False
            image.load()
        return True
    except (OSError, ValueError):
        return False


def choose_clips(choices, cache, exact, prefixes, train_count=6, val_count=2):
    selected = []
    for split, count in (("train", train_count), ("val", val_count)):
        eligible = [row for row in choices if row["split"] == split and all(
            frame["filename"] in cache or infer_archive(frame["filename"], exact, prefixes)
            for frame in row["frames"])]
        # 每个 split 尽可能只解压一个分片，同时优先已有原图。
        primary_archive = None
        for row in eligible:
            missing = [frame["filename"] for frame in row["frames"] if frame["filename"] not in cache]
            archives = {infer_archive(name, exact, prefixes) for name in missing}
            if len(archives) != 1:
                continue
            archive = next(iter(archives))
            if primary_archive is None:
                primary_archive = archive
            if archive == primary_archive:
                selected.append(row)
            if sum(item["split"] == split for item in selected) == count:
                break
        if sum(item["split"] == split for item in selected) < count:
            raise RuntimeError(f"not enough mapped {split} scenes")
    return selected


def write_json_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def materialize(selected, cache, exact, prefixes, archive_dir, data_dir, run_dir,
                write_status=True):
    data_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    needed = defaultdict(set)
    origin = {}
    for clip in selected:
        for row in clip["frames"]:
            name = row["filename"]
            if name in paths:
                continue
            cached = cache.get(name)
            if cached and verify_rgb(cached):
                paths[name] = cached
                origin[name] = "existing_original_cache"
                continue
            extracted = data_dir / name
            if extracted.is_file() and verify_rgb(extracted):
                paths[name] = str(extracted)
                origin[name] = "selected_archive_member"
                continue
            archive = infer_archive(name, exact, prefixes)
            if not archive:
                raise RuntimeError(f"cannot infer archive: {name}")
            needed[archive].add(name)
    lock = threading.Lock()
    emitted = set()

    def commit_ready():
        with lock:
            for split in ("train", "val"):
                ready = [clip for clip in selected if clip["split"] == split and all(
                    row["filename"] in paths for row in clip["frames"])]
                new = [clip for clip in ready if clip["scene_name"] not in emitted]
                if not new:
                    continue
                rows = []
                metadata = []
                for clip in ready:
                    video_id = f"nuscenes-{clip['scene_name']}-cam-front-{clip['timestamps_us'][0]}"
                    names = [frame["filename"] for frame in clip["frames"]]
                    rows.append({"video_id": video_id, "split": split,
                                 "frames": [paths[name] for name in names]})
                    metadata.append({"video_id": video_id, "scene_name": clip["scene_name"],
                                     "scene_token": clip["scene_token"],
                                     "sample_data_tokens": [frame["token"] for frame in clip["frames"]],
                                     "timestamps_us": clip["timestamps_us"],
                                     "is_key_frame": [frame["is_key_frame"] for frame in clip["frames"]],
                                     "source_filenames": names,
                                     "source_origin": [origin[name] for name in names],
                                     "mean_fps": 1_000_000 / clip["mean_dt_us"],
                                     "mean_dt_us": clip["mean_dt_us"],
                                     "median_dt_us": clip["median_dt_us"],
                                     "max_dt_us": clip["max_dt_us"],
                                     "sweep_count": sum(not frame["is_key_frame"] for frame in clip["frames"]),
                                     "frames": FRAMES, "width": 1600, "height": 900})
                manifest = run_dir / f"{split}.jsonl"
                temporary = manifest.with_name(manifest.name + ".tmp")
                temporary.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                os.replace(temporary, manifest)
                write_json_atomic(run_dir / f"{split}_metadata.json", metadata)
                for clip in new:
                    emitted.add(clip["scene_name"])
                    print(f"READY {split} {clip['scene_name']} {manifest}", flush=True)
            if write_status:
                write_json_atomic(run_dir / "preparation_status.json", {
                    "selected": [clip["scene_name"] for clip in selected],
                    "ready": sorted(emitted),
                    "missing_members": {archive: len(members) for archive, members in needed.items()}})

    commit_ready()

    def extract_archive(archive, members):
        archive_path = archive_dir / archive
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        pending = set(members)
        print(f"SCAN {archive} selected_members={len(pending)}", flush=True)
        with tarfile.open(archive_path, "r|gz") as stream:
            for member in stream:
                name = member.name.removeprefix("./")
                if name not in pending or not member.isfile():
                    continue
                target = data_dir / name
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".tmp")
                source = stream.extractfile(member)
                if source is None:
                    continue
                with source, temporary.open("wb") as sink:
                    while block := source.read(1024 * 1024):
                        sink.write(block)
                if not verify_rgb(temporary):
                    temporary.unlink(missing_ok=True)
                    raise RuntimeError(f"archive member is not native RGB: {name}")
                os.replace(temporary, target)
                with lock:
                    paths[name] = str(target)
                    origin[name] = f"{archive}:{name}"
                    pending.remove(name)
                    needed[archive].remove(name)
                print(f"FOUND {archive} remaining={len(pending)} {name}", flush=True)
                commit_ready()
                if not pending:
                    break
        if pending:
            raise RuntimeError(f"missing {len(pending)} selected members in {archive}: {sorted(pending)[:3]}")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(extract_archive, archive, members)
                   for archive, members in needed.items() if members]
        for future in as_completed(futures):
            future.result()
    commit_ready()
    if len(emitted) != len(selected):
        raise RuntimeError(f"only {len(emitted)}/{len(selected)} clips ready")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=META_DEFAULT)
    parser.add_argument("--archive-dir", type=Path, default=ARCHIVE_DEFAULT)
    parser.add_argument("--data-dir", type=Path, default=DATA_DEFAULT)
    parser.add_argument("--run-dir", type=Path, default=RUN_DEFAULT)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--cache-only-val", action="store_true")
    args = parser.parse_args()
    scenes = load_json(args.metadata_dir / "scene.json")
    samples = load_json(args.metadata_dir / "sample.json")
    sample_scene = {row["token"]: row["scene_token"] for row in samples}
    print("streaming sample_data.json", flush=True)
    records = camera_records(args.metadata_dir)
    print(f"CAM_FRONT records: {len(records)}", flush=True)
    cache = cache_files()
    print(f"cached original-name images: {len(cache)}", flush=True)
    exact, prefixes = shard_index()
    choices = candidates(records, sample_scene, scenes, cache)
    for split in ("train", "val"):
        rows = [row for row in choices if row["split"] == split]
        mapped = [row for row in rows if all(
            frame["filename"] in cache or infer_archive(frame["filename"], exact, prefixes)
            for frame in row["frames"])]
        print(json.dumps({"split": split, "mapped_top": [
            {"scene": row["scene_name"], "cached": row["cached"],
             "archives": sorted({infer_archive(frame["filename"], exact, prefixes)
                                 for frame in row["frames"] if frame["filename"] not in cache})}
            for row in mapped[:10]]}), flush=True)
        print(json.dumps({"split": split, "eligible_scenes": len(rows), "top": [
            {"scene": row["scene_name"], "cached": row["cached"],
             "median_dt_us": row["median_dt_us"], "max_dt_us": row["max_dt_us"],
             "first": row["frames"][0]["filename"],
             "missing": [{"file": frame["filename"],
                          "archive": infer_archive(frame["filename"], exact, prefixes)}
                         for frame in row["frames"] if frame["filename"] not in cache]}
            for row in rows[:8]]}), flush=True)
    if args.cache_only_val:
        for candidate in choices:
            if candidate["split"] != "val" or candidate["cached"] != FRAMES:
                continue
            if all(verify_rgb(cache[row["filename"]]) for row in candidate["frames"]):
                print(f"CACHE_ONLY_VAL {candidate['scene_name']}", flush=True)
                materialize([candidate], cache, exact, prefixes, args.archive_dir,
                            args.data_dir, args.run_dir, write_status=False)
                return
        raise SystemExit("no fully cached and decodable val clip")
    if not args.plan_only:
        selected = choose_clips(choices, cache, exact, prefixes)
        print(json.dumps({"selected": [{"split": clip["split"], "scene": clip["scene_name"],
                                        "cached": clip["cached"]} for clip in selected]}), flush=True)
        materialize(selected, cache, exact, prefixes, args.archive_dir, args.data_dir, args.run_dir)


if __name__ == "__main__":
    main()

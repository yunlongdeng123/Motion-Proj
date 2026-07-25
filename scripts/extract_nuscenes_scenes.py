#!/usr/bin/env python
"""按 scene 从本地 nuScenes trainval blob 归档中选择性抽取指定通道的全量帧。

设计目标（对应 N2 raw-evidence 前置的"只抽目标 scene"约束）：

- 接口动态读取 scene 名单（``--scene-list`` 指向的文本文件，每行一个 scene 名或 token，
  支持 ``#`` 注释与空行）；不把 scene 写死在代码里。
- 默认只抽 ``CAM_FRONT`` 与 ``LIDAR_TOP``，且同时包含关键帧（``samples/``）与中间帧
  （``sweeps/``）——即"视频与雷达包"的全量帧。
- 数据源是本地已挂载的 AutoDL 公共归档 ``*.tgz``，不联网下载。
- 只从命中目标 scene 的 shard 里解压目标成员，避免铺开全量 305GB。

用法::

    # 首次或归档变更后：建立 filename->shard 索引（需顺序解压全部 shard，较慢，只做一次）
    python scripts/extract_nuscenes_scenes.py --build-index

    # 按名单抽取（缺索引时会自动构建）
    python scripts/extract_nuscenes_scenes.py \
        --scene-list configs/resim/n2_extract_scene_list.example.txt

索引与状态写入 ``<dst>/.trainval-scene-extract/``；抽取完成后输出逐 scene / 逐通道 manifest 并做完整性校验。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import ijson

DEFAULT_SRC = Path("/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval")
DEFAULT_DST = Path("/root/autodl-tmp/data/nuscenes")
DEFAULT_CHANNELS = ("CAM_FRONT", "LIDAR_TOP")
SHARDS = tuple(f"{i:02d}" for i in range(1, 11))


def _load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_scene_list(path: Path) -> list[str]:
    entries = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            entries.append(line)
    if not entries:
        raise ValueError(f"scene 名单为空: {path}")
    return entries


def _resolve_scenes(meta: Path, requested: list[str]) -> list[dict]:
    scenes = _load_json(meta / "scene.json")
    by_name = {row["name"]: row for row in scenes}
    by_token = {row["token"]: row for row in scenes}
    resolved = []
    missing = []
    for entry in requested:
        row = by_name.get(entry) or by_token.get(entry)
        if row is None:
            missing.append(entry)
        else:
            resolved.append(row)
    if missing:
        raise ValueError(f"scene 名单中无法解析: {missing}")
    return resolved


def _channel_by_sd_token(meta: Path) -> dict[str, str]:
    sensors = {row["token"]: row["channel"] for row in _load_json(meta / "sensor.json")}
    calibrated = {
        row["token"]: sensors[row["sensor_token"]]
        for row in _load_json(meta / "calibrated_sensor.json")
    }
    return calibrated


def _scene_sample_tokens(meta: Path, scenes: list[dict]) -> dict[str, str]:
    """返回 sample_token -> scene_token（只覆盖目标 scene）。"""
    wanted = {row["token"] for row in scenes}
    mapping = {}
    for sample in _load_json(meta / "sample.json"):
        if sample["scene_token"] in wanted:
            mapping[sample["token"]] = sample["scene_token"]
    return mapping


def _target_files(
    meta: Path, scenes: list[dict], channels: tuple[str, ...], include_sweeps: bool
) -> dict[str, dict[str, list[str]]]:
    """计算 scene_token -> channel -> [filename] 的精确抽取清单。

    ``sample_data.json`` 达 GB 级，在 2GB cgroup 下必须流式解析，不能一次性 load。
    """
    calibrated = _channel_by_sd_token(meta)
    sample_to_scene = _scene_sample_tokens(meta, scenes)
    channel_set = set(channels)
    result: dict[str, dict[str, list[str]]] = {
        row["token"]: {channel: [] for channel in channels} for row in scenes
    }
    with (meta / "sample_data.json").open("rb") as handle:
        for row in ijson.items(handle, "item"):
            scene_token = sample_to_scene.get(row["sample_token"])
            if scene_token is None:
                continue
            channel = calibrated[row["calibrated_sensor_token"]]
            if channel not in channel_set:
                continue
            if not row["is_key_frame"] and not include_sweeps:
                continue
            result[scene_token][channel].append(row["filename"])
    return result


def _index_path(state_dir: Path) -> Path:
    return state_dir / "filename_to_shard.json"


def build_index(src: Path, state_dir: Path) -> dict[str, str]:
    """顺序列出每个 shard 的成员，构建 filename->shard 映射并缓存。"""
    state_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, str] = {}
    for shard in SHARDS:
        archive = src / f"v1.0-trainval{shard}_blobs.tgz"
        if not archive.is_file():
            raise FileNotFoundError(archive)
        print(f"[index] 列出 shard {shard} 成员（解压扫描，较慢）...", flush=True)
        proc = subprocess.Popen(
            ["tar", "-tzf", str(archive)],
            stdout=subprocess.PIPE,
            text=True,
        )
        assert proc.stdout is not None
        count = 0
        for member in proc.stdout:
            name = member.strip()
            if name and not name.endswith("/"):
                index[name] = shard
                count += 1
        if proc.wait() != 0:
            raise RuntimeError(f"tar -tzf 失败: {archive}")
        print(f"[index] shard {shard}: {count} 成员", flush=True)
    _index_path(state_dir).write_text(
        json.dumps(index, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[index] 写入 {_index_path(state_dir)}（{len(index)} 条）", flush=True)
    return index


def load_or_build_index(src: Path, state_dir: Path) -> dict[str, str]:
    path = _index_path(state_dir)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    print("[index] 未找到缓存索引，开始构建（一次性成本）", flush=True)
    return build_index(src, state_dir)


def _extract_from_shard(
    archive: Path, dst: Path, members: list[str]
) -> None:
    """从单个归档抽取指定成员。分批传参以避免命令行过长。"""
    batch = 2000
    for start in range(0, len(members), batch):
        chunk = members[start : start + batch]
        subprocess.run(
            [
                "tar",
                "-xzf",
                str(archive),
                "-C",
                str(dst),
                "--skip-old-files",
                *chunk,
            ],
            check=True,
        )


def extract(
    src: Path,
    dst: Path,
    scene_entries: list[str],
    channels: tuple[str, ...],
    include_sweeps: bool,
    dry_run: bool,
) -> dict:
    meta = dst / "v1.0-trainval"
    if not (meta / "scene.json").is_file():
        raise FileNotFoundError(
            f"缺少 trainval metadata: {meta}；请先解压 v1.0-trainval_meta.tgz"
        )
    state_dir = dst / ".trainval-scene-extract"
    state_dir.mkdir(parents=True, exist_ok=True)

    scenes = _resolve_scenes(meta, scene_entries)
    targets = _target_files(meta, scenes, channels, include_sweeps)

    # 汇总需要的文件与所属 shard
    all_files: list[str] = []
    for per_channel in targets.values():
        for names in per_channel.values():
            all_files.extend(names)

    plan = {
        "scenes": [{"name": row["name"], "token": row["token"]} for row in scenes],
        "channels": list(channels),
        "include_sweeps": include_sweeps,
        "file_count": len(all_files),
        "per_scene_file_count": {
            row["name"]: sum(len(v) for v in targets[row["token"]].values())
            for row in scenes
        },
    }

    # dry-run 不构建索引（索引构建需顺序解压全部 shard，成本高）；仅在索引已存在时附带 shard 分布。
    index = None
    if dry_run:
        index_path = _index_path(state_dir)
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = load_or_build_index(src, state_dir)

    if index is not None:
        missing_in_index = [name for name in all_files if name not in index]
        if missing_in_index:
            raise RuntimeError(
                f"{len(missing_in_index)} 个目标文件不在归档索引中，示例: {missing_in_index[:3]}"
            )
        by_shard: dict[str, list[str]] = {}
        for name in all_files:
            by_shard.setdefault(index[name], []).append(name)
        plan["shards"] = {shard: len(names) for shard, names in sorted(by_shard.items())}

    print(json.dumps({"extract_plan": plan}, ensure_ascii=False, indent=2))

    if dry_run:
        return {"plan": plan, "dry_run": True}

    for shard in sorted(by_shard):
        archive = src / f"v1.0-trainval{shard}_blobs.tgz"
        print(
            f"[extract] shard {shard}: 抽取 {len(by_shard[shard])} 个成员",
            flush=True,
        )
        _extract_from_shard(archive, dst, by_shard[shard])

    # 完整性校验
    manifest_scenes = []
    complete = True
    for row in scenes:
        per_channel = targets[row["token"]]
        channel_rows = {}
        for channel, names in per_channel.items():
            present = sum(1 for name in names if (dst / name).is_file())
            channel_complete = present == len(names)
            complete = complete and channel_complete
            channel_rows[channel] = {
                "expected": len(names),
                "present": present,
                "missing": len(names) - present,
                "complete": channel_complete,
                "bytes": sum(
                    (dst / name).stat().st_size
                    for name in names
                    if (dst / name).is_file()
                ),
            }
        manifest_scenes.append(
            {"name": row["name"], "token": row["token"], "channels": channel_rows}
        )

    manifest = {
        "schema_version": 1,
        "src": str(src),
        "dst": str(dst),
        "channels": list(channels),
        "include_sweeps": include_sweeps,
        "scenes": manifest_scenes,
        "complete": complete,
    }
    manifest_path = state_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(manifest_path), "complete": complete}))
    if not complete:
        raise SystemExit("ERROR: 选择性抽取不完整")
    return {"manifest": manifest, "dry_run": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    parser.add_argument(
        "--scene-list",
        type=Path,
        help="scene 名单文件；每行一个 scene 名（scene-0655）或 scene token",
    )
    parser.add_argument(
        "--channels",
        default=",".join(DEFAULT_CHANNELS),
        help="逗号分隔的通道，默认 CAM_FRONT,LIDAR_TOP",
    )
    parser.add_argument(
        "--no-sweeps",
        action="store_true",
        help="只抽关键帧 samples/，不抽 sweeps/ 中间帧",
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="仅构建 filename->shard 索引后退出",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印抽取计划，不实际解压",
    )
    args = parser.parse_args()

    state_dir = args.dst / ".trainval-scene-extract"
    if args.build_index:
        build_index(args.src, state_dir)
        return

    if not args.scene_list:
        parser.error("需要 --scene-list（或使用 --build-index）")
    channels = tuple(c.strip() for c in args.channels.split(",") if c.strip())
    scene_entries = _read_scene_list(args.scene_list)
    extract(
        args.src,
        args.dst,
        scene_entries,
        channels,
        include_sweeps=not args.no_sweeps,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

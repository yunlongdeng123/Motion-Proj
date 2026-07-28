#!/usr/bin/env python3
"""为 AD-GS 官方六场景生成精确资产清单，并从公共 tar 选择性提取。

协议（与 third_party/AD-GS/scripts/nuscene/nuscene.py 一致）：
- scenes: 0230/0242/0255/0295/0518/0749
- 从每个 scene 的 first_sample 起，沿三相机 sample_data 链表取帧 0..69，保留 10..69
- 相机: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT（上游循环顺序）
- 每帧取与 CAM_FRONT 时间戳最近的 LIDAR_TOP

输出写入 /root/autodl-tmp/data/dynamic_recon/manifests/ 与 raw_subset/。
不修改现有 /root/autodl-tmp/data/nuscenes。
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

SCENES = [
    "scene-0230",
    "scene-0242",
    "scene-0255",
    "scene-0295",
    "scene-0518",
    "scene-0749",
]
# AD-GS nuscene.py 中的相机顺序（决定 image/000000.png 起的编号）
SENSORS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]
FIRST_FRAME = 10
LAST_FRAME = 69
SHARDS = [f"{i:02d}" for i in range(1, 11)]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def get_nearest_lidar_token(sorted_lidar, timestamp, start=0, end=-1):
    if end == -1:
        end = len(sorted_lidar)
    assert end - start >= 2
    if end - start == 2:
        a, b = sorted_lidar[start], sorted_lidar[start + 1]
        return a[1] if abs(timestamp - a[0]) < abs(timestamp - b[0]) else b[1]
    mid = (start + end) // 2
    if sorted_lidar[mid][0] == timestamp:
        return sorted_lidar[mid][1]
    if sorted_lidar[mid][0] > timestamp:
        return get_nearest_lidar_token(sorted_lidar, timestamp, start, mid + 1)
    return get_nearest_lidar_token(sorted_lidar, timestamp, mid, end)


def collect_required(nusc_meta_root: Path) -> dict:
    """用与 AD-GS 相同的链表游走逻辑收集 filename。"""
    from nuscenes.nuscenes import NuScenes

    # dataroot only needs metadata; sensor files may be missing.
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(nusc_meta_root), verbose=False)
    required = []
    per_scene = {}

    for scene_name in SCENES:
        scene_token = None
        for sc in nusc.scene:
            if sc["name"] == scene_name:
                scene_token = sc["token"]
                break
        assert scene_token, scene_name
        scene = nusc.get("scene", scene_token)
        sample = nusc.get("sample", scene["first_sample_token"])

        lidar_iter = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        lidar_tokens = [(lidar_iter["timestamp"], lidar_iter)]
        while lidar_iter["next"] != "":
            lidar_iter = nusc.get("sample_data", lidar_iter["next"])
            lidar_tokens.append((lidar_iter["timestamp"], lidar_iter))
        lidar_tokens = sorted(lidar_tokens, key=lambda x: x[0])

        cameras_iter = [nusc.get("sample_data", sample["data"][cam]) for cam in SENSORS]
        scene_rows = []
        files = set()

        for idx in range(LAST_FRAME + 1):
            if idx < FIRST_FRAME:
                cameras_iter = [nusc.get("sample_data", i["next"]) for i in cameras_iter]
                continue

            lidar_sd = get_nearest_lidar_token(lidar_tokens, cameras_iter[0]["timestamp"])
            lidar_dt_ms = abs(lidar_sd["timestamp"] - cameras_iter[0]["timestamp"]) / 1000.0
            files.add(lidar_sd["filename"])

            for cam_i, data in enumerate(cameras_iter):
                files.add(data["filename"])
                scene_rows.append(
                    {
                        "frame_idx": idx,
                        "rel_frame": idx - FIRST_FRAME,
                        "camera": SENSORS[cam_i],
                        "image_id": (idx - FIRST_FRAME) * len(SENSORS) + cam_i,
                        "sample_data_token": data["token"],
                        "filename": data["filename"],
                        "timestamp": data["timestamp"],
                        "is_key_frame": data["is_key_frame"],
                        "height": data["height"],
                        "width": data["width"],
                        "lidar_filename": lidar_sd["filename"],
                        "lidar_dt_ms": lidar_dt_ms,
                    }
                )
            cameras_iter = [nusc.get("sample_data", i["next"]) for i in cameras_iter]

        file_list = sorted(files)
        per_scene[scene_name] = {
            "scene_token": scene_token,
            "n_rgb": sum(1 for f in file_list if f.startswith("samples/CAM") or f.startswith("sweeps/CAM")),
            "n_lidar": sum(1 for f in file_list if "LIDAR" in f),
            "n_files": len(file_list),
            "files": file_list,
            "frames": scene_rows,
            "expected_rgb": (LAST_FRAME - FIRST_FRAME + 1) * len(SENSORS),  # 180
        }
        for f in file_list:
            required.append({"scene": scene_name, "filename": f})

    return {"protocol": {
        "scenes": SCENES,
        "sensors": SENSORS,
        "first_frame": FIRST_FRAME,
        "last_frame": LAST_FRAME,
        "expected_rgb_per_scene": (LAST_FRAME - FIRST_FRAME + 1) * len(SENSORS),
    }, "required": required, "per_scene": per_scene}


def build_shard_index(tar_dir: Path, members: set[str], index_path: Path) -> dict[str, str]:
    """兼容旧调用；只建立索引，不执行提取。"""
    mapping, _ = scan_shards(
        tar_dir=tar_dir,
        members=members,
        index_path=index_path,
        dst=None,
        workers=2,
    )
    return mapping


def _scan_one_shard(task: tuple[str, set[str], str | None]) -> dict:
    """单次流式扫描一个 shard，并原子写出命中的成员。"""
    archive_raw, members, dst_raw = task
    archive = Path(archive_raw)
    dst = Path(dst_raw) if dst_raw else None
    found: dict[str, dict] = {}
    t0 = time.time()
    print(f"[shard] scanning {archive.name} for {len(members)} candidates", flush=True)

    with tarfile.open(archive, "r:gz") as tf:
        for info in tf:
            name = info.name[2:] if info.name.startswith("./") else info.name
            if name not in members:
                continue
            if not info.isfile():
                raise RuntimeError(f"required member 不是普通文件: {archive.name}:{info.name}")
            if name in found:
                raise RuntimeError(f"同一 shard 内 required member 重复: {archive.name}:{name}")

            extracted = False
            if dst is not None:
                target = dst / name
                if not (target.is_file() and target.stat().st_size > 0):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    src = tf.extractfile(info)
                    if src is None:
                        raise RuntimeError(f"无法读取 tar member: {archive.name}:{info.name}")
                    tmp = target.with_name(f"{target.name}.partial.{os.getpid()}")
                    try:
                        with src, tmp.open("wb") as out:
                            shutil.copyfileobj(src, out, length=1 << 20)
                        if tmp.stat().st_size != info.size:
                            raise RuntimeError(
                                f"提取字节数不匹配: {name} {tmp.stat().st_size} != {info.size}"
                            )
                        os.replace(str(tmp), str(target))
                    finally:
                        if tmp.exists():
                            tmp.unlink()
                    extracted = True

            found[name] = {
                "shard": archive.name,
                "extracted": extracted,
            }
            if len(found) == len(members):
                break

    print(
        f"[shard] done {archive.name}: found={len(found)} elapsed={time.time()-t0:.1f}s",
        flush=True,
    )
    if hasattr(os, "posix_fadvise"):
        handle = os.open(str(archive), os.O_RDONLY)
        try:
            os.posix_fadvise(handle, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(handle)
    return found


def scan_shards(
    tar_dir: Path,
    members: set[str],
    index_path: Path,
    dst: Path | None,
    workers: int,
) -> tuple[dict[str, str], set[str]]:
    """并行单遍扫描 tar；已有完整 index 时只扫描各 shard 的命中集合。"""
    for member in members:
        path = Path(member)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"不安全的 required member 路径: {member}")

    old_mapping: dict[str, str] = {}
    if index_path.exists():
        old_mapping = json.loads(index_path.read_text())
        old_mapping = {
            name: shard
            for name, shard in old_mapping.items()
            if name in members and (tar_dir / shard).is_file()
        }

    missing_index = members - set(old_mapping)
    tasks = []
    if missing_index:
        print(f"[index] missing {len(missing_index)} entries; scan all shards", flush=True)
        known_by_shard: dict[str, set[str]] = defaultdict(set)
        for name, shard in old_mapping.items():
            known_by_shard[shard].add(name)
        for shard in SHARDS:
            archive = tar_dir / f"v1.0-trainval{shard}_blobs.tgz"
            shard_name = archive.name
            candidates = set(missing_index) | known_by_shard.get(shard_name, set())
            tasks.append((str(archive), candidates, str(dst) if dst else None))
    else:
        print(f"[index] reuse complete {index_path} ({len(old_mapping)} entries)", flush=True)
        payload_complete = (
            dst is not None
            and all(
                (dst / name).is_file() and (dst / name).stat().st_size > 0
                for name in members
            )
        )
        if payload_complete:
            print("[extract] reuse all existing non-empty sensor payloads", flush=True)
        else:
            by_shard: dict[str, set[str]] = defaultdict(set)
            for name, shard in old_mapping.items():
                by_shard[shard].add(name)
            for shard, assigned in sorted(by_shard.items()):
                tasks.append((str(tar_dir / shard), assigned, str(dst) if dst else None))

    found_rows: dict[str, dict] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for rows in pool.map(_scan_one_shard, tasks):
            for name, row in rows.items():
                if name in found_rows:
                    raise RuntimeError(
                        f"required member 同时出现在多个 shard: "
                        f"{name} ({found_rows[name]['shard']}, {row['shard']})"
                    )
                found_rows[name] = row

    mapping = dict(old_mapping)
    for name, row in found_rows.items():
        mapping[name] = row["shard"]
    missing = members - set(mapping)
    if missing:
        raise RuntimeError(
            f"仍有 {len(missing)} 个成员在 tar 中找不到，示例: {sorted(missing)[:5]}"
        )

    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_index = index_path.with_suffix(index_path.suffix + ".partial")
    tmp_index.write_text(json.dumps({m: mapping[m] for m in sorted(members)}, indent=2) + "\n")
    os.replace(str(tmp_index), str(index_path))
    extracted = {name for name, row in found_rows.items() if row["extracted"]}
    return {m: mapping[m] for m in members}, extracted


def link_existing_files(src_root: Path, dst_root: Path, members: set[str]) -> set[str]:
    """优先复用现有非空 sensor 文件，避免从公共 tar 重复写入。"""
    linked = set()
    for member in sorted(members):
        src = src_root / member
        dst = dst_root / member
        if dst.is_file() and dst.stat().st_size > 0:
            continue
        if not (src.is_file() and src.stat().st_size > 0):
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(str(src), str(dst))
        except OSError:
            shutil.copy2(str(src), str(dst))
        linked.add(member)
    print(f"[reuse] linked/copied {len(linked)} existing sensor files", flush=True)
    return linked


def link_metadata(src_meta: Path, dst_root: Path) -> None:
    """把 v1.0-trainval metadata 以硬链接/复制挂到 raw_subset（不改写源）。"""
    dst_meta = dst_root / "v1.0-trainval"
    if dst_meta.exists():
        return
    dst_meta.mkdir(parents=True, exist_ok=True)
    for p in src_meta.iterdir():
        if p.is_file():
            dest = dst_meta / p.name
            try:
                os.link(p, dest)
            except OSError:
                shutil.copy2(p, dest)


def link_auxiliary_files(src_root: Path, dst_root: Path) -> list[dict]:
    """复制 nuScenes 初始化所需的静态 map mask，并登记内容哈希。"""
    map_records = json.loads(
        (src_root / "v1.0-trainval" / "map.json").read_text()
    )
    rows = []
    for record in sorted(map_records, key=lambda item: item["filename"]):
        name = record["filename"]
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"不安全的 auxiliary 路径: {name}")
        src = src_root / rel
        dst = dst_root / rel
        if not (src.is_file() and src.stat().st_size > 0):
            raise RuntimeError(f"缺少 nuScenes auxiliary 文件: {src}")
        if not (dst.is_file() and dst.stat().st_size > 0):
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(str(src), str(dst))
            except OSError:
                shutil.copy2(str(src), str(dst))
        rows.append({
            "filename": name,
            "bytes": dst.stat().st_size,
            "sha256": sha256_file(dst),
            "source": "local_nuscenes",
            "role": "nuscenes_map_mask",
        })
    print(f"[reuse] linked/copied {len(rows)} nuScenes map masks", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-root", default="/root/autodl-tmp/data/nuscenes")
    ap.add_argument("--tar-dir", default="/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval")
    ap.add_argument("--out-root", default="/root/autodl-tmp/data/dynamic_recon")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    man_dir = out_root / "manifests"
    raw_dir = out_root / "raw_subset" / "adgs_nuscenes_v1"
    man_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("[1] collect required members via NuScenes API")
    # Run collection inside a subprocess with adgs python for nuscenes-devkit
    collect_script = r'''
import json, sys
sys.path.insert(0, "")
from pathlib import Path
# inline import of this module functions by re-exec
exec(open("/root/autodl-tmp/motion_proj/scripts/build_adgs_nuscenes_assets.py").read().split("if __name__")[0])
meta = Path(sys.argv[1])
out = collect_required(meta)
print(json.dumps(out))
'''
    # Simpler: call collect_required directly if nuscenes importable
    try:
        payload = collect_required(Path(args.meta_root))
    except Exception as e:
        print("direct collect failed:", e)
        print("retrying with adgs python...")
        cmd = [
            "/root/autodl-tmp/envs/adgs/bin/python",
            "-c",
            "import json,sys; from pathlib import Path; "
            "import importlib.util; "
            "spec=importlib.util.spec_from_file_location('m','/root/autodl-tmp/motion_proj/scripts/build_adgs_nuscenes_assets.py'); "
            "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
            "print(json.dumps(m.collect_required(Path(sys.argv[1]))))",
            args.meta_root,
        ]
        # Avoid recursive import issues: write a tiny helper instead
        helper = Path("/tmp/collect_adgs_members.py")
        helper.write_text(
            "import json,sys\n"
            "from pathlib import Path\n"
            "import importlib.util\n"
            "p='/root/autodl-tmp/motion_proj/scripts/build_adgs_nuscenes_assets.py'\n"
            "spec=importlib.util.spec_from_file_location('assets', p)\n"
            "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            "print(json.dumps(m.collect_required(Path(sys.argv[1]))))\n"
        )
        out = subprocess.check_output(
            ["/root/autodl-tmp/envs/adgs/bin/python", str(helper), args.meta_root],
            text=True,
        )
        # last line is JSON
        payload = json.loads(out.strip().splitlines()[-1])

    required_files = sorted({r["filename"] for r in payload["required"]})
    req_txt = man_dir / "adgs_nuscenes_v1_required_members.txt"
    req_txt.write_text("\n".join(required_files) + "\n")
    print(f"  total unique files: {len(required_files)}")
    for sn, info in payload["per_scene"].items():
        print(f"  {sn}: files={info['n_files']} rgb={info['n_rgb']} lidar={info['n_lidar']} expected_rgb={info['expected_rgb']}")

    print("[2] build member→shard index")
    index_path = man_dir / "adgs_nuscenes_v1_member_shards.json"
    linked = set()
    auxiliary_rows = []
    if not args.skip_extract:
        link_metadata(Path(args.meta_root) / "v1.0-trainval", raw_dir)
        linked = link_existing_files(Path(args.meta_root), raw_dir, set(required_files))
        auxiliary_rows = link_auxiliary_files(Path(args.meta_root), raw_dir)
    mapping, extracted = scan_shards(
        tar_dir=Path(args.tar_dir),
        members=set(required_files),
        index_path=index_path,
        dst=None if args.skip_extract else raw_dir,
        workers=args.workers,
    )
    tsv = man_dir / "adgs_nuscenes_v1_member_shards.tsv"
    tsv.write_text("filename\tshard\n" + "\n".join(f"{k}\t{v}" for k, v in sorted(mapping.items())) + "\n")

    print("[3] verify extracted payload" if not args.skip_extract else "[3] verify existing payload")
    extract_rows = []
    for f in required_files:
        p = raw_dir / f
        present = p.is_file() and p.stat().st_size > 0
        if f in extracted:
            source = "public_tar"
        elif f in linked:
            source = "local_nuscenes"
        elif present:
            source = "existing_raw_subset"
        else:
            source = None
        extract_rows.append({
            "filename": f,
            "shard": mapping[f],
            "sha256": sha256_file(p) if present else None,
            "bytes": p.stat().st_size if present else 0,
            "source": source,
            "extracted": f in extracted,
        })

    missing = [
        r for r in extract_rows
        if not ((raw_dir / r["filename"]).is_file()
                and (raw_dir / r["filename"]).stat().st_size > 0)
    ]
    manifest = {
        "schema_version": 1,
        "protocol": payload["protocol"],
        "meta_root_source": args.meta_root,
        "tar_dir": args.tar_dir,
        "raw_subset": str(raw_dir),
        "n_required": len(required_files),
        "n_present": len(required_files) - len(missing),
        "complete": len(missing) == 0,
        "missing": [r["filename"] for r in missing],
        "auxiliary_files": auxiliary_rows,
        "per_scene": {
            sn: {
                "scene_token": info["scene_token"],
                "n_files": info["n_files"],
                "n_rgb": info["n_rgb"],
                "n_lidar": info["n_lidar"],
                "expected_rgb": info["expected_rgb"],
                "rgb_ok": info["n_rgb"] == info["expected_rgb"],
                "mean_lidar_dt_ms": sum(fr["lidar_dt_ms"] for fr in info["frames"]) / max(len(info["frames"]), 1),
                "max_lidar_dt_ms": max((fr["lidar_dt_ms"] for fr in info["frames"]), default=0),
            }
            for sn, info in payload["per_scene"].items()
        },
        "files": extract_rows,
        "frame_tables": {sn: info["frames"] for sn, info in payload["per_scene"].items()},
    }
    man_json = man_dir / "adgs_nuscenes_v1_manifest.json"
    # frame_tables 很大，单独存
    frames_path = man_dir / "adgs_nuscenes_v1_frame_tables.json"
    frames_path.write_text(json.dumps(manifest.pop("frame_tables"), indent=2) + "\n")
    man_json.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[done] complete={manifest['complete']} manifest={man_json}")
    if missing:
        print("MISSING:", missing[:10])
        sys.exit(2)


if __name__ == "__main__":
    main()

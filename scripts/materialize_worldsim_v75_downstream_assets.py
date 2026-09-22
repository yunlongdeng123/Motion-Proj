#!/usr/bin/env python3
"""将已下载的 V7.5 资产整理成稳定、可重复使用的运行目录。"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Any


def _record_path(path: Path) -> dict[str, Any]:
    exists = path.exists() or path.is_symlink()
    row: dict[str, Any] = {"path": str(path), "exists": exists}
    if exists:
        row["is_symlink"] = path.is_symlink()
        if path.is_file():
            row["size_bytes"] = path.stat().st_size
    return row


def _completion_issue(
    path: Path,
    *,
    min_size_bytes: int | None = None,
    required_files: tuple[Path, ...] = (),
) -> str | None:
    """Return why a downloaded path is incomplete, or ``None`` when usable."""

    if not path.exists():
        return "missing"
    resolved = path.resolve()
    if Path(f"{resolved}.aria2").exists():
        return "aria2_in_progress"
    if resolved.name.endswith((".part", ".incomplete")):
        return "partial_filename"
    if min_size_bytes is not None:
        if not resolved.is_file() or resolved.stat().st_size < min_size_bytes:
            return "size_below_minimum"
    for relative in required_files:
        issue = _completion_issue(resolved / relative)
        if issue is not None:
            return f"required_file_{issue}:{relative}"
    return None


def _ensure_link(
    source: Path,
    target: Path,
    *,
    min_size_bytes: int | None = None,
    required_files: tuple[Path, ...] = (),
) -> dict[str, Any]:
    issue = _completion_issue(
        source,
        min_size_bytes=min_size_bytes,
        required_files=required_files,
    )
    if issue is not None:
        return {
            "status": "pending_source",
            "reason": issue,
            **_record_path(source),
            "target": str(target),
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        resolved = (target.parent / os.readlink(target)).resolve()
        if resolved == source.resolve():
            return {"status": "ready", "source": str(source), **_record_path(target)}
        raise RuntimeError(f"refusing to replace unrelated symlink: {target} -> {resolved}")
    if target.exists():
        if target.resolve() == source.resolve():
            return {"status": "ready", "source": str(source), **_record_path(target)}
        raise RuntimeError(f"refusing to overwrite existing path: {target}")
    target.symlink_to(source, target_is_directory=source.is_dir())
    return {"status": "ready", "source": str(source), **_record_path(target)}


def _safe_tar_members(archive: tarfile.TarFile, target: Path) -> list[tarfile.TarInfo]:
    root = target.resolve()
    members = archive.getmembers()
    for member in members:
        destination = (target / member.name).resolve()
        if destination != root and root not in destination.parents:
            raise RuntimeError(f"unsafe tar member: {member.name}")
    return members


def _extract_tar(archive_path: Path, target: Path) -> dict[str, Any]:
    marker = target / f".{archive_path.name}.complete"
    issue = _completion_issue(archive_path)
    if issue is not None:
        return {
            "status": "pending_source",
            "reason": issue,
            **_record_path(archive_path),
            "target": str(target),
        }
    if marker.is_file():
        return {"status": "ready", "archive": str(archive_path), **_record_path(marker)}
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = _safe_tar_members(archive, target)
        archive.extractall(target, members=members, filter="data")
    marker.write_text("complete\n", encoding="utf-8")
    return {
        "status": "ready",
        "archive": str(archive_path),
        "member_count": len(members),
        **_record_path(marker),
    }


def _extract_zip(archive_path: Path, target: Path) -> dict[str, Any]:
    marker = target / f".{archive_path.name}.complete"
    issue = _completion_issue(archive_path)
    if issue is not None:
        return {
            "status": "pending_source",
            "reason": issue,
            **_record_path(archive_path),
            "target": str(target),
        }
    if marker.is_file():
        return {"status": "ready", "archive": str(archive_path), **_record_path(marker)}
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        for name in names:
            destination = (target / name).resolve()
            if destination != root and root not in destination.parents:
                raise RuntimeError(f"unsafe zip member: {name}")
        archive.extractall(target)
    marker.write_text("complete\n", encoding="utf-8")
    return {
        "status": "ready",
        "archive": str(archive_path),
        "member_count": len(names),
        **_record_path(marker),
    }


def _resim(asset_root: Path) -> dict[str, Any]:
    root = asset_root / "resim"
    checkpoint = root / "upstream" / "resim_assets" / "resim_ckpts" / "exp0_no_carla"
    cog = root / "upstream" / "cogvideox-2b"
    runtime = root / "runtime"
    links = [
        _ensure_link(checkpoint / "latest", runtime / "transformer" / "latest"),
        _ensure_link(
            checkpoint / "30000",
            runtime / "transformer" / "30000",
            required_files=(Path("mp_rank_00_model_states.pt"),),
        ),
        _ensure_link(
            root / "upstream" / "resim_assets" / "vae" / "3d-vae.pt",
            runtime / "vae" / "3d-vae.pt",
        ),
    ]
    t5_sources = [
        cog / "text_encoder" / "config.json",
        cog / "text_encoder" / "model-00001-of-00002.safetensors",
        cog / "text_encoder" / "model-00002-of-00002.safetensors",
        cog / "text_encoder" / "model.safetensors.index.json",
        cog / "tokenizer" / "added_tokens.json",
        cog / "tokenizer" / "special_tokens_map.json",
        cog / "tokenizer" / "spiece.model",
        cog / "tokenizer" / "tokenizer_config.json",
    ]
    links.extend(
        _ensure_link(source, runtime / "t5-v1_1-xxl" / source.name)
        for source in t5_sources
    )
    return {"status": "ready" if all(row["status"] == "ready" for row in links) else "pending", "links": links}


def _driveeditor(asset_root: Path) -> dict[str, Any]:
    root = asset_root / "driveeditor"
    links = [
        _ensure_link(root / "downloads" / "data.pkl", root / "checkpoints" / "data.pkl"),
        _ensure_link(
            root / "downloads" / "model.safetensors",
            root / "checkpoints" / "model.safetensors",
            min_size_bytes=12_059_467_678,
        ),
    ]
    return {"status": "ready" if all(row["status"] == "ready" for row in links) else "pending", "links": links}


def _gaussiandwm(asset_root: Path, archive_root: Path) -> dict[str, Any]:
    target = asset_root / "gaussiandwm" / "gauss"
    rows = [
        _extract_tar(archive_root / f"scene_{scene}.tar.gz", target)
        for scene in ("0179", "0191", "0204")
    ]
    model = asset_root / "gaussiandwm" / "model" / "model.safetensors"
    model_issue = _completion_issue(model)
    return {
        "status": "ready"
        if model_issue is None and all(row["status"] == "ready" for row in rows)
        else "pending",
        "model": {**_record_path(model), "complete": model_issue is None, "reason": model_issue},
        "sampled_scenes": rows,
    }


def _hugsim(asset_root: Path) -> dict[str, Any]:
    root = asset_root / "hugsim"
    release = root / "official_release"
    runtime = root / "runtime"
    rows = {
        "sample_reconstruction_input": _extract_zip(
            root / "upstream" / "sample_data" / "sample_data" / "data.zip",
            runtime / "sample_data",
        ),
        "exported_scene": _extract_zip(
            release / "scene-0383.zip",
            runtime / "official" / "scenes",
        ),
        "scenarios": _extract_zip(
            release / "scenarios.zip",
            runtime / "official" / "scenarios",
        ),
        "map_cache": _extract_zip(
            release / "nusc_map_cache.zip",
            runtime / "official" / "map_cache",
        ),
    }
    car_sizes = {
        "2024_06_04_16_00_28": 247_239_117,
        "2024_07_02_14_25_45": 186_119_821,
        "2024_07_05_10_58_02": 127_189_901,
        "2024_07_05_15_58_29": 120_879_565,
        "2024_07_05_16_10_02": 156_049_549,
        "2024_07_07_05_43_00": 92_302_605,
    }
    cars = []
    for car_id, expected_size in car_sizes.items():
        source = release / "3DRealCar" / car_id
        row = _ensure_link(
            source,
            runtime / "official" / "3DRealCar" / car_id,
            required_files=(Path("gs.pth"), Path("wlh.json")),
        )
        if row["status"] == "ready":
            issue = _completion_issue(source / "gs.pth", min_size_bytes=expected_size)
            if issue is not None:
                row["status"] = "pending_source"
                row["reason"] = issue
        cars.append({"car_id": car_id, **row})
    ready = all(row["status"] == "ready" for row in rows.values()) and all(
        row["status"] == "ready" for row in cars
    )
    return {"status": "ready" if ready else "pending", **rows, "cars": cars}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--gaussian-archive-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    args = parser.parse_args()
    asset_root = args.asset_root.resolve()
    inventory = {
        "schema_version": "worldsim_v75_downstream_assets_v1",
        "gpu_operations_run": False,
        "models": {
            "resim": _resim(asset_root),
            "driveeditor": _driveeditor(asset_root),
            "gaussiandwm": _gaussiandwm(asset_root, args.gaussian_archive_root.resolve()),
            "hugsim": _hugsim(asset_root),
        },
    }
    args.inventory.parent.mkdir(parents=True, exist_ok=True)
    args.inventory.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value["status"] for key, value in inventory["models"].items()}, indent=2))


if __name__ == "__main__":
    main()

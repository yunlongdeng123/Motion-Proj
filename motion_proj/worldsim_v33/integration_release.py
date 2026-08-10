"""WorldSim V3.3 R0 的可离线验证发布包与确定性归档工具。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, Mapping, Sequence
import zipfile


RELEASE_SCHEMA_VERSION = "worldsim_v33_release_v1"
CONTENT_MANIFEST = "content_manifest.json"
FORBIDDEN_MODEL_SUFFIXES = {".pth", ".pt", ".ckpt", ".safetensors"}


def sha256_file(path: str | Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def verify_file_record(record: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    path = Path(str(record["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"{role} 不存在: {path}")
    actual = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    if actual["sha256"] != str(record["sha256"]) or actual["bytes"] != int(
        record["bytes"]
    ):
        raise RuntimeError(f"{role} 内容漂移: {actual}")
    return actual


def nested_get(payload: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for name in dotted_path.split("."):
        if not isinstance(value, Mapping) or name not in value:
            raise KeyError(f"JSON 缺字段: {dotted_path}")
        value = value[name]
    return value


def validate_expectations(
    payload: Mapping[str, Any], expectations: Mapping[str, Any], *, role: str
) -> None:
    for path, expected in expectations.items():
        actual = nested_get(payload, path)
        if actual != expected:
            raise RuntimeError(
                f"{role} expectation 失败: {path}: expected={expected!r} actual={actual!r}"
            )


def verify_json_input(name: str, spec: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    audit = verify_file_record(spec, role=name)
    payload = json.loads(Path(audit["path"]).read_text(encoding="utf-8"))
    validate_expectations(payload, spec.get("expect", {}), role=name)
    return audit, payload


def copy_verified(
    source: str | Path,
    target: str | Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> dict[str, Any]:
    source_path, target_path = Path(source), Path(target)
    verify_file_record(
        {"path": str(source_path), "sha256": expected_sha256, "bytes": expected_bytes},
        role=f"copy source {source_path}",
    )
    if target_path.exists():
        raise FileExistsError(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return verify_file_record(
        {"path": str(target_path), "sha256": expected_sha256, "bytes": expected_bytes},
        role=f"copy target {target_path}",
    )


def copy_spatial_delta_package(
    package_manifest_path: str | Path, target_root: str | Path
) -> dict[str, Any]:
    manifest_path = Path(package_manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "worldsim_v33_spatial_delta_package_v1":
        raise RuntimeError("S4 package manifest schema 漂移")
    source_root, target = manifest_path.parent, Path(target_root)
    if target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)
    copied = []
    for row in manifest["inventory"]:
        relative = PurePosixPath(str(row["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"S4 package 非法相对路径: {relative}")
        source = source_root.joinpath(*relative.parts)
        destination = target.joinpath(*relative.parts)
        copy_verified(
            source,
            destination,
            expected_sha256=str(row["sha256"]),
            expected_bytes=int(row["bytes"]),
        )
        copied.append(dict(row))
    copy_verified(
        manifest_path,
        target / "package_manifest.json",
        expected_sha256=sha256_file(manifest_path),
        expected_bytes=manifest_path.stat().st_size,
    )
    if any(path.suffix.lower() in FORBIDDEN_MODEL_SUFFIXES for path in target.rglob("*")):
        raise RuntimeError("S4 delta package 非法包含完整模型文件")
    return {
        "schema_version": manifest["schema_version"],
        "files_copied": len(copied) + 1,
        "inventory_bytes": sum(int(row["bytes"]) for row in copied),
        "base_checkpoint_copied": False,
        "inventory": copied,
    }


def copy_production_renders(
    production_manifest_path: str | Path, target_root: str | Path
) -> dict[str, Any]:
    path = Path(production_manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = manifest.get("rows", [])
    if len(rows) != 5:
        raise RuntimeError("S5 production manifest 必须有 5 views")
    target = Path(target_root)
    if target.exists():
        raise FileExistsError(target)
    copied = []
    for row in rows:
        if row.get("selected_arm") != "G0_raw_3d" or not row.get("delete_exact"):
            raise RuntimeError("S5 production arm/delete exact 合同漂移")
        view = f"f{int(row['frame']):03d}_c{int(row['camera_id'])}"
        output = {"frame": int(row["frame"]), "camera_id": int(row["camera_id"])}
        for name in ("insertion", "delete"):
            spec = row[name]
            destination = target / view / f"{name}.png"
            copied_record = copy_verified(
                spec["path"],
                destination,
                expected_sha256=str(spec["sha256"]),
                expected_bytes=int(spec["bytes"]),
            )
            output[name] = {
                "path": str(destination.relative_to(target.parent).as_posix()),
                "sha256": copied_record["sha256"],
                "bytes": copied_record["bytes"],
            }
        if output["delete"]["sha256"] != row["delete_raw_sha256"]:
            raise RuntimeError("S5 delete production 不等于 raw 3D render")
        copied.append(output)
    return {"views": copied, "view_count": len(copied), "selected_arm": "G0_raw_3d"}


def build_content_manifest(release_root: str | Path) -> dict[str, Any]:
    root = Path(release_root)
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == CONTENT_MANIFEST:
            continue
        rows.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    if not rows:
        raise RuntimeError("release package 为空")
    return {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "files": rows,
        "file_count": len(rows),
        "payload_bytes": sum(int(row["bytes"]) for row in rows),
        "full_checkpoint_copy_count": sum(
            Path(row["path"]).suffix.lower() in FORBIDDEN_MODEL_SUFFIXES for row in rows
        ),
    }


def write_content_manifest(release_root: str | Path) -> dict[str, Any]:
    root = Path(release_root)
    manifest = build_content_manifest(root)
    if manifest["full_checkpoint_copy_count"]:
        raise RuntimeError("release package 包含被禁止的完整模型文件")
    atomic_json(root / CONTENT_MANIFEST, manifest)
    return manifest


def verify_release_directory(release_root: str | Path) -> dict[str, Any]:
    root = Path(release_root)
    manifest_path = root / CONTENT_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != RELEASE_SCHEMA_VERSION:
        raise RuntimeError("release content manifest schema 漂移")
    expected = {row["path"]: row for row in manifest["files"]}
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual_paths != set(expected):
        raise RuntimeError(
            f"release file set 漂移: missing={sorted(set(expected)-actual_paths)} "
            f"extra={sorted(actual_paths-set(expected))}"
        )
    for relative, row in expected.items():
        verify_file_record(
            {
                "path": str(root.joinpath(*PurePosixPath(relative).parts)),
                "sha256": row["sha256"],
                "bytes": row["bytes"],
            },
            role=f"release {relative}",
        )
    forbidden = [
        relative
        for relative in actual_paths
        if Path(relative).suffix.lower() in FORBIDDEN_MODEL_SUFFIXES
    ]
    if forbidden or int(manifest.get("full_checkpoint_copy_count", -1)) != 0:
        raise RuntimeError(f"release 含完整模型文件: {forbidden}")
    if int(manifest["file_count"]) != len(expected) or int(
        manifest["payload_bytes"]
    ) != sum(int(row["bytes"]) for row in expected.values()):
        raise RuntimeError("release content aggregate 漂移")
    return {
        "valid": True,
        "file_count": len(expected),
        "payload_bytes": int(manifest["payload_bytes"]),
        "manifest_sha256": sha256_file(manifest_path),
        "full_checkpoint_copy_count": 0,
    }


def write_deterministic_archive(release_root: str | Path, archive_path: str | Path) -> dict[str, Any]:
    root, target = Path(release_root), Path(archive_path)
    if target.exists():
        raise FileExistsError(target)
    verify_release_directory(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".partial.{os.getpid()}")
    with zipfile.ZipFile(
        temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(
                info,
                path.read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    os.replace(temporary, target)
    return {"path": str(target), "sha256": sha256_file(target), "bytes": target.stat().st_size}


def extract_and_verify_archive(archive_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    archive_file, output = Path(archive_path), Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    with zipfile.ZipFile(archive_file, mode="r") as archive:
        for info in archive.infolist():
            relative = PurePosixPath(info.filename)
            if relative.is_absolute() or ".." in relative.parts or info.is_dir():
                raise RuntimeError(f"release archive 非法 entry: {info.filename}")
            target = output.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination)
    return verify_release_directory(output)


def verify_archive_standalone(archive_path: str | Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="worldsim-v33-release-") as temporary:
        return extract_and_verify_archive(archive_path, Path(temporary) / "release")


def _main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    directory = subparsers.add_parser("verify-dir")
    directory.add_argument("path", type=Path)
    archive = subparsers.add_parser("verify-archive")
    archive.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "verify-dir":
        result = verify_release_directory(args.path)
    else:
        result = verify_archive_standalone(args.path)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

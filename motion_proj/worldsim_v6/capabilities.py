"""WorldSim V6 本机能力发现与 logical URI 解析。"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml


SCHEMA_VERSION = "worldsim_v6.local_capabilities.v1"
SUPPORTED_SCHEMES = {
    "repo",
    "runs",
    "cache",
    "asset",
    "dataset",
    "env",
    "third_party",
    "checkpoint",
}


class CapabilityError(RuntimeError):
    """能力清单或 logical URI 不满足 fail-closed 合同。"""


def sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_root_from(start: Path | None = None) -> Path:
    """只从 Git 推导仓库根目录。"""
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=start,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(completed.stdout.strip()).resolve()


def _first_existing(candidates: list[Path]) -> Path | None:
    return next((path.resolve() for path in candidates if path.exists()), None)


def _git_value(root: Path | None, *args: str) -> str | None:
    if root is None or not (root / ".git").exists():
        return None
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _python_record(path: Path | None) -> dict[str, Any]:
    return {
        "path": str(path.resolve()) if path is not None else None,
        "exists": bool(path and path.is_file()),
    }


def _repo_record(path: Path | None, required_files: list[str]) -> dict[str, Any]:
    exists = bool(path and path.is_dir())
    commit = _git_value(path, "rev-parse", "HEAD") if exists else None
    status = _git_value(path, "status", "--porcelain") if exists else None
    origin = _git_value(path, "remote", "get-url", "origin") if exists else None
    license_files = [] if not exists else sorted(item.name for item in path.glob("LICENSE*"))
    return {
        "path": str(path.resolve()) if path is not None else None,
        "exists": exists,
        "commit": commit,
        "clean": status == "" if status is not None else None,
        "origin": origin,
        "license_files": license_files,
        "required_files": {
            relative: bool(path and (path / relative).exists()) for relative in required_files
        },
    }


def _checkpoint_record(path: Path | None) -> dict[str, Any]:
    exists = bool(path and path.is_file())
    return {
        "path": str(path.resolve()) if path is not None else None,
        "exists": exists,
        "bytes": path.stat().st_size if exists and path is not None else None,
        "sha256": sha256_file(path) if exists and path is not None else None,
    }


def _baseline_checkpoint_counts(repo_root: Path) -> dict[str, int]:
    matrix_path = repo_root / "configs/worldsim_v4/baseline_matrix_v1.yaml"
    if not matrix_path.is_file():
        return {"streetgs": 0, "ad_gs": 0}
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    baselines = matrix.get("baselines", {})
    street_rows = baselines.get("streetgs", {}).get("checkpoints", {})
    street_count = sum(Path(row.get("path", "")).is_file() for row in street_rows.values())
    ad_rows = baselines.get("ad_gs", {}).get("executable_checkpoints", {})
    ad_count = 0
    for row in ad_rows.values():
        files = row.get("files", {})
        if files and all(Path(value.get("path", "")).is_file() for value in files.values()):
            ad_count += 1
    return {"streetgs": street_count, "ad_gs": ad_count}


def _gpu_record(primary_python: Path) -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if query.returncode != 0 or not query.stdout.strip():
        return {
            "available": False,
            "name": None,
            "total_vram_mib": None,
            "used_vram_mib": None,
            "driver_version": None,
            "compute_capability": None,
        }
    name, total, used, driver = [part.strip() for part in query.stdout.splitlines()[0].split(",")]
    capability = None
    if primary_python.is_file():
        probe = subprocess.run(
            [
                str(primary_python),
                "-c",
                "import torch; print('.'.join(map(str, torch.cuda.get_device_capability(0))))",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            capability = probe.stdout.strip()
    return {
        "available": True,
        "name": name,
        "total_vram_mib": int(total),
        "used_vram_mib": int(used),
        "driver_version": driver,
        "compute_capability": capability,
    }


def discover_local_capabilities(repo_root: Path) -> dict[str, Any]:
    """按仓库相邻布局发现事实，不在提交文件中固化宿主绝对路径。"""
    repo_root = repo_root.resolve()
    storage_root = repo_root.parent
    third_party_root = storage_root / "third_party"
    env_root = storage_root / "envs"
    data_root = storage_root / "data"
    checkpoint_root = storage_root / "checkpoints"
    primary_python = _first_existing(
        [env_root / "motionproj/bin/python", Path(sys.executable)]
    ) or Path(sys.executable).resolve()
    baseline_counts = _baseline_checkpoint_counts(repo_root)

    nuscenes = _first_existing(
        [
            data_root / "worldsim_v4/drivestudio_raw_trainval",
            data_root / "dynamic_editing_v2/drivestudio_processed_10Hz/trainval",
        ]
    )
    kitti = _first_existing([data_root / "worldsim_v5/kitti_tracking_smoke"])
    waymo = _first_existing([data_root / "waymo", data_root / "waymo_open_dataset"])

    repo_specs = {
        "streetgs": (
            _first_existing(
                [
                    third_party_root / "drivestudio-worldsim-v4-b0",
                    third_party_root / "drivestudio",
                ]
            ),
            ["models/gaussians/basics.py", "tools/train.py"],
        ),
        "ad_gs": (
            _first_existing([third_party_root / "AD-GS"]),
            ["train.py", "render.py"],
        ),
        "recondrive": (
            _first_existing([third_party_root / "recondrive-worldsim-v6-r1"]),
            ["models/recondrive_model.py", "scripts/inference.py", "configs/nuscenes/recondrive.yaml"],
        ),
        "tokengs": (
            _first_existing([third_party_root / "tokengs-worldsim-v6-r1"]),
            ["tokengs/evaluate.py", "tokengs/options.py", "pyproject.toml"],
        ),
        "dggt": (
            _first_existing([third_party_root / "dggt"]),
            ["demo.py"],
        ),
        "instant_nurec": (
            _first_existing([third_party_root / "instant-nurec-worldsim-v3-f0"]),
            ["run_inference.py", "instant_nurec"],
        ),
        "citygs": (_first_existing([third_party_root / "CityGaussian"]), []),
        "lihi_gs": (_first_existing([third_party_root / "LiHi-GS"]), []),
    }
    third_party = {
        name: _repo_record(path, required) for name, (path, required) in repo_specs.items()
    }

    envs = {
        "primary": _python_record(primary_python),
        "drivestudio": _python_record(
            _first_existing([env_root / "drivestudio/bin/python"])
        ),
        "ad_gs": _python_record(_first_existing([env_root / "adgs/bin/python"])),
        "recondrive": _python_record(
            _first_existing([env_root / "recondrive-v6-r1/bin/python"])
        ),
        "tokengs": _python_record(
            _first_existing([env_root / "tokengs-v6-r1/bin/python"])
        ),
        "dggt": _python_record(_first_existing([env_root / "dggt-v2/bin/python"])),
        "instant_nurec": _python_record(
            _first_existing(
                [third_party_root / "instant-nurec-worldsim-v3-f0/.venv/bin/python"]
            )
        ),
    }
    checkpoints = {
        "recondrive_stage2": _checkpoint_record(
            _first_existing(
                [checkpoint_root / "worldsim_v6/recondrive/recondrive_stage2.ckpt"]
            )
        ),
        "tokengs_dynamic": _checkpoint_record(
            _first_existing(
                [checkpoint_root / "worldsim_v6/tokengs/kubric_dyn.safetensors"]
            )
        ),
        "dggt": _checkpoint_record(
            _first_existing([checkpoint_root / "dggt-v2/model_latest_nuscenes.pt"])
        ),
        "instant_nurec": _checkpoint_record(
            Path(os.environ["INSTANT_NUREC_FULL_PT"])
            if os.environ.get("INSTANT_NUREC_FULL_PT")
            else None
        ),
    }

    dggt_runs = sorted(
        (storage_root / "runs/dynamic_editing_v2/DR-V2-M1-DGGT-REPAIR-01").glob("*")
    )
    disk = shutil.disk_usage(storage_root)
    gpu = _gpu_record(primary_python)
    frontends = {
        "streetgs": {
            "repo_key": "streetgs",
            "env_key": "drivestudio",
            "checkpoint_count": baseline_counts["streetgs"],
            "input_ready": nuscenes is not None,
        },
        "ad_gs": {
            "repo_key": "ad_gs",
            "env_key": "ad_gs",
            "checkpoint_count": baseline_counts["ad_gs"],
            "input_ready": nuscenes is not None,
        },
        "recondrive": {
            "repo_key": "recondrive",
            "env_key": "recondrive",
            "checkpoint_key": "recondrive_stage2",
            "input_ready": bool(nuscenes and (nuscenes / "interp_12Hz_trainval").is_dir()),
            "base_nuscenes_ready": nuscenes is not None,
        },
        "tokengs": {
            "repo_key": "tokengs",
            "env_key": "tokengs",
            "checkpoint_key": "tokengs_dynamic",
            "input_ready": False,
            "base_nuscenes_ready": nuscenes is not None,
        },
        "dggt": {
            "repo_key": "dggt",
            "env_key": "dggt",
            "checkpoint_key": "dggt",
            "input_ready": nuscenes is not None,
            "historical_run_count": len(dggt_runs),
        },
        "instant_nurec": {
            "repo_key": "instant_nurec",
            "env_key": "instant_nurec",
            "checkpoint_key": "instant_nurec",
            "input_ready": False,
        },
        "citygs": {"repo_key": "citygs", "checkpoint_count": 0, "input_ready": False},
        "lihi_gs": {"repo_key": "lihi_gs", "checkpoint_count": 0, "input_ready": False},
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "roots": {
            "repo": str(repo_root),
            "runs": str((storage_root / "runs").resolve()),
            "cache": str((storage_root / "cache").resolve()),
            "asset": str((storage_root / "assets").resolve()),
        },
        "datasets": {
            "nuscenes": {
                "path": str(nuscenes) if nuscenes else None,
                "exists": nuscenes is not None,
                "trainval": bool(nuscenes and (nuscenes / "v1.0-trainval").is_dir()),
                "interp_12hz": bool(nuscenes and (nuscenes / "interp_12Hz_trainval").is_dir()),
            },
            "kitti": {"path": str(kitti) if kitti else None, "exists": kitti is not None},
            "waymo": {"path": str(waymo) if waymo else None, "exists": waymo is not None},
        },
        "envs": envs,
        "third_party": third_party,
        "checkpoints": checkpoints,
        "frontends": frontends,
        "gpu": gpu,
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
    }


def write_local_capabilities(path: Path, manifest: Mapping[str, Any]) -> None:
    """原子写入 Git 忽略的本机能力清单。"""
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise CapabilityError("local capability schema_version 漂移")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(
        yaml.safe_dump(dict(manifest), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    partial.replace(path)


def load_local_capabilities(path: Path) -> dict[str, Any]:
    """读取并验证本机能力清单。"""
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise CapabilityError("local capability manifest schema 非法")
    return manifest


class LogicalURIResolver:
    """把提交配置中的 logical URI 安全解析到本机能力清单。"""

    def __init__(self, manifest: Mapping[str, Any]):
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise CapabilityError("resolver 收到未知 capability schema")
        self.manifest = manifest

    def resolve(self, uri: str) -> Path:
        if "://" not in uri:
            raise CapabilityError(f"不是 logical URI：{uri}")
        scheme, payload = uri.split("://", 1)
        if scheme not in SUPPORTED_SCHEMES or not payload:
            raise CapabilityError(f"不支持的 logical URI：{uri}")
        if "\\" in payload or payload.startswith("/"):
            raise CapabilityError(f"logical URI 含平台路径：{uri}")
        parts = [part for part in payload.split("/") if part]
        if any(part in {".", ".."} for part in parts):
            raise CapabilityError(f"logical URI 含路径穿越：{uri}")

        if scheme in {"repo", "runs", "cache", "asset"}:
            base_value = self.manifest["roots"].get(scheme)
            suffix = parts
        else:
            table_name = {
                "dataset": "datasets",
                "env": "envs",
                "third_party": "third_party",
                "checkpoint": "checkpoints",
            }[scheme]
            if not parts:
                raise CapabilityError(f"logical URI 缺少 key：{uri}")
            entry = self.manifest.get(table_name, {}).get(parts[0])
            if not isinstance(entry, Mapping):
                raise CapabilityError(f"logical URI key 不存在：{uri}")
            base_value = entry.get("path")
            suffix = parts[1:]
        if not base_value:
            raise CapabilityError(f"logical URI 本机未映射：{uri}")

        base = Path(str(base_value)).resolve()
        if base.is_file() and suffix:
            raise CapabilityError(f"logical URI 在文件后追加路径：{uri}")
        candidate = base.joinpath(*suffix).resolve() if suffix else base
        if suffix:
            try:
                candidate.relative_to(base)
            except ValueError as error:
                raise CapabilityError(f"logical URI 越出映射根：{uri}") from error
        return candidate

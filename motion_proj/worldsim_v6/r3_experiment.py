"""WorldSim V6 R3：在冻结 checkpoint 上执行 support-deviation 正式实验。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from motion_proj.worldsim_v6.r3_support import analyze_support_deviation


TASK_ID = "WS-V6-R3-SUPPORT-DEVIATION-01"


class R3ExperimentError(RuntimeError):
    """正式运行合同或资源合同失败。"""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _gpu_memory_mib() -> int:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]
    return int(output.strip())


def _cgroup_memory() -> tuple[int | None, int | None]:
    candidates = (
        (Path("/sys/fs/cgroup/memory.current"), Path("/sys/fs/cgroup/memory.max")),
        (
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
        ),
    )
    for current_path, limit_path in candidates:
        if current_path.is_file() and limit_path.is_file():
            current_text = current_path.read_text(encoding="utf-8").strip()
            limit_text = limit_path.read_text(encoding="utf-8").strip()
            return int(current_text), None if limit_text == "max" else int(limit_text)
    return None, None


def _resource_preflight(run_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    disk = shutil.disk_usage(run_root.parent if not run_root.exists() else run_root)
    free_gib = disk.free / (1024**3)
    used_gpu = _gpu_memory_mib()
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R3ExperimentError(f"磁盘剩余 {free_gib:.1f} GiB，不满足 R3")
    if used_gpu > 2048:
        raise R3ExperimentError(f"GPU 起始占用 {used_gpu} MiB，拒绝与其他重任务并发")
    current, limit = _cgroup_memory()
    ratio = None if current is None or limit in (None, 0) else current / limit
    if ratio is not None and ratio >= float(config["resources"]["stop_cgroup_ratio"]):
        raise R3ExperimentError(f"cgroup 起始内存比例 {ratio:.3f} 已越界")
    return {
        "disk_free_gib": free_gib,
        "gpu_used_mib": used_gpu,
        "cgroup_current_bytes": current,
        "cgroup_limit_bytes": limit,
        "cgroup_ratio": ratio,
    }


def _run_process(
    label: str,
    command: Sequence[str],
    cwd: Path,
    log_path: Path,
    config: Mapping[str, Any],
    *,
    monitor_gpu: bool,
) -> dict[str, Any]:
    print(f"[R3] start {label}", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["OMP_NUM_THREADS"] = "4"
    started = time.monotonic()
    peak_gpu = 0
    peak_cgroup = 0
    samples = 0
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            if monitor_gpu:
                used_gpu = _gpu_memory_mib()
                peak_gpu = max(peak_gpu, used_gpu)
                if used_gpu > int(config["resources"]["maximum_peak_gpu_memory_mib"]):
                    process.terminate()
                    process.wait(timeout=30)
                    raise R3ExperimentError(f"{label} GPU 峰值越过冻结上限：{used_gpu} MiB")
            current, limit = _cgroup_memory()
            if current is not None:
                peak_cgroup = max(peak_cgroup, current)
            if current is not None and limit not in (None, 0):
                ratio = current / limit
                if ratio >= float(config["resources"]["stop_cgroup_ratio"]):
                    process.terminate()
                    process.wait(timeout=30)
                    raise R3ExperimentError(f"{label} cgroup 内存比例越界：{ratio:.3f}")
            samples += 1
            time.sleep(2)
    if process.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise R3ExperimentError(f"{label} 失败，returncode={process.returncode}\n{tail}")
    result = {
        "label": label,
        "wall_seconds": time.monotonic() - started,
        "peak_gpu_memory_mib": peak_gpu if monitor_gpu else None,
        "peak_cgroup_memory_bytes": peak_cgroup,
        "sample_count": samples,
        "command": list(command),
        "log": str(log_path),
    }
    print(
        f"[R3] done {label} wall={result['wall_seconds']:.1f}s peak_gpu={peak_gpu}MiB",
        flush=True,
    )
    return result


def _verify_adgs_bundle(row: Mapping[str, Any]) -> Path:
    run = Path(row["run"])
    for name, file_row in row["files"].items():
        path = Path(file_row["path"])
        if not path.is_file() or _sha256(path) != file_row["sha256"]:
            raise R3ExperimentError(f"AD-GS {name} checkpoint 漂移：{path}")
    return run / "model"


def _materialize_inference_only_depth_placeholders(adapter: Path) -> dict[str, Any]:
    """满足 AD-GS 强制 loader 字段；R3 renderer 与指标均不消费这些值。"""
    manifest = json.loads((adapter / "adapter_manifest.json").read_text(encoding="utf-8"))
    width, height = (int(value) for value in manifest["target_size"])
    depth_dir = adapter / "depth"
    image_paths = sorted((adapter / "image").glob("*.png"))
    if not image_paths:
        raise R3ExperimentError("AD-GS adapter 没有图像")
    placeholder = np.zeros((height, width, 1), dtype=np.float32)
    for image_path in image_paths:
        target = depth_dir / f"{image_path.stem}.npy"
        if target.exists():
            raise R3ExperimentError(f"拒绝覆盖 adapter depth：{target}")
        np.save(target, placeholder, allow_pickle=False)
    audit = {
        "schema_version": "worldsim_v6.r3_inference_only_depth_placeholder.v1",
        "count": len(image_paths),
        "shape": [height, width, 1],
        "dtype": "float32",
        "value": 0.0,
        "common_file_sha256": _sha256(depth_dir / f"{image_paths[0].stem}.npy"),
        "adgs_loader_field_only": True,
        "adgs_renderer_consumes_depth": False,
        "r3_metrics_consume_placeholder_depth": False,
        "metric_depth_source": "DriveStudio sparse LiDAR exported by StreetGS worker",
    }
    _write_json(adapter / "R3_INFERENCE_ONLY_DEPTH_PLACEHOLDERS.json", audit)
    return audit


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R3ExperimentError("正式 R3 run 禁止使用 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R3ExperimentError("R3 task_id 漂移")
    matrix_path = repo_root / "configs/worldsim_v4/baseline_matrix_v1.yaml"
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    street = matrix["baselines"]["streetgs"]
    adgs = matrix["baselines"]["ad_gs"]
    street_root = Path(street["implementation_root"])
    adgs_root = Path(adgs["implementation_root"])
    if _git(street_root, "rev-parse", "HEAD") != street["implementation_commit"]:
        raise R3ExperimentError("StreetGS source commit 漂移")
    if _git(adgs_root, "rev-parse", "HEAD") != adgs["implementation_commit"]:
        raise R3ExperimentError("AD-GS source commit 漂移")
    street_python = Path(street["environment"]) / "bin/python"
    adgs_python = Path(adgs["environment"]) / "bin/python"
    for python in (street_python, adgs_python):
        if not python.is_file():
            raise R3ExperimentError(f"Python 环境缺失：{python}")

    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / (
        f"{now.strftime('%Y%m%dT%H%M%SZ')}__support-deviation-s{config['seed']}-r1"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    resources = {"preflight": _resource_preflight(run_root, config), "processes": []}
    try:
        offsets = ",".join(str(value) for value in config["camera_deviation_profile"]["lateral_offsets_m"])
        forward = str(config["camera_deviation_profile"]["forward_extension_m"])
        for scene_row in config["cohort"]["scenes"]:
            scene = scene_row["scene"]
            frames = ",".join(str(value) for value in scene_row["source_frame_indices"])
            scene_index = int(matrix["scene_contract"][scene]["scene_index"])
            street_checkpoint_row = street["checkpoints"][scene]
            street_checkpoint = Path(street_checkpoint_row["path"])
            if not street_checkpoint.is_file() or _sha256(street_checkpoint) != street_checkpoint_row["sha256"]:
                raise R3ExperimentError(f"StreetGS checkpoint 漂移：{scene}")
            adgs_row = adgs["executable_checkpoints"][scene]
            adgs_model = _verify_adgs_bundle(adgs_row)
            resolved = yaml.safe_load((Path(adgs_row["run"]) / "resolved.yaml").read_text(encoding="utf-8"))
            source_scene = Path(resolved["data"]["source_root"]) / f"{scene_index:03d}"
            adapter = run_dir / "development_adapters" / scene
            adapter.parent.mkdir(parents=True, exist_ok=True)
            resources["processes"].append(
                _run_process(
                    f"adapter-{scene}",
                    [
                        str(adgs_python),
                        str(repo_root / "scripts/prepare_worldsim_v4_adgs.py"),
                        "--source",
                        str(source_scene),
                        "--destination",
                        str(adapter),
                        "--partitions",
                        "train",
                        "development",
                    ],
                    repo_root,
                    run_dir / f"adapter-{scene}.log",
                    config,
                    monitor_gpu=False,
                )
            )
            _materialize_inference_only_depth_placeholders(adapter)
            street_output = run_dir / "renders" / scene / "streetgs"
            street_output.parent.mkdir(parents=True, exist_ok=True)
            resources["processes"].append(
                _run_process(
                    f"streetgs-{scene}",
                    [
                        str(street_python),
                        str(repo_root / "scripts/worldsim_v6/r3_streetgs_worker.py"),
                        "--repo-root",
                        str(repo_root),
                        "--checkpoint",
                        str(street_checkpoint),
                        "--upstream-root",
                        str(street_root),
                        "--output",
                        str(street_output),
                        "--scene",
                        scene,
                        "--frames",
                        frames,
                        "--offsets",
                        offsets,
                        "--forward-extension",
                        forward,
                    ],
                    repo_root,
                    run_dir / f"streetgs-{scene}.log",
                    config,
                    monitor_gpu=True,
                )
            )
            adgs_output = run_dir / "renders" / scene / "ad_gs"
            resources["processes"].append(
                _run_process(
                    f"adgs-{scene}",
                    [
                        str(adgs_python),
                        str(repo_root / "scripts/worldsim_v6/r3_adgs_worker.py"),
                        "--source-root",
                        str(adgs_root),
                        "--model-root",
                        str(adgs_model),
                        "--adapter",
                        str(adapter),
                        "--output",
                        str(adgs_output),
                        "--scene",
                        scene,
                        "--frames",
                        frames,
                        "--offsets",
                        offsets,
                        "--forward-extension",
                        forward,
                    ],
                    repo_root,
                    run_dir / f"adgs-{scene}.log",
                    config,
                    monitor_gpu=True,
                )
            )

        case_rows, ranking, actor_rows, forward_rows = analyze_support_deviation(run_dir, config)
        _write_jsonl(run_dir / "PER_CASE_METRICS.jsonl", case_rows)
        _write_json(run_dir / "SUPPORT_RANKING.json", ranking)
        _write_jsonl(run_dir / "ACTOR_EDIT_EFFECTS.jsonl", actor_rows)
        _write_jsonl(run_dir / "FORWARD_EXTENSION_METRICS.jsonl", forward_rows)
        _write_json(run_dir / "RESOURCE_AUDIT.json", resources)
        summary = {
            "schema_version": "worldsim_v6.r3_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "hypothesis_outcome": "accepted" if ranking["gate"]["passed"] else "rejected",
            "method_decision": ranking["decision"],
            "source_commit": source_commit,
            "source_dirty": False,
            "scene_count": ranking["scene_count"],
            "frontend_count": ranking["frontend_count"],
            "training_started": False,
            "confirmation_content_read": False,
            "development_content_read": True,
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = [
            "PER_CASE_METRICS.jsonl",
            "SUPPORT_RANKING.json",
            "ACTOR_EDIT_EFFECTS.jsonl",
            "FORWARD_EXTENSION_METRICS.jsonl",
            "RESOURCE_AUDIT.json",
            "SUMMARY.json",
        ]
        for scene_row in config["cohort"]["scenes"]:
            scene = scene_row["scene"]
            tracked.extend(
                [
                    f"development_adapters/{scene}/adapter_manifest.json",
                    f"development_adapters/{scene}/R3_INFERENCE_ONLY_DEPTH_PLACEHOLDERS.json",
                    f"renders/{scene}/streetgs/AUDIT.json",
                    f"renders/{scene}/streetgs/RENDER_MAP.jsonl",
                    f"renders/{scene}/ad_gs/AUDIT.json",
                    f"renders/{scene}/ad_gs/RENDER_MAP.jsonl",
                ]
            )
        manifest = {
            "schema_version": "worldsim_v6.r3_run_manifest.v1",
            "files": {
                relative: {
                    "bytes": (run_dir / relative).stat().st_size,
                    "sha256": _sha256(run_dir / relative),
                }
                for relative in tracked
            },
        }
        _write_json(run_dir / "MANIFEST.json", manifest)
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": "done",
                "summary_sha256": _sha256(run_dir / "SUMMARY.json"),
                "manifest_sha256": _sha256(run_dir / "MANIFEST.json"),
            },
        )
        print(str(run_dir), flush=True)
        return run_dir
    except Exception as error:
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/worldsim_v6/r3_support_deviation_v1.yaml"),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/root/autodl-tmp/runs/worldsim_v6"),
    )
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

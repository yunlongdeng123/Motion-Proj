"""在隔离目录用真实远端源图重建四包，记录CPU合同/逐数组比较。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    args = parser.parse_args()
    root, legacy = args.campaign.resolve(), args.legacy.resolve()
    imported = root / "supervision/imported_r2"
    output = root / "supervision/cpu_rebuild_r1"
    if output.exists():
        raise FileExistsError("保留已有重建目录: " + str(output))
    output.mkdir()
    cpu_quota, cpu_period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
    memory_limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
    if cpu_quota == "max" or float(cpu_quota) / float(cpu_period) > .5:
        raise RuntimeError("当前 cgroup CPU 上限超过0.5")
    if memory_limit == "max" or int(memory_limit) > 2 * 1024**3:
        raise RuntimeError("当前 cgroup 内存上限超过2GB")
    receipt = dict(schema="road_2d_teacher_cpu_remote_validation.v1", task_id="ROAD-2D-TEACHER-HF-CPU-20261119",
                   complete=False, CPU_only=True, GPU_executed=False, native_render_calls=0,
                   training_updates=0, parent_files_modified=False, output=str(output),
                   cgroup_cpu_cores=float(cpu_quota) / float(cpu_period), cgroup_memory_bytes=int(memory_limit),
                   threads=1, numpy_version=np.__version__, python=sys.version, package_comparisons=[])
    target_receipt = root / "supervision/cpu_remote_validation.json"
    started = time.monotonic()
    try:
        unit = subprocess.run([sys.executable, "-m", "unittest", "-v", "test_teacher_supervision_cpu.py"],
                              cwd=root / "code", text=True, capture_output=True, check=False)
        (output / "unit_tests.log").write_text(unit.stdout + unit.stderr, encoding="utf-8")
        receipt["unit_tests"] = dict(exit_code=unit.returncode, required_tests=7,
                                     log=str(output / "unit_tests.log"))
        if unit.returncode:
            raise RuntimeError("CPU合同测试失败")
        prepared = read(legacy / "prepared/prepared.json")
        predictions = read(legacy / "predictions/predictions.json")
        case_sources = {row["id"]: row for row in prepared["cases"]}
        case_outputs = {row["id"]: row["outputs"]["B"] for row in predictions["cases"]}
        for raw in (86, 110):
            sid = f"scene065_raw{raw:03d}"
            for wh in ((480, 320), (768, 512)):
                name = f"{raw:03d}_{wh[0]}x{wh[1]}"
                config = read(imported / name / "rebuild_config.json")
                config["raw_model"] = case_outputs[sid]["raw_model"]["path"]
                config["observation"] = case_sources[sid]["observation"]["path"]
                config["native_domains"] = str(imported / f"{raw:03d}_native_domains.npz")
                config["camera"]["convention_is_assumed_from_legacy_CPU_rays"] = True
                config["camera"]["actual_raster_camera_convention_verified"] = False
                for key in ("raw_model", "observation", "native_domains"):
                    if not Path(config[key]).is_file():
                        raise FileNotFoundError(config[key])
                config_path = output / (name + ".config.json")
                # 仅修改本次新上传的重建配置；原数值包和历史源文件保持原值。
                write(imported / name / "rebuild_config.json", config)
                write(config_path, config)
                command = [sys.executable, str(root / "code/teacher_supervision_cpu.py"),
                           "--config", str(config_path), "--output", str(output / name)]
                process = subprocess.run(command, text=True, capture_output=True, check=False)
                (output / (name + ".cli.log")).write_text(process.stdout + process.stderr, encoding="utf-8")
                if process.returncode:
                    raise RuntimeError("实际CLI重建失败: " + name)
                comparisons = []
                with np.load(imported / name / "supervision.npz", allow_pickle=False) as source, np.load(output / name / "supervision.npz", allow_pickle=False) as rebuilt:
                    if source.files != rebuilt.files:
                        raise RuntimeError("重建数组键不一致")
                    for key in source.files:
                        a, b = source[key], rebuilt[key]
                        exact = bool(np.array_equal(a, b))
                        max_abs = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
                        close = bool(np.allclose(a, b, rtol=0, atol=1e-7)) if a.dtype.kind == "f" else exact
                        comparisons.append(dict(key=key, dtype=str(a.dtype), shape=list(a.shape), exact=exact,
                                                allclose_atol_1e_7=close, maximum_absolute_difference=max_abs))
                        if not close:
                            raise RuntimeError("重建数组不一致: " + name + "/" + key)
                first, second = read(imported / name / "manifest.json"), read(output / name / "manifest.json")
                if first["counts"] != second["counts"]:
                    raise RuntimeError("重建域计数不一致")
                for flag in ("q_not_material_permission", "preserve_parent_high_frequency_policy", "no_new_3d_high_frequency_authorization"):
                    if first.get(flag) is not True or second.get(flag) is not True:
                        raise RuntimeError("缺少二维q与三维高频许可分离合同: " + flag)
                if first["training_ready"] or second["training_ready"]:
                    raise RuntimeError("CPU候选不能标记为训练就绪")
                row = dict(name=name, sources=config, counts_exact=True, counts=second["counts"],
                           all_arrays_exact=all(x["exact"] for x in comparisons),
                           all_arrays_allclose_atol_1e_7=all(x["allclose_atol_1e_7"] for x in comparisons),
                           maximum_absolute_difference=max(x["maximum_absolute_difference"] for x in comparisons), arrays=comparisons)
                receipt["package_comparisons"].append(row)
                write(target_receipt, receipt)
                print(json.dumps(dict(package=name, all_arrays_exact=row["all_arrays_exact"],
                                      maximum_absolute_difference=row["maximum_absolute_difference"])), flush=True)
        receipt.update(complete=True, all_packages_exact=all(row["all_arrays_exact"] for row in receipt["package_comparisons"]),
                       package_count=4, camera_convention_assumed=True, actual_raster_camera_convention_verified=False,
                       preserve_parent_high_frequency_policy=True, new_3d_high_frequency_authorization=False)
    except BaseException as error:
        receipt.update(complete=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        write(target_receipt, receipt)


if __name__ == "__main__":
    main()

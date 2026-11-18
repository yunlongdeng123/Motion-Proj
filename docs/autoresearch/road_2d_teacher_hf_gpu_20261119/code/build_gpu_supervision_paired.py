"""冻结GPU原图的统一方法CPU监督包；不选择每图/seed最优，不导入GPU。"""
import os
for key in ("CUDA_VISIBLE_DEVICES",):
    os.environ[key] = ""
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from teacher_supervision_view_cpu import (build_view_supervision, fixed_view_spec,
                                         mapped_resample, save_view_package)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def rgb(path):
    with Image.open(path) as im:
        assert im.mode == "RGB" and im.format == "PNG"
        return np.asarray(im).copy()


def pinned(record):
    assert hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest() == record["sha256"]
    return Path(record["path"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("保留已有监督包，请指定新output")
    predictions = read(args.predictions)
    assert predictions["complete"] and predictions["attempted_diffusion_calls"] == predictions["completed_diffusion_calls"] == 48
    manifest_path = pinned(predictions["manifest"])
    manifest = read(manifest_path)
    cases = {case["id"]: case for case in manifest["cases"]}
    replay = [job for job in predictions["jobs"] if job.get("legacy_exact_replay_expected")]
    assert len(replay) == 6 and all(job["legacy_replay_exact"] and all(job["legacy_replay_exact"].values()) for job in replay)
    jobs = {(job["case_id"], job["arm"], job["seed"]): job for job in predictions["jobs"]}
    args.output.mkdir(parents=True)
    imported = args.campaign / "supervision/imported_r2"
    receipt = dict(schema="road_2d_gpu_uniform_paired_supervision.v1", complete=False, GPU_executed=False,
        training_updates=0, native_3d_calls=0, source_predictions=str(args.predictions), source_manifest=str(manifest_path),
        uniform_arms=["full768_T0", "full1152_T0"], raw_frames=[86, 110], seeds=[101, 202],
        per_case_or_seed_selection=False, all_candidates_training_ready=False,
        parent_high_frequency_policy_preserved=True, new_3d_HF_authorization=False,
        native_domains_source=str(imported), packages=[], comparisons=[],
        note="仅保留冻结教师的2D候选与配对来源许可，不是方法晋升或3D质量结论")
    start = time.monotonic()
    masks_at_output = {}
    for raw in (86, 110):
        sid = f"scene065_raw{raw:03d}"
        case = cases[sid]
        observed = rgb(pinned(case["observation"]))
        domain_path = imported / f"{raw:03d}_native_domains.npz"
        with np.load(domain_path, allow_pickle=False) as saved:
            domains = {key: saved[key].copy() for key in ("A", "C", "S", "teacher_write", "K", "protected")}
        camera = read(imported / f"{raw:03d}_768x512/rebuild_config.json")["camera"]
        for seed in (101, 202):
            for arm in ("full768_T0", "full1152_T0"):
                job = jobs[(sid, arm, seed)]
                raw_path = pinned(job["raw_model"])
                raw_rgb = rgb(raw_path)
                spec = fixed_view_spec(job["view"])
                assert list(raw_rgb.shape[1::-1]) == spec["model_wh"]
                native_valid_path = pinned(job["candidate_native_valid"])
                with Image.open(native_valid_path) as im:
                    a = np.asarray(im)
                    assert a.ndim == 2 and np.isin(a, [0, 255]).all()
                    native_valid = a == 255
                assert native_valid.shape == observed.shape[:2] and native_valid.all()
                source_valid = mapped_resample(native_valid, spec["model_wh"], (0, 0, 1920, 1280),
                                              spec["native_window_xyxy"], mask_mode="all")
                source_dir = args.output / "sources" / job["id"]
                source_dir.mkdir(parents=True)
                source_valid_path = source_dir / "model_valid.png"
                Image.fromarray(source_valid.astype(np.uint8) * 255).save(source_valid_path)
                targets = [(768, 512), (480, 320)] if arm == "full768_T0" else [(1152, 768), (768, 512), (480, 320)]
                for wh in targets:
                    folder = args.output / f"raw{raw:03d}" / f"{arm}_seed{seed}" / f"{wh[0]}x{wh[1]}"
                    config = dict(raw_model=str(raw_path), observation=case["observation"]["path"],
                                  source_model_valid=str(source_valid_path), native_domains=str(domain_path),
                                  source_kind="lossless_raw_model", output_wh=list(wh), camera=camera, view=spec)
                    arrays, report = build_view_supervision(raw_rgb, observed, domains, source_model_valid=source_valid,
                        native_window_xyxy=spec["native_window_xyxy"], native_from_model=spec["native_from_model"],
                        output_wh=wh, intrinsics=camera["intrinsics"], camera_wh=camera["camera_wh"],
                        camera_convention=camera["convention"], source_kind="lossless_raw_model", view_name=spec["name"])
                    report.update(task_id="ROAD-2D-TEACHER-HF-CPU-20261119", source_prediction_job=job["id"],
                                  arm=arm, seed=seed, raw=raw, source_prediction_file=job["raw_model"],
                                  camera_convention_assumed=True, uniform_method_comparison_only=True)
                    is_source_level = list(wh) == spec["model_wh"]
                    source_exact = bool(np.array_equal(arrays["target_C_srgb"], raw_rgb.transpose(2, 0, 1).astype(np.float32) / 255)) if is_source_level else None
                    assert source_exact is not False
                    baseline_prior_exact = None
                    if arm == "full768_T0" and seed == 101:
                        with np.load(imported / f"{raw:03d}_{wh[0]}x{wh[1]}/supervision.npz", allow_pickle=False) as prior:
                            baseline_prior_exact = all(np.array_equal(arrays[key], prior[key]) for key in prior.files)
                        assert baseline_prior_exact
                    save_view_package(folder, arrays, report, config)
                    write(folder / "rebuild_config.json", config)
                    masks_at_output[(raw, seed, arm, wh)] = {key: arrays[key].copy() for key in ("A", "C", "S")}
                    row = dict(path=str(folder), raw=raw, arm=arm, seed=seed, output_wh=list(wh), source_level=is_source_level,
                               source_level_teacher_RGB_exact=source_exact, baseline_seed101_all_prior_arrays_exact=baseline_prior_exact,
                               counts=report["counts"], candidate_only=True, training_ready=False)
                    receipt["packages"].append(row)
                    write(args.output / "build_receipt.json", receipt)
                    print(json.dumps(dict(package=str(folder.relative_to(args.output)), source_level_exact=source_exact,
                                          baseline_prior_exact=baseline_prior_exact)), flush=True)
                del raw_rgb
    for raw in (86, 110):
        for seed in (101, 202):
            for wh in ((480, 320), (768, 512)):
                a = masks_at_output[(raw, seed, "full768_T0", wh)]
                b = masks_at_output[(raw, seed, "full1152_T0", wh)]
                counts = {key: dict(baseline=int(a[key].sum()), highres=int(b[key].sum()), common=int((a[key] & b[key]).sum()),
                                    baseline_only=int((a[key] & ~b[key]).sum()), highres_only=int((b[key] & ~a[key]).sum())) for key in ("A", "C", "S")}
                receipt["comparisons"].append(dict(raw=raw, seed=seed, output_wh=list(wh), two_dimensional_permission_counts=counts,
                    common_training_denominator=False, note="未来比较仍需交相同固定3D资格；此处仅为来源域共同分母"))
    assert len(receipt["packages"]) == 20
    receipt.update(complete=True, package_count=20, source_level_identity_checks=8,
                   baseline_seed101_prior_exact_checks=4, elapsed_seconds=time.monotonic() - start)
    write(args.output / "build_receipt.json", receipt)


if __name__ == "__main__":
    main()

"""预测全部冻结后导出轻量报告图，不计算科学指标，不改任何原始图。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from PIL import Image, ImageFilter


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(record):
    assert sha(record["path"]) == record["sha256"], record["path"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--truth-config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--confirmed-frozen", action="store_true")
    args = parser.parse_args()
    assert args.confirmed_frozen, "只允许在48次预测全部冻结后导出"
    predictions, manifest = read(args.predictions), read(args.manifest)
    assert predictions["complete"] and predictions["completed_diffusion_calls"] == 48
    assert len(predictions["jobs"]) == len(manifest["jobs"]) == 48
    assert not predictions["GT_opened"] and predictions["training_updates"] == 0
    assert predictions["manifest"]["sha256"] == sha(args.manifest)
    assert {j["id"] for j in predictions["jobs"]} == {j["id"] for j in manifest["jobs"]}
    replays = [job for job in predictions["jobs"] if job["legacy_exact_replay_expected"]]
    assert len(replays) == 6 and all(all(job["legacy_replay_exact"].values()) for job in replays)
    assert all(job["W_outside_RGB_exact"] and job["K_RGB_exact"] for job in predictions["jobs"])
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("保留已有导出，请指定新目录: " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    for folder in ["raw_model", "previews", "roi", "inputs", "VAE"]:
        (out / folder).mkdir()
    truth = None if args.truth_config is None else read(args.truth_config)
    if truth is not None:
        assert truth["independent_scoring_only"] and truth["generation_must_not_read"]
    files = []

    def record(path, role, source=None):
        path = Path(path)
        item = dict(path=path.relative_to(out).as_posix(), sha256=sha(path), bytes=path.stat().st_size,
                    role=role, source=source)
        with Image.open(path) as image:
            item["wh"] = list(image.size)
        files.append(item)
        return item["path"]

    def preview(image, path, source):
        value = image.copy()
        value.thumbnail((960, 640), Image.Resampling.LANCZOS)
        value.save(path, "WEBP", quality=90, method=4)
        return record(path, "lossy full-frame preview, never a metric source", source)

    def png(image, path, role, source):
        image.save(path, "PNG", compress_level=4)
        return record(path, role, source)

    cases = []
    for case in manifest["cases"]:
        check(case["observation"])
        check(case["roles_native"])
        with np.load(case["roles_native"]["path"], allow_pickle=False) as z:
            write_mask = z["W_output"].copy()
            hidden = z["H_input"].copy()
            keep = z["K"].copy()
        yy, xx = np.where(write_mask)
        h, w = write_mask.shape
        # 所有显示窗口仅由原先冻结W确定，与任何模型输出无关。
        window = [max(0, int(xx.min()) - 96), max(0, int(yy.min()) - 96),
                  min(w, int(xx.max()) + 1 + 96), min(h, int(yy.max()) + 1 + 96)]
        observation = Image.open(case["observation"]["path"]).convert("RGB")
        reference = dict(
            full_preview=preview(observation, out / "inputs" / (case["id"] + "_observation.webp"), case["observation"]),
            roi=png(observation.crop(window), out / "inputs" / (case["id"] + "_observation_roi.png"),
                    "lossless fixed W+96px ROI from observation", case["observation"]))
        mask_sources = {}
        for name, mask in [("W", write_mask), ("H", hidden), ("K", keep)]:
            img = Image.fromarray(mask.astype(np.uint8) * 255)
            mask_sources[name] = png(img.crop(window), out / "inputs" / (case["id"] + "_" + name + "_roi.png"),
                                     "frozen native role, lossless fixed ROI", case["roles_native"])
        overlay = observation.convert("RGBA")
        for mask, color in [(hidden, (78, 165, 255)), (write_mask, (91, 225, 181)), (keep, (244, 190, 99))]:
            image_mask = Image.fromarray(mask.astype(np.uint8) * 255)
            edge = np.asarray(image_mask.filter(ImageFilter.MaxFilter(5))) != np.asarray(image_mask.filter(ImageFilter.MinFilter(5)))
            color_layer = np.zeros((h, w, 4), np.uint8)
            color_layer[..., :3] = color
            color_layer[..., 3][edge] = 235
            overlay = Image.alpha_composite(overlay, Image.fromarray(color_layer))
        reference["overlay_roi"] = png(overlay.convert("RGB").crop(window),
                                         out / "inputs" / (case["id"] + "_roles_roi.png"),
                                         "role outline illustration, not a metric source", case["roles_native"])
        if truth is not None and "clean_rgb" in truth["cases"][case["id"]]:
            source = truth["cases"][case["id"]]["clean_rgb"]
            check(source)
            clean = Image.open(source["path"]).convert("RGB")
            assert clean.size == observation.size
            reference["clean_roi"] = png(clean.crop(window), out / "inputs" / (case["id"] + "_clean_roi.png"),
                                          "clean reference ROI; only certified road is scientific truth", source)
        jobs = []
        for job in predictions["jobs"]:
            if job["case_id"] != case["id"]:
                continue
            for name in ["raw_model", "raw_native", "final_native"]:
                check(job[name])
            raw_model = out / "raw_model" / (job["id"] + ".png")
            shutil.copyfile(job["raw_model"]["path"], raw_model)
            native = Image.open(job["raw_native"]["path"]).convert("RGB")
            composite = Image.open(job["final_native"]["path"]).convert("RGB")
            assert native.size == composite.size == observation.size
            jobs.append(dict(id=job["id"], case_id=job["case_id"], seed=job["seed"], arm=job["arm"],
                             model_wh=case["views"][job["view"]]["model_wh"],
                             seconds=job["seconds"], raw_model=record(raw_model, "byte-exact original generated PNG", job["raw_model"]),
                             full_preview=preview(native, out / "previews" / (job["id"] + ".webp"), job["raw_native"]),
                             raw_roi=png(native.crop(window), out / "roi" / (job["id"] + "_raw.png"),
                                         "lossless crop of original raw_native; native-coordinate interpolation already occurred upstream", job["raw_native"]),
                             final_roi=png(composite.crop(window), out / "roi" / (job["id"] + "_final.png"),
                                           "lossless crop of original final_native", job["final_native"]),
                             remote_originals={name: job[name] for name in ["raw_model", "raw_native", "final_native"]}))
        assert len(jobs) == 8
        cases.append(dict(id=case["id"], domain=case["domain"], native_hw=case["native_hw"],
                          roi_xyxy=window, reference=reference, masks=mask_sources, jobs=jobs))
        print(json.dumps(dict(case=case["id"], jobs=len(jobs), CPU_only=True)), flush=True)
    for control in predictions.get("VAE_controls", []):
        source = control["roundtrip_model"]
        check(source)
        shutil.copyfile(source["path"], out / "VAE" / (control["id"] + ".png"))
        record(out / "VAE" / (control["id"] + ".png"), "original VAE mode roundtrip model PNG", source)
    result = dict(schema="road_2d_teacher_report_assets.v1", complete=True, CPU_only=True,
                  predictions=dict(path=str(args.predictions), sha256=sha(args.predictions)),
                  generation_manifest=dict(path=str(args.manifest), sha256=sha(args.manifest)),
                  GT_read_only_after_all48_frozen=truth is not None, source_predictions_modified=False,
                  metric_source="remote original PNG only; exported previews are not scored",
                  ROI_policy="frozen native W bounding box + 96 native pixels, clipped to canvas; shared by every arm and seed",
                  webp_quality=90, raw_model_original_count=48, cases=cases, files=files,
                  total_bytes=sum(item["bytes"] for item in files))
    (out / "asset_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(complete=True, files=len(files), bytes=result["total_bytes"])), flush=True)


if __name__ == "__main__":
    main()

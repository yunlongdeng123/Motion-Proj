"""60次预测冻结后导出CPU报告资产；不加载模型、不评分、不改原PNG。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from PIL import Image, ImageFilter

ROIS = {
    "visible_scene081_raw000_v0": [477, 921, 1173, 1280],
    "visible_scene081_raw000_v1": [457, 856, 1153, 1280],
    "visible_scene081_raw050_v0": [477, 818, 1468, 1280],
    "visible_scene081_raw050_v1": [364, 769, 1161, 1263],
    "scene065_raw086": [597, 704, 1270, 1209],
    "scene065_raw110": [512, 751, 1342, 1280],
}
SEEDS = [101, 202, 303, 404, 505]
PROMPTS = ["legacy", "material_match_v1"]


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(record):
    need(Path(record["path"]).is_file() and sha(record["path"]) == record["sha256"], "source pin mismatch: " + record["path"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--parallel-receipt", type=Path)
    parser.add_argument("--smoke-receipt", type=Path)
    parser.add_argument("--truth-config", type=Path, help="optional independent clean-reference export; read only after frozen gate")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    p, gate = read(args.predictions), read(args.gate)
    need(p.get("schema") == "road_2d_teacher_prompt_round_predictions.v1" and p.get("complete") is True, "60 predictions not complete")
    need(p["attempted_diffusion_calls"] == p["completed_diffusion_calls"] == 60 and p["historical_replays_passed"] == 12, "60/12 counts failed")
    need(gate.get("passed") is True and gate.get("jobs") == 60 and gate.get("historical_replays") == 12, "freeze gate not passed")
    need(gate["predictions"]["sha256"] == sha(args.predictions), "gate/predictions mismatch")
    manifest_path = args.manifest or Path(p["manifest"]["path"])
    need(p["manifest"]["sha256"] == sha(manifest_path) == gate["manifest"]["sha256"], "manifest mismatch")
    m = read(manifest_path)
    need(m["schema"] == "road_2d_teacher_prompt_round_plan.v1", "new plan schema required")
    need(m["configuration"]["seeds"] == SEEDS and [x["id"] for x in m["configuration"]["prompts"]] == PROMPTS, "frozen matrix changed")
    need(len(p["jobs"]) == len(m["jobs"]) == 60 and {j["id"] for j in p["jobs"]} == {j["id"] for j in m["jobs"]}, "job matrix mismatch")
    planned = {j["id"]: j for j in m["jobs"]}
    for job in p["jobs"]:
        need(all(job.get(k) == planned[job["id"]][k] for k in ("case_id", "prompt_id", "seed", "arm", "view")), "job metadata mismatch")
        need(job["W_outside_RGB_exact"] and job["K_RGB_exact"], "scope failed")
    replay = [j for j in p["jobs"] if j["historical_replay_expected"]]
    need(len(replay) == 12 and all(j["historical_replay_exact"] and all(j["historical_replay_exact"].values()) for j in replay), "historical full1152 replay failed")
    need(not p["GT_opened"] and p["training_updates"] == 0, "generation scope changed")
    out = args.out.resolve()
    need(not out.exists(), "preserve existing export: " + str(out))
    out.mkdir(parents=True)
    for folder in ("raw_model", "previews", "roi", "inputs"):
        (out / folder).mkdir()
    truth = read(args.truth_config) if args.truth_config else None
    files, cases = [], []

    def record(path, role, source):
        item = dict(path=path.relative_to(out).as_posix(), sha256=sha(path), bytes=path.stat().st_size, role=role, source=source)
        with Image.open(path) as image:
            item["wh"] = list(image.size)
        files.append(item)
        return item["path"]

    def png(image, path, role, source):
        image.save(path, "PNG", compress_level=4)
        return record(path, role, source)

    def preview(image, path, source):
        value = image.copy()
        value.thumbnail((960, 640), Image.Resampling.LANCZOS)
        value.save(path, "WEBP", quality=90, method=4)
        return record(path, "lossy full final/observation preview; not a metric or grain source", source)

    for case in m["cases"]:
        sid = case["id"]
        need(sid in ROIS, "unexpected case")
        check(case["observation"]); check(case["roles_native"])
        with np.load(case["roles_native"]["path"], allow_pickle=False) as z:
            masks = {k: z[k].copy() for k in ("W_output", "H_input", "K")}
        W, H, K = (masks[k] for k in ("W_output", "H_input", "K"))
        need(W.shape == H.shape == K.shape == (1280, 1920), "native grid changed")
        yy, xx = np.where(W)
        computed = [max(0, int(xx.min())-96), max(0, int(yy.min())-96), min(1920, int(xx.max())+97), min(1280, int(yy.max())+97)]
        need(computed == ROIS[sid], "W changed; do not silently reselect ROI")
        window = ROIS[sid]
        observation = Image.open(case["observation"]["path"]).convert("RGB")
        reference = dict(roi=png(observation.crop(window), out/"inputs"/(sid+"_observation_roi.png"), "lossless original observation fixed ROI", case["observation"]),
                         full_preview=preview(observation, out/"inputs"/(sid+"_observation.webp"), case["observation"]))
        role_paths = {}
        for key, mask in (("W", W), ("H", H), ("K", K)):
            role_paths[key] = png(Image.fromarray(mask.astype(np.uint8)*255).crop(window), out/"inputs"/(sid+"_"+key+"_roi.png"), "original native role fixed ROI", case["roles_native"])
        overlay = observation.convert("RGBA")
        for mask, color in ((H, (78,165,255)), (W, (91,225,181)), (K, (244,190,99))):
            imask = Image.fromarray(mask.astype(np.uint8)*255)
            edge = np.asarray(imask.filter(ImageFilter.MaxFilter(5))) != np.asarray(imask.filter(ImageFilter.MinFilter(5)))
            layer = np.zeros((1280,1920,4), np.uint8)
            layer[...,:3] = color; layer[...,3][edge] = 235
            overlay = Image.alpha_composite(overlay, Image.fromarray(layer))
        reference["overlay_roi"] = png(overlay.convert("RGB").crop(window), out/"inputs"/(sid+"_roles_roi.png"), "visible H/W/K outlines; illustration only", case["roles_native"])
        if truth and "clean_rgb" in truth["cases"][sid]:
            source = truth["cases"][sid]["clean_rgb"]
            check(source)
            clean = Image.open(source["path"]).convert("RGB")
            need(clean.size == observation.size, "clean grid changed")
            reference["clean_roi"] = png(clean.crop(window), out/"inputs"/(sid+"_clean_roi.png"), "independent clean reference after freeze; certified road only", source)
        jobs = []
        selected = [j for j in p["jobs"] if j["case_id"] == sid]
        need(len(selected) == 10 and {(j["seed"],j["prompt_id"]) for j in selected} == {(s,prompt) for s in SEEDS for prompt in PROMPTS}, "case does not contain all10 candidates")
        for job in sorted(selected, key=lambda j:(SEEDS.index(j["seed"]),PROMPTS.index(j["prompt_id"]))):
            for name in ("raw_model", "raw_native", "final_native"):
                check(job[name])
            model_path = out/"raw_model"/(job["id"]+".png")
            shutil.copyfile(job["raw_model"]["path"], model_path)
            need(sha(model_path) == job["raw_model"]["sha256"], "raw PNG copy not byte exact")
            raw = Image.open(job["raw_native"]["path"]).convert("RGB")
            final = Image.open(job["final_native"]["path"]).convert("RGB")
            need(raw.size == final.size == observation.size == (1920,1280), "full native grid changed")
            jobs.append(dict(id=job["id"], case_id=sid, prompt_id=job["prompt_id"], seed=job["seed"], arm=job["arm"], model_wh=[1152,768], seconds=job["seconds"],
                raw_model=record(model_path,"byte-exact original1152 generated PNG",job["raw_model"]),
                full_final_preview=preview(final,out/"previews"/(job["id"]+"_final.webp"),job["final_native"]),
                raw_roi=png(raw.crop(window),out/"roi"/(job["id"]+"_raw.png"),"lossless fixed native ROI of raw; upstream bilinear inverse already applied",job["raw_native"]),
                final_roi=png(final.crop(window),out/"roi"/(job["id"]+"_final.png"),"lossless fixed native ROI of final hard-writeback",job["final_native"]),
                historical_replay_expected=job["historical_replay_expected"], historical_replay_exact=job.get("historical_replay_exact"),
                remote_originals={name:job[name] for name in ("raw_model","raw_native","final_native")}))
        cases.append(dict(id=sid,domain=case["domain"],native_hw=case["native_hw"],roi_xyxy=window,reference=reference,masks=role_paths,jobs=jobs))
        print(json.dumps(dict(case=sid,candidates=10,CPU_only=True)),flush=True)
    result = dict(schema="road_2d_teacher_prompt_report_assets.v1",complete=True,CPU_only=True,source_predictions_modified=False,
        predictions=dict(path=str(args.predictions),sha256=sha(args.predictions)), generation_manifest=dict(path=str(manifest_path),sha256=sha(manifest_path)),
        freeze_gate=dict(path=str(args.gate),sha256=sha(args.gate)), prompts=m["configuration"]["prompts"],seeds=SEEDS,
        raw_model_original_count=60,candidate_count=60,default_case="scene065_raw110",default_seed=101,
        ROI_policy="exact same six native windows as2026-11-19 W-bbox+96; never selected by generated output",
        webp_quality=90,metric_source="original remote PNGs only; WebP previews never scored", GT_reference_exported_after_freeze=truth is not None,
        method_human_verdict="accepted_method_current_cases",prompt_human_verdict=None,new_prompt_status="pending_paired_review",
        execution=dict(formal_diffusion_calls=60,engineering_smoke_calls=4,total_diffusion_calls=64,formal_workers=4,keep_open_as_user_requested=True),
        cases=cases,files=files,total_bytes=sum(f["bytes"] for f in files))
    for name,path in (("parallel_receipt",args.parallel_receipt),("smoke_receipt",args.smoke_receipt)):
        if path:
            result[name]=dict(path=str(path),sha256=sha(path),data=read(path))
    (out/"asset_manifest.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(complete=True,original_raw_PNGs=60,files=len(files),bytes=result["total_bytes"])),flush=True)


if __name__ == "__main__":
    main()

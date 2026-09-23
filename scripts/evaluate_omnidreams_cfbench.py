"""异步CPU读出：逐case核对真实图/生成图，保留代理指标，不代填人审。"""
import argparse
import json
import sys
import time
from pathlib import Path
import itertools

import av
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
from skimage.metrics import structural_similarity
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "worldsim_v75"))
from evaluate_localization import model, predict, iou

FRAMES = [0, 9, 15, 24, 33, 42, 51, 60, 69]
CORNERS = np.array(list(itertools.product([-.5, .5], repeat=3)))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


def project(track, frame, camera, K):
    if frame not in track["frames"]:
        return None
    idx = track["frames"].index(frame)
    points = (CORNERS*np.array(track["dimensions"])) @ Rotation.from_quat(track["quaternions"][idx]).as_matrix().T+track["centers"][idx]
    cam = np.linalg.inv(camera)
    points = points @ cam[:3, :3].T+cam[:3, 3]
    if points[:, 2].min() <= .1:
        return None
    pixels = points[:, :2]/points[:, 2, None]*K[:2]+K[2:]
    box = np.r_[pixels.min(0), pixels.max(0)]
    box = np.clip(box, [0, 0, 0, 0], [1280, 704, 1280, 704])
    if np.any(box[2:]-box[:2] < 8):
        return None
    return box.tolist()


def detections(m, rgb):
    pred = predict(m, rgb)
    return [{"box": b.tolist(), "score": float(s), "label": int(c)} for b, s, c in zip(pred["boxes"], pred["scores"], pred["labels"])
            if int(c) in [1, 2, 3, 4, 6, 8] and float(s) >= .5]


def associate(tracks, f, camera, K, detected):
    projections = {t["actor_key"]: project(t, f, camera, K) for t in tracks}
    valid = [(t["actor_key"], projections[t["actor_key"]], t["category"]) for t in tracks if projections[t["actor_key"]]]
    result = {}
    if valid and detected:
        allowed = {"REGULAR_VEHICLE": [3, 6, 8], "TRUCK": [3, 6, 8], "PEDESTRIAN": [1], "BICYCLIST": [2, 4], "OTHER": []}
        matrix = np.array([[iou(box, d["box"]) if d["label"] in allowed.get(cat, []) else 0 for d in detected] for _, box, cat in valid])
        ti, di = linear_sum_assignment(-matrix)
        for a, b in zip(ti, di):
            if matrix[a, b] >= .3:
                key, box, _ = valid[a]
                db = np.array(detected[b]["box"])
                error = np.linalg.norm((db[:2]+db[2:])/2-(np.array(box[:2])+box[2:])/2)
                result[key] = {"detection": detected[b], "iou": float(matrix[a, b]), "center_error_px": float(error)}
    return projections, result


def read_frames(path):
    result = {}
    with av.open(str(path)) as reader:
        for i, frame in enumerate(reader.decode(video=0)):
            if i in FRAMES:
                result[i] = frame.to_ndarray(format="rgb24")
    assert len(result) == len(FRAMES)
    return result


def evaluate(row, out, detector):
    start = time.monotonic()
    src = Path(row["input_dir"])
    tr = np.load(src / "trajectory.npz")
    scenes = {b: json.loads((src / f"{b}-scene.json").read_text()) for b in ["factual", "counterfactual"]}
    videos = {b: read_frames(out / f"{b}.mp4") for b in scenes}
    case, target = row["case"], row["target_key"]
    ego = case["target"]["role"] == "ego"
    family = case["intervention"]["family"]
    source = Path(case["dataset"]["root"]) / case["dataset"]["scene_id"]
    real, records = {}, []
    for f in FRAMES:
        sf = int(round(float(tr["source_frames"][f])))
        real[f] = np.asarray(Image.open(source / f"images/{sf:03d}_{row['camera_index']}.jpg").convert("RGB").resize((1280, 704), Image.Resampling.BOX))
        images = {"real_factual": real[f], "factual": videos["factual"][f], "counterfactual": videos["counterfactual"][f]}
        item = {"frame_30hz": f, "source_frame": sf, "readouts": {}}
        for branch, image in images.items():
            basis = "factual" if branch == "real_factual" else branch
            projected, matched = associate(scenes[basis]["tracks"], f, tr[basis][f], tr["K"], [] if ego else detections(detector, image))
            key = "inserted:"+target if family == "actor_insertion" and branch == "counterfactual" and f >= 15 else target
            own = projected.get(key)
            # 移除后仍检查原框中是否有残留；该额外检测不被当作身份可靠的轨迹。
            original = next((t for t in scenes["factual"]["tracks"] if t["actor_key"] == target), None)
            ghost = project(original, f, tr[basis][f], tr["K"]) if original else None
            item["readouts"][branch] = {"target_projection": own, "original_projection": ghost, "target_match": matched.get(key),
                "all_assignments": matched, "projected_target_visible": own is not None,
                "target_role": "ego" if ego else key, "detector_skipped": ego}
        a = np.array(Image.fromarray(videos["factual"][f]).resize((640, 352)), dtype=np.float32)/255
        b = np.array(Image.fromarray(videos["counterfactual"][f]).resize((640, 352)), dtype=np.float32)/255
        gt = np.array(Image.fromarray(real[f]).resize((640, 352)), dtype=np.float32)/255
        mse = float(np.mean((a-gt)**2))
        item["factual_vs_observed"] = {"psnr_db": -10*np.log10(max(mse, 1e-12)), "ssim_halfres": float(structural_similarity(a, gt, channel_axis=2, data_range=1))}
        if not ego:
            mask = np.ones((704, 1280), bool)
            for branch in ["factual", "counterfactual"]:
                for name in ["target_projection", "original_projection"]:
                    box = item["readouts"][branch][name]
                    if box:
                        x1, y1, x2, y2 = box
                        mask[max(0, int(y1)-16):min(704, int(np.ceil(y2))+16), max(0, int(x1)-16):min(1280, int(np.ceil(x2))+16)] = False
            mask = np.array(Image.fromarray(mask).resize((640, 352), Image.Resampling.NEAREST))
            _, ssim = structural_similarity(a, b, channel_axis=2, data_range=1, full=True)
            item["outside_edit_region"] = {"mae_01": float(np.abs(a-b)[mask].mean()), "ssim_halfres": float(ssim.mean(-1)[mask].mean()),
                "pixel_fraction": float(mask.mean()), "mask_is_bbox_proxy_not_exact_occlusion": True}
        else:
            item["outside_edit_region"] = None
        records.append(item)
        print(json.dumps({"evaluating": row["case_id"], "frame": f}), flush=True)
    review = [0, 15, 33, 51, 69]
    sheet = Image.new("RGB", (1440, 5*288), "#172338")
    draw = ImageDraw.Draw(sheet)
    for ri, f in enumerate(review):
        record = next(x for x in records if x["frame_30hz"] == f)
        for ci, branch in enumerate(["real_factual", "factual", "counterfactual"]):
            image = Image.fromarray(real[f] if ci == 0 else videos[branch][f]).copy()
            d = ImageDraw.Draw(image)
            readout = record["readouts"][branch]
            for key, color in [("original_projection", "#fa5858"), ("target_projection", "#00e8c7")]:
                if readout[key]:
                    d.rectangle(readout[key], outline=color, width=4)
            if readout["target_match"]:
                d.rectangle(readout["target_match"]["detection"]["box"], outline="#ffe36e", width=3)
            sheet.paste(image.resize((480, 264)), (ci*480, ri*288+24))
            draw.text((ci*480+5, ri*288+5), f"{branch} | f={f} t={f/30:.1f}s", fill="white")
    sheet.save(out / "evaluation-review.jpg", quality=93)
    post = [x for x in records if x["frame_30hz"] >= 15]
    summary = {}
    for branch in ["real_factual", "factual", "counterfactual"]:
        r = [x["readouts"][branch] for x in post]
        errors = [x["target_match"]["center_error_px"] for x in r if x["target_match"]]
        summary[branch] = {"target_matches": sum(x["target_match"] is not None for x in r), "sampled_post_frames": len(r),
                           "visible_target_projection_frames": sum(x["projected_target_visible"] for x in r),
                           "matched_target_center_error_median_px": float(np.median(errors)) if errors else None}
    e = [x["outside_edit_region"] for x in post if x["outside_edit_region"]]
    summary["outside_edit_mae_01_mean"] = float(np.mean([x["mae_01"] for x in e])) if e else None
    summary["outside_edit_ssim_mean"] = float(np.mean([x["ssim_halfres"] for x in e])) if e else None
    summary["factual_psnr_db_mean"] = float(np.mean([x["factual_vs_observed"]["psnr_db"] for x in post]))
    summary["factual_ssim_mean"] = float(np.mean([x["factual_vs_observed"]["ssim_halfres"] for x in post]))
    result = {"case_id": row["case_id"], "status": "automatic_readout_complete_ai_review_pending", "scoring_version": "cfbench-six-dimension-v1",
              "input": row, "rows": records, "summary": summary, "ai_preliminary": None, "human_verdict": None,
              "notes": ["2D association proxy, not identity ground truth or 3D ADE", "no target detection is not proof of successful removal", "ego background metrics abstain without reprojection"],
              "wall_s": time.monotonic()-start, "failure_ledger_delta": "none"}
    save(out / "evaluation.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    rows = json.loads((args.inputs / "index.json").read_text())["cases"]
    detector = model()
    torch.set_num_threads(3)
    for row in rows:
        out = args.output / row["case_id"]
        if (out / "evaluation.json").exists():
            continue
        while not (out / "result.json").exists() or json.loads((out / "result.json").read_text())["status"] == "started":
            terminal = args.output / "generate-result.json"
            if terminal.exists() and json.loads(terminal.read_text())["status"] not in ["started", "complete"]:
                raise RuntimeError("生成队列失败，停止等候")
            if not args.watch:
                raise RuntimeError("case尚未生成完成")
            time.sleep(10)
        assert json.loads((out / "result.json").read_text())["status"] == "generation_complete"
        evaluate(row, out, detector)


if __name__ == "__main__":
    main()

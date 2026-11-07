"""Read saved F400/target/observation arrays; no renderer or optimisation.

The ranking/thresholds below are diagnostic displays, not replacement gates.
Run on the AutoDL host with the existing Python+NumPy+Pillow environment.
"""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from PIL import Image, ImageDraw, ImageFont


RAWS = (94, 105, 128, 137)
STATES = ("keep", "delete")
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
ROIS = {94: (180, 230, 360, 320), 105: (150, 230, 380, 320),
        128: (150, 250, 380, 320), 137: (150, 250, 380, 320)}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def snap(path):
    p = Path(path)
    s = p.stat()
    return {"path": str(p), "bytes": s.st_size, "mtime_ns": s.st_mtime_ns,
            "sha256": sha(p)}


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def hwc(a):
    a = np.asarray(a)
    if a.shape == (3, 320, 480):
        return a.transpose(1, 2, 0).astype(np.float64)
    assert a.shape == (320, 480, 3), a.shape
    return a.astype(np.float64)


def mv(a):
    return None if a.size == 0 else float(np.mean(a, dtype=np.float64))


def maxv(a):
    return None if a.size == 0 else float(np.max(a))


def metric(base, cand, target, gt, mask, dy):
    count = int(mask.sum())
    if count == 0:
        return {"pixels": 0}
    db = np.abs(base[mask] - target[mask])
    df = np.abs(cand[mask] - target[mask])
    return {"pixels": count, "rgb_mean_abs_movement": mv(np.abs(cand[mask] - base[mask])),
            "rgb_max_abs_movement": maxv(np.abs(cand[mask] - base[mask])),
            "luma_delta_mean": mv(dy[mask]), "luma_delta_min": float(dy[mask].min()),
            "darkened_more_than_0p01": int((dy[mask] < -0.01).sum()),
            "darkened_more_than_0p05": int((dy[mask] < -0.05).sum()),
            "baseline_target_mae": mv(db), "F_target_mae": mv(df),
            "baseline_observed_mae": mv(np.abs(base[mask] - gt[mask])),
            "F_observed_mae": mv(np.abs(cand[mask] - gt[mask]))}


def rgba_image(a):
    # The metrics retain native floats, including HDR/negative values.
    return Image.fromarray(np.rint(np.clip(a, 0, 1) * 255).astype(np.uint8), "RGB")


def top_rows(dy, mask, base, cand, target, obs, source, keep37, state37, limit=20):
    ids = np.flatnonzero(mask)
    if not len(ids):
        return []
    ids = ids[np.argsort(dy.ravel()[ids], kind="stable")[:limit]]
    rows = []
    for idx in ids:
        y, x = np.unravel_index(int(idx), dy.shape)
        v = {"x": int(x), "y": int(y), "luma_delta": float(dy[y, x]),
             "rgb_mean_delta": float((cand[y, x] - base[y, x]).mean()),
             "observed_rgb": obs["rgb"][y, x].tolist(), "target_rgb": target[y, x].tolist(),
             "baseline_rgb": base[y, x].tolist(), "F400_rgb": cand[y, x].tolist(),
             "source_fusion_rgb": source["fusion_rgb"][y, x].tolist(),
             "fit": bool(source["fit_mask"][y, x]), "road": bool(obs["road"][y, x]),
             "road_edit": bool(obs["road_edit"][y, x]),
             "full37": bool(obs["actor_labels"][y, x] == 5),
             "any_actor": bool(obs["actor_labels"][y, x] > 0),
             "actor_label": int(obs["actor_labels"][y, x]),
             "visible37": bool(obs["visible"][1, y, x]),
             "KEEP37_alpha_T": float(keep37[y, x]), "this_state37_alpha_T": float(state37[y, x]),
             "fusion_replace": bool(source["fusion_replace_mask"][y, x]),
             "atlas_support": bool(source["atlas_support_mask"][y, x]),
             "edgeblend_weight": float(source["edgeblend_weight"][y, x]),
             "target_observed_max_abs": float(np.abs(target[y, x] - obs["rgb"][y, x]).max()),
             "target_error_before": float(np.abs(base[y, x] - target[y, x]).mean()),
             "target_error_after": float(np.abs(cand[y, x] - target[y, x]).mean())}
        rows.append(v)
    return rows


def make_board(out, raw, obs, source, base, cand, full37, dy):
    roi = ROIS[raw]
    x0, y0, x1, y1 = roi
    scale = 3
    w, h = (x1-x0)*scale, (y1-y0)*scale
    images = [("Observed RGB", hwc(obs["rgb"])), ("C shared target", hwc(source["edgeblend_rgb"])),
              ("Latest DELETE", base), ("F400 DELETE", cand)]
    board = Image.new("RGB", (4*w, h+76), "white")
    d = ImageDraw.Draw(board)
    d.text((8, 5), "raw%d crop %s; nearest x%d; display clips [0,1]; magenta=full actor37 boundary" % (raw, roi, scale), fill="black")
    edge = full37 & ~(np.roll(full37, 1, 0) & np.roll(full37, -1, 0) & np.roll(full37, 1, 1) & np.roll(full37, -1, 1))
    for col, (label, a) in enumerate(images):
        im = np.rint(np.clip(a, 0, 1)*255).astype(np.uint8)
        im[edge] = [255, 0, 255]
        tile = Image.fromarray(im).crop(roi).resize((w, h), Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST)
        board.paste(tile, (col*w, 54))
        d.text((col*w+8, 30), label, fill="black")
    p = out / ("raw%03d_delete_identity_crop.png" % raw)
    board.save(p)
    # Native-size unclipped arrays remain in the source NPZ; this heatmap is diagnostic.
    dark = np.clip(-dy / 0.15, 0, 1)
    heat = np.stack([dark, np.zeros_like(dark), np.zeros_like(dark)], -1)
    hp = out / ("raw%03d_delete_darkening_heatmap.png" % raw)
    rgba_image(heat).save(hp)
    return [{"path": str(p), "sha256": sha(p), "size": list(board.size), "crop_xyxy": list(roi),
             "boundary_overlay": "actor_labels==5; display only; no scoring domain changed"},
            {"path": str(hp), "sha256": sha(hp), "size": [480, 320],
             "red_scale": "red=-delta_encoded_luma/0.15 clipped to [0,1]; display only"}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--obs", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configpath = args.root / "F_formal400_r1/executed_config.json"
    config = json.loads(configpath.read_text())
    sources = [configpath, Path(__file__)]
    for raw in RAWS:
        sources += [Path(config["targets"][str(raw)]["path"]), args.obs/("%03d.npz" % raw)]
        for state in STATES:
            sources += [args.root / ("baseline4/frames/%03d/%s_float.npz" % (raw, state)),
                        args.root / ("F_formal400_r1/evaluation/frames/%03d/%s_float.npz" % (raw, state))]
    before = [snap(p) for p in sources]
    report = {"schema": "supervision_identity_audit_v1", "status": "COMPLETED_CPU_ARRAY_AUDIT_NO_METHOD_CHANGE",
              "time_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(__file__),
              "actual_new_gpu_forwards": 0, "actual_new_backwards": 0, "actual_new_optimizer_updates": 0,
              "actor_label_mapping": {"1":5, "2":26, "3":18, "4":6, "5":37},
              "luma_definition": "diagnostic encoded RGB dot [0.2126,0.7152,0.0722]; native float, no clipping in statistics",
              "notes": ["All four frames are current BUILD views; the shared C target is used in both KEEP/DELETE.",
                        "Darkening thresholds and top20 are post-result diagnostic summaries, not registered gates.",
                        "Observed actor labels identify observed body pixels; unlabeled dark pixels are not thereby proven shadows.",
                        "Original visible-road/guard means cannot alone certify the removed actor footprint.",
                        "No hidden clean-delete ground truth exists here; road_edit_rgb is only an existing proxy."],
              "frames": [], "figures": []}
    for raw in RAWS:
        tp = Path(config["targets"][str(raw)]["path"])
        assert sha(tp) == config["targets"][str(raw)]["sha256"]
        source, obs = load(tp), load(args.obs/("%03d.npz" % raw))
        target, gt, O = hwc(source["edgeblend_rgb"]), hwc(obs["rgb"]), hwc(source["observation_rgb"])
        assert np.array_equal(O, gt)
        fit, labels = source["fit_mask"].astype(bool), obs["actor_labels"]
        assert int(fit.sum()) == config["targets"][str(raw)]["fit_pixels"]
        bases = {s: load(args.root/("baseline4/frames/%03d/%s_float.npz" % (raw,s))) for s in STATES}
        cands = {s: load(args.root/("F_formal400_r1/evaluation/frames/%03d/%s_float.npz" % (raw,s))) for s in STATES}
        keep37 = bases["keep"]["group_alpha_T"][1]
        full37, vis37 = labels == 5, obs["visible"][1]
        domains = {"fit": fit, "full37": full37, "any_actor": labels > 0, "visible37": vis37,
                   "road": obs["road"], "road_edit": obs["road_edit"],
                   "KEEP37W_positive": keep37 > 0, "KEEP37W_gt0p5": keep37 > .5,
                   "fit_and_full37": fit & full37, "fit_and_any_actor": fit & (labels > 0),
                   "fit_and_visible37": fit & vis37, "fit_and_KEEP37W_positive": fit & (keep37 > 0),
                   "fit_and_KEEP37W_gt0p5": fit & (keep37 > .5),
                   "outside_fit_full37": ~fit & full37,
                   "outside_fit_road_edit": ~fit & obs["road_edit"]}
        roi = np.zeros_like(fit)
        x0,y0,x1,y1=ROIS[raw]; roi[y0:y1,x0:x1] = True
        fr = {"raw": raw, "target_source": config["targets"][str(raw)],
              "source_observation_obs_rgb_full_image_max_abs": float(abs(O-gt).max()),
              "domain_pixels": {k:int(m.sum()) for k,m in domains.items()},
              "fit_actor_label_counts": {str(i):int((fit & (labels==i)).sum()) for i in range(6)},
              "target_observation_by_domain": {}, "states": [], "diagnostic_roi_xyxy": list(ROIS[raw])}
        for name, mask in domains.items():
            fr["target_observation_by_domain"][name] = {
                "pixels": int(mask.sum()), "mae": mv(np.abs(target[mask]-O[mask])),
                "max_abs": maxv(np.abs(target[mask]-O[mask])),
                "fusion_replace_pixels": int((mask & source["fusion_replace_mask"]).sum()),
                "max_edgeblend_weight": maxv(source["edgeblend_weight"][mask])}
        for state in STATES:
            b, f = bases[state], cands[state]
            base, cand = hwc(b["rgb"]), hwc(f["rgb"])
            dy = (cand-base) @ LUMA
            unchanged = {k: bool(np.array_equal(b[k], f[k])) for k in
                         ("alpha", "depth_sum", "depth_m", "group_alpha_T", "added_owner_rgb_T")}
            assert all(unchanged.values())
            st = {"state":state, "geometry_group_and_semantic_RGB_exact": unchanged,
                  "actor37_group_alpha_max_base":float(b["group_alpha_T"][1].max()),
                  "actor37_group_alpha_max_F400":float(f["group_alpha_T"][1].max()),
                  "domains": {k:metric(base,cand,target,gt,m,dy) for k,m in domains.items()},
                  "top20_darkening_all_pixels":top_rows(dy,np.ones_like(fit),base,cand,target,obs,source,keep37,b["group_alpha_T"][1]),
                  "top20_darkening_diagnostic_roi":top_rows(dy,roi,base,cand,target,obs,source,keep37,b["group_alpha_T"][1]),
                  "top20_darkening_full37":top_rows(dy,full37,base,cand,target,obs,source,keep37,b["group_alpha_T"][1])}
            fr["states"].append(st)
            if state == "delete":
                report["figures"] += make_board(args.output,raw,obs,source,base,cand,full37,dy)
        report["frames"].append(fr)
    after = [snap(p) for p in sources]
    assert before == after
    report["source_integrity"] = {"status":"PASS_ALL_INPUT_SIZE_MTIME_SHA_UNCHANGED", "files":before}
    report["array_fact_summary"] = {
        "fit_intersection_all_actor_body_pixels": sum(x["domain_pixels"]["fit_and_any_actor"] for x in report["frames"]),
        "fit_intersection_full37_body_pixels": sum(x["domain_pixels"]["fit_and_full37"] for x in report["frames"]),
        "fit_intersection_visible37_pixels": sum(x["domain_pixels"]["fit_and_visible37"] for x in report["frames"]),
        "all_delete_actor37_native_group_alpha_max_zero": all(s["actor37_group_alpha_max_base"] == s["actor37_group_alpha_max_F400"] == 0 for x in report["frames"] for s in x["states"] if s["state"] == "delete"),
        "direct_labelled_vehicle_body_supervision": "NOT_PRESENT_IN_ACTUAL_FIT_MASK",
        "appearance_coupling_into_deleted_body_or_shadow_regions": "ASSESS_SAVED_CHANGES_AND_FIGURES; no renderer rerun in this audit"}
    outjson = args.output.parent / (args.output.name+".json")
    outjson.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(outjson),"sha256":sha(outjson),"summary":report["array_fact_summary"],
                      "figures":len(report["figures"])},ensure_ascii=False))


if __name__ == "__main__":
    main()

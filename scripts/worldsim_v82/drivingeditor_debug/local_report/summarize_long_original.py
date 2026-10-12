"""CPU-only independent aggregation and true-PNG audit for original B1 at32k.

Read canonical region statistics generated from raw float32 arrays. Recompute
equal-frame and pixel-count-weighted aggregates; never derive scores from MP4.
Does not write task/automation, import training or use a GPU.
"""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from PIL import Image

HERE = Path(__file__).resolve().parent
CASE = "original30k_to32000"
RUN = HERE / "continuation_diagnostic" / CASE
REMOTE = "/root/autodl-tmp/drivingeditor_repro/campaign/vgpu_bugfix_20261012/continuation_diagnostic/" + CASE + "/"
FIXED5 = ["image_0/0017" + str(i).zfill(3) for i in (0, 10, 20, 30, 48)]
ALL49 = ["image_0/0017" + str(i).zfill(3) for i in range(49)]
DOMAINS = ("full", "non_sky_outside_actor", "actor")


def load(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize_domain(frames, domain):
    rows = [(f["regions"][domain]["pixel_count"], f["regions"][domain]["metrics"]) for f in frames]
    rows = [(n, m) for n, m in rows if n]
    total = sum(n for n, _ in rows)
    output = {"nonempty_frame_count": len(rows), "pixel_count_summed": total, "equal_frame_mean": {}, "pixel_weighted_mean": {}}
    if not total:
        return output
    metrics = {
        "composite": ("clamped_rgb_mae_vs_gt", "clamped_rgb_mse_vs_gt", "clamped_rgb_psnr_db"),
        "static": ("clamped_rgb_mae_vs_gt", "clamped_rgb_mse_vs_gt", "clamped_rgb_psnr_db"),
        "alpha_d": ("mean", "fraction_gt_0_1", "fraction_gt_0_5", "fraction_gt_0_9"),
        "alpha_s": ("mean", "fraction_gt_0_1", "fraction_gt_0_5", "fraction_gt_0_9"),
        "dynamic": ("mean", "rgb_support_gt2_uint8_pixel_fraction"),
        "static_env_contribution": ("mean", "fraction_lt0", "fraction_gt1"),
        "static_foreground_contribution": ("mean", "fraction_lt0", "fraction_gt1"),
        "environment_sample": ("mean", "fraction_lt0", "fraction_gt1"),
    }
    for channel, fields in metrics.items():
        output["equal_frame_mean"][channel] = {}
        output["pixel_weighted_mean"][channel] = {}
        for field in fields:
            values = [m[channel][field] for _, m in rows]
            if any(v is None for v in values):
                output["equal_frame_mean"][channel][field] = None
                output["pixel_weighted_mean"][channel][field] = None
                continue
            output["equal_frame_mean"][channel][field] = mean(values)
            if field != "clamped_rgb_psnr_db":
                output["pixel_weighted_mean"][channel][field] = sum(n * m[channel][field] for n, m in rows) / total
        if channel in ("composite", "static"):
            mse = output["pixel_weighted_mean"][channel]["clamped_rgb_mse_vs_gt"]
            output["pixel_weighted_mean"][channel]["pooled_mse_psnr_db"] = -10 * math.log10(mse) if mse else None
    return output


def extract(frame):
    full = frame["regions"]["full"]["metrics"]
    outside = frame["regions"]["non_sky_outside_actor"]["metrics"]
    return {
        "frame_id": frame["frame_id"],
        "composite_psnr_full": full["composite"]["clamped_rgb_psnr_db"],
        "static_psnr_observed_non_sky_outside_actor": outside["static"]["clamped_rgb_psnr_db"],
        "dynamic_alpha_observed_non_sky_outside_actor": outside["alpha_d"]["mean"],
        "static_alpha_observed_non_sky_outside_actor": outside["alpha_s"]["mean"],
        "dynamic_rgb_support_observed_non_sky_outside_actor": outside["dynamic"]["rgb_support_gt2_uint8_pixel_fraction"],
        "observed_non_sky_outside_actor_pixels": frame["regions"]["non_sky_outside_actor"]["pixel_count"],
    }


def main():
    status = load(RUN / "status.json")
    completed = load(RUN / "continuation_diagnostic.json")
    authoritative = load(RUN / "receipts/full49_summary.json")
    assert status["status"] == "complete" and status["actual_exit_code"] == 0
    assert completed["step_count"] == authoritative["step_count"] == 2000
    assert completed["stop_iteration"] == 32000 and completed["horizon"] == 50000
    assert completed["original_algorithm_changed"] is False and authoritative["normal_algorithm_success_claimed"] is False
    assert authoritative["first_step_original10_exact"] and authoritative["all4_node_eval_RNG_preserved"]
    assert authoritative["harness_exit_code"] == authoritative["wrapper_exit_code"] == 0
    steps = [json.loads(line) for line in (RUN / "step_losses.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert [row["iteration"] for row in steps] == list(range(30001, 32001))
    nodes = []
    final = None
    for iteration in (30001, 30300, 31000, 32000):
        path = RUN / "extra_eval" / str(iteration) / "diagnostic_eval.json"
        data = load(path)
        assert data["iteration"] == iteration and data["all_rng_unchanged"]
        expected = ALL49 if iteration == 32000 else FIXED5
        assert [f["frame_id"] for f in data["frames"]] == expected
        fixed = [extract(f) for f in data["frames"] if f["frame_id"] in FIXED5]
        nodes.append({"iteration": iteration, "evaluated_frame_count": len(data["frames"]), "fixed5_frame_ids": [f["frame_id"] for f in fixed], "fixed5_equal_frame_means": {key: mean(f[key] for f in fixed) for key in fixed[0] if key not in ("frame_id", "observed_non_sky_outside_actor_pixels")}, "diagnostic_eval_sha256": sha(path), "all_rng_unchanged": True})
        if iteration == 32000:
            final = data
    domains = {domain: summarize_domain(final["frames"], domain) for domain in DOMAINS}
    comparisons = []
    for domain in ("full", "non_sky_outside_actor"):
        ours = domains[domain]
        theirs = authoritative["final49_regions"][domain]
        assert ours["nonempty_frame_count"] == theirs["frame_count"] == 49
        assert ours["pixel_count_summed"] == theirs["pixel_count_summed"]
        for channel in ("composite", "static"):
            for lhs, rhs in (
                (ours["equal_frame_mean"][channel]["clamped_rgb_psnr_db"], theirs[channel]["mean_frame_psnr_db"]),
                (ours["pixel_weighted_mean"][channel]["clamped_rgb_mse_vs_gt"], theirs[channel]["pooled_mse"]),
                (ours["pixel_weighted_mean"][channel]["pooled_mse_psnr_db"], theirs[channel]["pooled_psnr_db"]),
            ):
                assert abs(lhs - rhs) < 1e-12
        for channel in ("alpha_s", "alpha_d", "environment_sample", "static_foreground_contribution", "static_env_contribution", "dynamic"):
            assert abs(ours["pixel_weighted_mean"][channel]["mean"] - theirs[channel]["mean"]) < 1e-12
        assert abs(ours["pixel_weighted_mean"]["dynamic"]["rgb_support_gt2_uint8_pixel_fraction"] - theirs["dynamic"]["rgb_support_gt2_uint8_pixel_fraction"]) < 1e-12
        comparisons.append({"domain": domain, "authoritative_frame_mean_and_pixel_weighted_aggregates_match": True})
    pngs = []
    for pin in authoritative["artifact_pins"]:
        if "/extra_eval/32000/" not in pin["path"] or "channel" not in pin:
            continue
        assert pin["path"].startswith(REMOTE)
        path = RUN / pin["path"][len(REMOTE):]
        assert path.stat().st_size == pin["bytes"] and sha(path) == pin["sha256"]
        with Image.open(path) as image:
            assert image.size == (480, 320)
            image.load()
        pngs.append({"relative_path": str(path.relative_to(HERE)).replace("\\", "/"), "sha256": pin["sha256"], "frame_id": pin["frame_id"], "channel": pin["channel"]})
    assert len(pngs) == 294 and len({(r["channel"], r["frame_id"]) for r in pngs}) == 294
    result = {
        "status": "B1_LONG_COMPLETE_CPU_AGGREGATION_AND_MEDIA_INPUT_VERIFIED_NOT_ALGORITHM_ACCEPTANCE",
        "case": CASE,
        "arm": "B1",
        "window": [30001, 32000],
        "steps": 2000,
        "horizon": 50000,
        "source": "Canonical B1 diagnostic_eval region statistics computed from uncompressed raw float32 model results; this CPU script independently recomputes aggregation and verifies294 actual exported PNG hashes/dimensions. It does not re-render or re-load remote NPZ/checkpoints.",
        "aggregation": {
            "equal_frame_mean": "Arithmetic mean of the49 nonempty per-view region metrics. This matches the preregistered primary summaries.",
            "pixel_weighted_mean": "sum(region_pixel_count*per_frame_stat)/sum(region_pixel_count). Regions have different areas, so outside-actor means and support need this explicit label.",
            "pooled_mse_psnr_db": "-10*log10(pixel-count-weighted per-frame MSE); not an average of per-frame dB and not pixel-weighted per-frame dB.",
            "support_definition": "Any channel floor(clip(dynamic_rgb,0,1)*255+0.5)>2; eval dynamic is the foreground channel. This is a low nonblack-support cutoff, not an opacity threshold or actor segmentation accuracy."
        },
        "domains": domains,
        "nodes_same_fixed5": nodes,
        "per_frame_terminal": [extract(f) for f in final["frames"]],
        "authoritative_aggregate_comparisons": comparisons,
        "verified_PNG_count": 294,
        "verified_PNG_sha256": pngs,
        "receipts": {str(path.relative_to(HERE)).replace("\\", "/"): sha(path) for path in (RUN / "receipts/full49_summary.json", RUN / "status.json", RUN / "continuation_diagnostic.json", RUN / "step_losses.jsonl", RUN / "terminal_32000/diagnostic_snapshot_manifest.json", RUN / "extra_eval/32000/diagnostic_eval.json", RUN / "receipts/local_sync_receipt.json")},
        "interpretation": "Original-math B1 recovers composite reconstruction over2000 steps but still has poor static/background separation and broad dynamic RGB support. B2 repeat and V comparison are not complete in this receipt; do not combine32k B1 with30300 V as a paired result.",
        "short30300_pair_modified": False,
        "GPU_used": False,
        "normal_algorithm_success_claimed": False,
        "script_sha256": sha(__file__)
    }
    out = HERE / "long_original_summary_cpu.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "equal_frame_full_composite_psnr": domains["full"]["equal_frame_mean"]["composite"]["clamped_rgb_psnr_db"], "equal_frame_observed_static_psnr": domains["non_sky_outside_actor"]["equal_frame_mean"]["static"]["clamped_rgb_psnr_db"], "equal_frame_outside_dynamic_support": domains["non_sky_outside_actor"]["equal_frame_mean"]["dynamic"]["rgb_support_gt2_uint8_pixel_fraction"], "pixel_weighted_outside_dynamic_support": domains["non_sky_outside_actor"]["pixel_weighted_mean"]["dynamic"]["rgb_support_gt2_uint8_pixel_fraction"], "verified_PNG_count": len(pngs), "summary_sha256": sha(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

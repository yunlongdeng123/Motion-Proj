"""Read-only rendering audit for the original Scene008 measured proxy ledger.

Run on the remote host in the existing raster environment. This program does
not fit parameters, change a ledger, or retire old rows. Its only optional write
is the explicitly selected JSON output path.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
import torch


def distribution(values):
    value = np.asarray(values, dtype=np.float64).reshape(-1)
    value = value[np.isfinite(value)]
    if not len(value):
        return {"count": 0, "sum": 0.0, "mean": None}
    return {
        "count": int(len(value)), "sum": float(value.sum()),
        "mean": float(value.mean()), "min": float(value.min()),
        "max": float(value.max()),
        "quantile_0_10_25_50_75_90_100": np.quantile(
            value, [0, .1, .25, .5, .75, .9, 1]).tolist(),
    }


def to_cpu(value):
    return value.detach().cpu().numpy()


def add_values(collection, name, values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    collection.setdefault(name, []).append(values)


def summarize(collection):
    return {name: distribution(np.concatenate(values))
            for name, values in collection.items()}


def depth_contributions(trace, window, strength, measured_depth, panel, scale):
    """Exact live alpha*T partition at the measured z on selected panel rays.

    The old actor, fixed PVD, and newly added measured rows retain their actual
    joint transmittance. Depth residual contributions add to D-z_measured*A;
    dividing their sum by A recovers signed expected-depth error.
    """
    selected = torch.as_tensor(panel, device="cuda").reshape(-1)[trace.pixels.long()]
    counts_all = (trace.offsets[1:] - trace.offsets[:-1]).long()
    slots = torch.where(selected & (counts_all > 0))[0]
    counts = counts_all[slots]
    hit_count = int(counts.sum())
    if not hit_count:
        return {}
    bounds = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    ray = torch.arange(len(slots), device="cuda").repeat_interleave(counts, output_size=hit_count)
    position = torch.arange(hit_count, device="cuda")
    source = trace.offsets[slots[ray]].long() + position - bounds[ray]
    rows = window.ids[source].long()
    alpha = window.alpha.clone()
    if strength is not None:
        changed = alpha[window.indices] * (1 - (window.mass @ strength).clamp(0, 1))
        alpha[window.indices] = torch.where(changed >= 1 / 255, changed, 0.0)
    alpha = alpha[source].double()
    logs = torch.cat((alpha.new_zeros(1), torch.log1p(-alpha).cumsum(0)))
    log_start = logs[bounds[ray]]
    transmittance = (logs[:-1] - log_start).exp()
    live = (logs[1:] - log_start).exp() >= .0001
    weight = torch.where(live, alpha * transmittance, 0.0)
    pixels = trace.pixels.long()[slots]
    target = torch.as_tensor(measured_depth, device="cuda").reshape(-1)[pixels].double()
    z = trace.feature[rows, 0].double() / scale
    center_z = trace.depth[rows].double() / scale
    residual = z - target[ray]
    before_measured = center_z < target[ray]
    kind = trace.kind[rows]
    parts = {
        "all": torch.ones_like(before_measured),
        "old_prefix": (kind > 0) & (kind < 3) & before_measured,
        "old_suffix": (kind > 0) & (kind < 3) & ~before_measured,
        "pvd": kind == 3,
        "new_prefix": (kind == 0) & before_measured,
        "new_suffix": (kind == 0) & ~before_measured,
    }
    output = {}
    for name, take in parts.items():
        mass = target.new_zeros(len(slots))
        depth_error = target.new_zeros(len(slots))
        mass.index_add_(0, ray, weight * take)
        depth_error.index_add_(0, ray, weight * residual * take)
        output[name + "_alpha_T"] = to_cpu(mass)
        output[name + "_signed_depth_residual_sum_m"] = to_cpu(depth_error)
    alpha_image = output["all_alpha_T"]
    output["recomposed_signed_expected_depth_error_m"] = (
        output["all_signed_depth_residual_sum_m"] / np.maximum(alpha_image, 1e-8))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign", type=Path,
        default=Path("/root/autodl-tmp/drivingeditor_repro/campaign"))
    parser.add_argument("--raws", default="105,113,130",
                        help="Comma separated raw IDs, or all for native49")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--joint", action="store_true",
                        help="Also rerender all four black arms to inspect signed joint depth residuals")
    parser.add_argument("--joint-raws", default="105,113,130",
                        help="Prespecified raws for the optional joint contribution decomposition")
    args = parser.parse_args()
    source = args.campaign / "scene008_a6_heuristic_validation_20261105/source"
    sys.path.insert(0, str(source))
    from rescue import RUN, RAWS, HELD, OBS, Trace
    from positive import Positive
    from heuristic_evaluate import panel_mask
    from ray_support import padded

    raw_ids = list(RAWS) if args.raws == "all" else [int(r) for r in args.raws.split(",")]
    if not raw_ids or any(r not in RAWS for r in raw_ids):
        raise ValueError("Select native49 raw IDs only")
    joint_raws = [int(r) for r in args.joint_raws.split(",")]
    if any(r not in RAWS for r in joint_raws):
        raise ValueError("Select native49 joint raw IDs only")
    ledger_path = args.ledger or RUN / "positive_ledger.npz"
    with np.load(ledger_path, allow_pickle=False) as bank:
        ledger = {key: bank[key] for key in bank.files}
    torch.set_num_threads(1)
    pools = {"all": {}, "build": {}, "held": {}}
    results = []
    with torch.no_grad():
        experiment = Positive()
        experiment.load_proxy(ledger)
        if experiment.proxy is None:
            raise ValueError("Empty proxy ledger")
        xyz = np.asarray(ledger["xyz"], dtype=np.float64)
        for raw in raw_ids:
            data = experiment.tr.raw(raw)
            # Match the precise construction used by Probe.merged, including
            # the 24-column feature ABI, while excluding every parent row.
            rows = experiment.proxy.raster_rows(
                data["pose"], data["cam"], experiment.tr.n.args,
                experiment.tr.n.static.get_max_sh_channels,
                bool(experiment.tr.n.args.convert_SHs_python))
            rows["opacities"] = experiment.proxy_strength[:, None]
            feature = rows["features"].new_zeros((len(xyz), 4))
            feature[:, :2] = rows["features"]
            rows["features"] = padded(dict(rows, features=feature))
            kind = torch.zeros(len(xyz), device="cuda", dtype=torch.int32)
            trace = Trace(experiment, raw, rows, kind, experiment.tr.black, 2)
            image = trace.base()
            alpha = to_cpu(image["alpha"][0])
            depth = to_cpu(image["depth_sum"][0]) / np.maximum(alpha, 1e-8) / experiment.tr.scale
            rgb = to_cpu(image["rgb"])
            with np.load(OBS / "frames" / f"{raw:03d}.npz", allow_pickle=False) as bank:
                measured_depth = bank["depth_m"][0].copy()
                measured_valid = bank["depth_valid"][0].copy()
                measured_rgb = bank["rgb"].copy()
                matrix = bank["world_to_camera"] @ bank["world_from_local"][0]
                intrinsic = bank["K"].copy()
                raw_model_pose = bank["model_from_local"][0].copy()
            camera = xyz @ matrix[:3, :3].T + matrix[:3, 3]
            expected_z = camera[:, 2]
            projected = camera @ intrinsic.T
            expected_xy = projected[:, :2] / expected_z[:, None]
            native_xy = to_cpu(trace.xy).reshape(-1, 2)
            native_z = to_cpu(trace.depth).reshape(-1) / experiment.tr.scale
            feature_z = to_cpu(rows["features"][:, 0]) / experiment.tr.scale
            active = to_cpu(trace.radii) > 0
            # Native NDC-to-pixel coordinates put the pixel center at the integer
            # raster coordinate; observation-bank rays use x+.5 and y+.5.
            delta_half_pixel = native_xy - (expected_xy - .5)
            delta_no_shift = native_xy - expected_xy
            xy = np.floor(expected_xy).astype(np.int64)
            x, y = xy.T
            inside = (expected_z > .1) & (x >= 0) & (x < trace.w) & (y >= 0) & (y < trace.h)
            x, y = x.clip(0, trace.w - 1), y.clip(0, trace.h - 1)
            matched = inside & measured_valid[y, x]
            anchor_error = expected_z[matched] - measured_depth[y[matched], x[matched]]
            panel = to_cpu(panel_mask(experiment, raw)).astype(bool)
            scope = "held" if raw in HELD else "build"
            values = {}
            add_values(values, "runtime_xy_half_pixel_l2_px", np.linalg.norm(delta_half_pixel[active], axis=1))
            add_values(values, "runtime_xy_no_shift_l2_px", np.linalg.norm(delta_no_shift[active], axis=1))
            add_values(values, "runtime_center_depth_error_m", native_z[active] - expected_z[active])
            add_values(values, "feature_center_depth_error_m", feature_z - expected_z)
            add_values(values, "anchor_center_signed_depth_error_m", anchor_error)
            add_values(values, "anchor_center_abs_depth_error_m", np.abs(anchor_error))
            add_values(values, "panel_proxy_alpha", alpha[panel])
            add_values(values, "proxy_radius_px", to_cpu(trace.radii)[active])
            add_values(values, "proxy_center_depth_m", expected_z)
            add_values(values, "panel_measured_depth_m", measured_depth[panel])
            for region_name, region in (("panel", panel), ("a6_measured", measured_valid)):
                for cutoff in (.02, .2, .5):
                    valid = region & (alpha >= cutoff)
                    prefix = f"{region_name}_alpha_ge_{cutoff:g}"
                    error = depth[valid] - measured_depth[valid]
                    add_values(values, prefix + "_signed_depth_error_m", error)
                    add_values(values, prefix + "_abs_depth_error_m", np.abs(error))
                    add_values(values, prefix + "_premultiplied_rgb_mae",
                               np.abs(rgb[:, valid].T - measured_rgb[valid]).mean(1))
                    # Shape-independent color: reveal whether a black background
                    # is the only reason the premultiplied RGB is dark.
                    add_values(values, prefix + "_unpremultiplied_rgb_mae",
                               np.abs(rgb[:, valid].T / alpha[valid, None] - measured_rgb[valid]).mean(1))
            if args.joint and raw in joint_raws:
                before, cut, old_trace, old_window = experiment.raster_pair(raw, "black", False)
                proxy_only, rescue, new_trace, new_window = experiment.raster_pair(raw, "black", True)
                for arm, package, joint_trace, joint_window, strength in (
                    ("parent", before, old_trace, old_window, None),
                    ("cut", cut, old_trace, old_window, experiment.retire),
                    ("proxy_only", proxy_only, new_trace, new_window, None),
                    ("rescue", rescue, new_trace, new_window, experiment.retire),
                ):
                    joint_alpha = to_cpu(package["alpha"][0])
                    joint_depth = (to_cpu(package["depth_sum"][0]) /
                                   np.maximum(joint_alpha, 1e-8) / experiment.tr.scale)
                    joint_rgb = to_cpu(package["rgb"])
                    for region_name, region in (("panel", panel),
                                                ("panel_proxy_covered", panel & (alpha >= .02))):
                        error = joint_depth[region] - measured_depth[region]
                        prefix = f"joint_{arm}_{region_name}"
                        add_values(values, prefix + "_signed_depth_error_m", error)
                        add_values(values, prefix + "_abs_depth_error_m", np.abs(error))
                        add_values(values, prefix + "_rgb_mae",
                                   np.abs(joint_rgb[:, region].T - measured_rgb[region]).mean(1))
                    contribution = depth_contributions(
                        joint_trace, joint_window, strength, measured_depth,
                        panel, experiment.tr.scale)
                    for key, value in contribution.items():
                        add_values(values, f"joint_{arm}_panel_{key}", value)
            item = {
                "raw": raw, "held": raw in HELD,
                "active_proxy_rows": int(active.sum()),
                "panel_measured_pixels": int(panel.sum()),
                "runtime_pose_raw_pose_maxabs": float(np.max(np.abs(to_cpu(data["pose"]) - raw_model_pose))),
                "model_scale": float(experiment.tr.scale),
                "pose_column_scale": np.linalg.norm(raw_model_pose[:3, :3], axis=0).tolist(),
                "stats": summarize(values),
            }
            results.append(item)
            for name, chunks in values.items():
                for scope_name in ("all", scope):
                    for value in chunks:
                        add_values(pools[scope_name], name, value)
            print("AUDIT_FRAME", raw, json.dumps({
                "xy_error_px": item["stats"]["runtime_xy_half_pixel_l2_px"]["max"],
                "feature_z_error_m": item["stats"]["feature_center_depth_error_m"]["max"],
                "panel_depth_mae_m": item["stats"]["panel_alpha_ge_0.02_abs_depth_error_m"]["mean"],
                "panel_proxy_alpha": item["stats"]["panel_proxy_alpha"]["mean"],
            }), flush=True)
    saved = json.loads((RUN / "positive_rgb_evaluation.json").read_text())
    current = {
        scope: {arm: {key: saved["stats"][scope][arm]["black"][key]
                      for key in ("panel_rgb", "panel_depth", "panel_coverage")}
                for arm in ("parent", "cut", "proxy_only", "rescue")}
        for scope in ("all", "build", "held")
    }
    output = {
        "status": "AUDITED_NO_PARAMETER_CHANGES", "raw_ids": raw_ids,
        "joint_rerender": args.joint,
        "joint_raw_ids": [r for r in raw_ids if args.joint and r in joint_raws],
        "ledger": str(ledger_path), "proxy_rows": len(xyz),
        "interpretation": (
            "Isolated proxy depth is measured only where proxy alpha meets each threshold; "
            "saved joint metrics cover the full panel and are not identical denominators. "
            "Runtime XY is checked against raw pinhole projection minus half a pixel."),
        "stats": {scope: summarize(pool) for scope, pool in pools.items()},
        "frames": results, "saved_joint_black_panel": current,
    }
    if args.out:
        args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print("AUDIT_SAVED", args.out, flush=True)
    else:
        print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()

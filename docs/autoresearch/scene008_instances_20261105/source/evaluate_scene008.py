"""Paired Scene008 metrics on observation-only masks and fixed geometry probes."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


RAWS = tuple(range(89, 138))
PILOT_RAWS = (105, 113, 130)
GROUPS = ("strong_baseline", "objects", "replacement", "pvd")
STATES = ("unedited", "delete_a37", "a6_only", "a37_only")
COMPONENTS = ("a6", "a37", "road", "legacy")
PROBE_NAMES = (
    "a6_measured_visible", "a6_observed_reprojected_unknown",
    "a6_pvd_generated_unknown", "ground_beneath_a37_proxy",
)
COVERAGE_THRESHOLD = 0.1


def _numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def make_packet(out, scale, tag_names):
    """Convert native CHW and weighted model depths to the evaluation packet."""
    alpha = _numpy(out["alpha"])[0]
    tags = _numpy(out["component_alpha_T"])
    if tags.shape[0] != len(tag_names):
        raise ValueError("Group tag names do not match the raster channels")
    z = _numpy(out["depth_sum"])[0]
    weights = {key: None for key in COMPONENTS}
    weights.update(zip(tag_names, tags))
    packet = dict(
        rgb=np.moveaxis(_numpy(out["rgb"]), 0, -1),
        alpha=alpha,
        depth=np.divide(z, alpha * scale, out=np.zeros_like(z), where=alpha > 1e-8),
        weights=weights,
    )
    if "component_depth_sum" in out:
        component_z = _numpy(out["component_depth_sum"])
        packet["component_depth_m"] = {
            key: np.divide(z, w * scale, out=np.zeros_like(z), where=w > 1e-8)
            for key, z, w in zip(tag_names, component_z, tags)
        }
    if "rgb_premultiplied" in out:
        packet["rgb_premultiplied"] = np.moveaxis(_numpy(out["rgb_premultiplied"]), 0, -1)
    return packet


def _selected_raws(bank, raw_ids):
    if tuple(bank.raw_ids) != RAWS:
        raise ValueError("Evaluation requires the fixed FRONT raw089..137 observation bank")
    raws = PILOT_RAWS + tuple(raw for raw in RAWS if raw not in PILOT_RAWS) if raw_ids is None else tuple(raw_ids)
    if not raws or len(set(raws)) != len(raws) or set(raws) - set(RAWS):
        raise ValueError("Evaluation raws must be unique and drawn from raw089..137")
    return raws


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FixedProbes:
    """Read-only common geometry hypotheses and matched historical probe provenance."""

    def __init__(self, path):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as archive:
            if archive["schema"].item() != "scene008.fixed_a6_probe.v1":
                raise ValueError("Unexpected probe schema")
            if tuple(archive["raw_ids"]) != RAWS or tuple(archive["layer_names"]) != PROBE_NAMES:
                raise ValueError("Unexpected probe frames or layer names")
            self.k = archive["K"]
            self.camera = archive["model_to_camera"]
            self.timestamps = archive["model_time"]
            self.a6_pose = archive["model_from_canonical"]
            self.depth = archive["depth_m"]
        if self.depth.shape != (len(RAWS), len(PROBE_NAMES), 320, 480):
            raise ValueError("Unexpected probe resolution")
        if not np.isfinite(self.depth).all() or (self.depth < 0).any():
            raise ValueError("Invalid probe depth")

    def for_frame(self, raw, frame):
        index = raw - RAWS[0]
        for name, value in (("K", self.k[index]), ("model_to_camera", self.camera[index])):
            if not np.allclose(frame[name], value, rtol=0, atol=1e-6):
                raise ValueError(f"Fixed probe {name} disagrees with the evaluation bank at raw{raw}")
        if not np.allclose(frame["model_from_local"][0], self.a6_pose[index], rtol=0, atol=1e-6):
            raise ValueError(f"Fixed probe a6 pose disagrees with the evaluation bank at raw{raw}")
        if "model_time" in frame:
            timestamp = np.asarray(frame["model_time"])
            if not np.isclose(timestamp, np.asarray(self.timestamps[index], dtype=timestamp.dtype), rtol=0, atol=1e-8):
                raise ValueError(f"Fixed probe time disagrees with the evaluation bank at raw{raw}")
        return dict(zip(PROBE_NAMES[1:], self.depth[index, 1:]))


def _map(value, shape, label):
    value = _numpy(value)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{label} has invalid shape or non-finite values")
    return value


def _packet(raw_packet, shape, group, state):
    packet = dict(raw_packet)
    packet["rgb"] = _map(packet["rgb"], (*shape, 3), "RGB").clip(0, 1)
    if packet.get("depth") is not None:
        packet["depth"] = _map(packet["depth"], shape, "Camera-z")
        if (packet["depth"] < 0).any():
            raise ValueError("Negative rendered camera-z")
    weights = dict(packet["weights"])
    packet["weights"] = weights
    required = COMPONENTS[1:] if group == "strong_baseline" else COMPONENTS
    if any(weights.get(key) is None for key in required):
        raise ValueError(f"Missing joint contribution channel in {group}/{state}")
    for key, value in weights.items():
        if key not in COMPONENTS:
            raise ValueError(f"Unexpected contribution channel {key}")
        if value is not None:
            value = weights[key] = _map(value, shape, key)
            if (value < -1e-5).any() or (value > 1.00001).any():
                raise ValueError(f"Invalid contribution in {group}/{state}/{key}")
    if state == "delete_a37" and weights["a37"] is not None and np.abs(weights["a37"]).max() > 1e-5:
        raise ValueError(f"Deleted a37 still contributes in {group}")
    if state.endswith("_only"):
        owner = state.removesuffix("_only")
        if any(np.abs(value).max() > 1e-5 for key, value in weights.items() if key != owner and value is not None):
            raise ValueError(f"An isolated render still includes other components in {group}/{state}")
    if packet.get("alpha") is not None:
        alpha = _map(packet["alpha"], shape, "alpha")
        sum_error = np.abs(sum(value for value in weights.values() if value is not None) - alpha).max()
        if sum_error > 1e-4:
            raise ValueError(f"Group tags do not conserve joint alpha in {group}/{state}: {sum_error}")
        packet["conservation_max_abs"] = float(sum_error)
    for key, value in packet.get("component_depth_m", {}).items():
        if key not in COMPONENTS or (_map(value, shape, f"{key} z") < 0).any():
            raise ValueError(f"Invalid component camera-z in {group}/{state}/{key}")
    return packet


class Metrics:
    def __init__(self):
        self.rows = []
        self.pooled = {}

    def add(self, raw, group, state, region, metric, values, samples=None):
        key = (group, state, region, metric)
        if values is None:
            count, total = 0, 0.0
        else:
            count = int(values.size)
            total = float(values.sum(dtype=np.float64))
        samples = count if samples is None else int(samples)
        self.rows.append(dict(raw=raw, group=group, state=state, region=region, metric=metric,
                              sum=total, count=count, samples=samples,
                              mean=total / count if count else None))
        pool = self.pooled.setdefault(key, [0.0, 0, 0])
        pool[0] += total
        pool[1] += count
        pool[2] += samples

    def write(self, output, details):
        with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        pooled = {
            "/".join(key): dict(sum=total, count=count, samples=samples, mean=total / count if count else None)
            for key, (total, count, samples) in self.pooled.items()
        }
        summary = dict(details, pooled=pooled)
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return summary


def _observed_regions(frame):
    a6, a37 = frame["visible"]
    if (a6 & a37).any():
        raise ValueError("Fixed visible instance masks overlap")
    regions = {"a6_visible": a6, "a37_visible": a37, "road_observed": frame["road"],
               "non_target": frame["protect"]}
    if "actor_labels" in frame:
        regions["other_vehicles"] = frame["protect"] & np.isin(frame["actor_labels"], (1, 2, 3))
    return regions


def _road_edit(frame):
    if "road_edit" not in frame:
        return None
    region = frame["road_edit"] & ~frame["box_candidate"][0] & ~frame["visible_reviewed"][0]
    region &= ~frame["possible_occluded"][0]
    if "actor_labels" in frame:
        region &= (frame["actor_labels"] == 0) | (frame["actor_labels"] == 5)
    return region


def _record_depth(metrics, ids, name, target_depth, prediction, weight):
    samples = len(target_depth)
    if prediction is None:
        metrics.add(*ids, f"{name}/depth_support", None, samples)
        metrics.add(*ids, f"{name}/depth_mae_m", None, samples)
        return
    supported = (prediction > 0) & (weight > COVERAGE_THRESHOLD)
    metrics.add(*ids, f"{name}/depth_support", supported, samples)
    metrics.add(*ids, f"{name}/depth_mae_m", np.abs(prediction[supported] - target_depth[supported]), samples)


def _record_render(metrics, raw, group, state, packet, frame, probes, prior_full, baseline):
    truth = frame["rgb"]
    rgb, depth, weights = packet["rgb"], packet.get("depth"), packet["weights"]
    alpha = packet.get("alpha")
    alpha = np.ones(rgb.shape[:2], dtype=np.float32) if alpha is None else alpha
    isolated = state.endswith("_only")
    if isolated:
        owner = state.removesuffix("_only")
        pre = packet.get("rgb_premultiplied")
        if pre is not None:
            rgb = np.divide(pre, alpha[..., None], out=np.zeros_like(pre),
                            where=alpha[..., None] > 1e-8).clip(0, 1)
    regions = _observed_regions(frame)
    rgb_error = np.abs(rgb - truth).mean(2)
    for name, mask in regions.items():
        owner = name.partition("_")[0]
        if (state == "delete_a37" and owner == "a37") or (isolated and owner + "_only" != state):
            continue
        ids = (raw, group, state, name)
        rgb_valid = mask & (alpha > COVERAGE_THRESHOLD) if isolated else mask
        metrics.add(*ids, "rgb_mae", rgb_error[rgb_valid], mask.sum())
        if prior_full is not None:
            metrics.add(*ids, "edit_rgb_delta", np.abs(rgb - prior_full["rgb"]).mean(2)[mask])
        if baseline is not None:
            baseline_error = np.abs(baseline["rgb"] - truth).mean(2)
            metrics.add(*ids, "paired_rgb_mae_delta_from_baseline", (rgb_error - baseline_error)[mask])
            metrics.add(*ids, "rgb_change_from_baseline", np.abs(rgb - baseline["rgb"]).mean(2)[mask])
        for key, value in weights.items():
            metrics.add(*ids, "contribution_" + key, value[mask] if value is not None else None, mask.sum())
        if owner in ("a6", "a37"):
            a = 0 if owner == "a6" else 1
            measured = mask & frame["depth_valid"][a]
            reference = frame["depth_m"][a, measured]
            _record_depth(metrics, ids, "measured", reference,
                          depth[measured] if depth is not None else None, alpha[measured])
            object_depth = packet.get("component_depth_m", {}).get(owner)
            _record_depth(metrics, ids, "owner_measured", reference,
                          object_depth[measured] if object_depth is not None else None,
                          weights[owner][measured] if weights[owner] is not None else None)
            if baseline is not None and depth is not None and baseline.get("depth") is not None:
                base_alpha = baseline.get("alpha")
                base_supported = base_alpha > COVERAGE_THRESHOLD if base_alpha is not None else True
                pair = measured & (depth > 0) & (alpha > COVERAGE_THRESHOLD)
                pair &= (baseline["depth"] > 0) & base_supported
                ref = frame["depth_m"][a, pair]
                delta = np.abs(depth[pair] - ref) - np.abs(baseline["depth"][pair] - ref)
                metrics.add(*ids, "measured/paired_depth_mae_delta_from_baseline_m", delta, measured.sum())
    if not isolated:
        _record_occlusion(metrics, raw, group, state, packet, frame, probes)
    road_edit = _road_edit(frame) if state == "delete_a37" else None
    if road_edit is not None:
        ids = (raw, group, state, "observed_recovered_road_proxy")
        err = np.abs(rgb[road_edit] - frame["road_edit_rgb"][road_edit]).mean(1)
        metrics.add(*ids, "rgb_mae", err)
        metrics.add(*ids, "contribution_road", weights["road"][road_edit])
        road_depth = frame["road_edit_depth_m"]
        measured = road_edit & (road_depth > 0) & np.isfinite(road_depth)
        _record_depth(metrics, ids, "road_proxy", road_depth[measured],
                      depth[measured] if depth is not None else None, alpha[measured])
        if baseline is not None:
            base_error = np.abs(baseline["rgb"][road_edit] - frame["road_edit_rgb"][road_edit]).mean(1)
            metrics.add(*ids, "paired_rgb_mae_delta_from_baseline", err - base_error)


def _record_occlusion(metrics, raw, group, state, packet, frame, probes):
    weights = packet["weights"]
    for actor, free in zip(("a6", "a37"), frame["free"]):
        contribution = weights.get(actor)
        metrics.add(raw, group, state, actor + "_confirmed_free", "forbidden_contribution",
                    contribution[free] if contribution is not None else None, free.sum())
    overlap = frame["occluded"][0] & frame["visible"][1]
    ids = (raw, group, state, "a37_in_front_of_a6_evidence")
    for key, value in weights.items():
        metrics.add(*ids, "contribution_" + key, value[overlap] if value is not None else None, overlap.sum())
    if all(k in packet.get("component_depth_m", {}) for k in ("a6", "a37")):
        both = overlap & (weights["a6"] > COVERAGE_THRESHOLD) & (weights["a37"] > COVERAGE_THRESHOLD)
        depth = packet["component_depth_m"]
        metrics.add(*ids, "both_owners_supported", both[overlap])
        metrics.add(*ids, "a6_group_mean_in_front_of_a37", (depth["a6"] < depth["a37"])[both], overlap.sum())
    for name, z in probes.items():
        mask = z > 0
        owner = "road" if name == "ground_beneath_a37_proxy" else "a6"
        ids = (raw, group, state, name)
        for key, value in weights.items():
            metrics.add(*ids, "contribution_" + key, value[mask] if value is not None else None, mask.sum())
        depth = packet.get("component_depth_m", {}).get(owner)
        if weights[owner] is None:
            continue
        w = weights[owner]
        metrics.add(*ids, "low_owner_support", (w < COVERAGE_THRESHOLD)[mask])
        if depth is not None and state == "delete_a37":
            valid = mask & (w > COVERAGE_THRESHOLD) & (depth > 0)
            # A diagnostic against a fixed proxy, never a hidden-truth accuracy score.
            metrics.add(*ids, "supported_owner_minus_probe_abs_m", np.abs(depth - z)[valid], mask.sum())
            for axis in (0, 1):
                za, zb = (depth[:-1], depth[1:]) if axis == 0 else (depth[:, :-1], depth[:, 1:])
                va, vb = (valid[:-1], valid[1:]) if axis == 0 else (valid[:, :-1], valid[:, 1:])
                pa, pb = (mask[:-1], mask[1:]) if axis == 0 else (mask[:, :-1], mask[:, 1:])
                metrics.add(*ids, f"supported_adjacent_depth_jump_{axis}_m",
                            np.abs(za - zb)[va & vb], (pa & pb).sum())


def _record_common_depth(metrics, raw, frame, rendered):
    for state in STATES[:2]:
        available = {group: states[state] for group, states in rendered.items()
                     if states[state].get("depth") is not None}
        if len(available) != len(rendered):
            continue
        for actor_index, actor in enumerate(("a6", "a37")):
            if state == "delete_a37" and actor == "a37":
                continue
            measured = frame["visible"][actor_index] & frame["depth_valid"][actor_index]
            common = measured.copy()
            for packet in available.values():
                common &= packet["depth"] > 0
                if packet.get("alpha") is not None:
                    common &= packet["alpha"] > COVERAGE_THRESHOLD
            target = frame["depth_m"][actor_index, common]
            for group, packet in available.items():
                metrics.add(raw, group, state, actor + "_visible", "measured/common_depth_mae_m",
                            np.abs(packet["depth"][common] - target), measured.sum())


def _temporal_frame(frame, probes, rendered):
    if "a6_pvd_generated_unknown" not in probes:
        return None
    camera_from_local = frame["model_to_camera"] @ frame["model_from_local"][0]
    scale = np.linalg.norm(camera_from_local[:3, 0])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid local-to-camera scale in temporal evaluation")
    mask = probes["a6_pvd_generated_unknown"] > 0
    models = {}
    for group, states in rendered.items():
        deleted = states["delete_a37"]
        depth = deleted.get("component_depth_m", {}).get("a6")
        if depth is not None:
            models[group] = (depth, mask & (deleted["weights"]["a6"] > COVERAGE_THRESHOLD) & (depth > 0))
    return dict(camera_from_local=camera_from_local, k=frame["K"], scale=scale, mask=mask, models=models)


def _record_temporal(metrics, raw, old, new):
    """Compare rendered a6 depth after moving source points with the frozen a6 pose."""
    inv_k = np.linalg.inv(old["k"])
    new_from_old = new["camera_from_local"] @ np.linalg.inv(old["camera_from_local"])
    h, w = new["mask"].shape
    for group, (depth, support) in old["models"].items():
        if group not in new["models"]:
            continue
        y, x = np.nonzero(support)
        rays = np.column_stack((x + .5, y + .5, np.ones_like(x))) @ inv_k.T
        source = rays * (depth[support] * old["scale"])[:, None]
        target = source @ new_from_old[:3, :3].T + new_from_old[:3, 3]
        target = target[target[:, 2] > 0]
        pixels = target @ new["k"].T
        pixels = np.floor(pixels[:, :2] / pixels[:, 2:]).astype(np.int64)
        x, y = pixels.T
        inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        x, y, z = x[inside], y[inside], target[inside, 2] / new["scale"]
        probe = new["mask"][y, x]
        new_depth, new_support = new["models"][group]
        valid = new_support[y, x]
        ids = (raw, group, "delete_a37", "a6_pvd_generated_unknown")
        metrics.add(*ids, "temporal_next_probe_support", valid[probe])
        metrics.add(*ids, "temporal_next_depth_reprojection_abs_m",
                    np.abs(new_depth[y[valid], x[valid]] - z[valid]), probe.sum())


def _save_image(path, image):
    Image.fromarray(np.rint(image.clip(0, 1) * 255).astype(np.uint8)).save(path)


def _save_frame(output, raw, frame, rendered):
    folder = output / "frames" / f"{raw:03d}"
    folder.mkdir(parents=True)
    _save_image(folder / "observed.png", frame["rgb"])
    for group, states in rendered.items():
        for state, packet in states.items():
            stem = f"{group}_{state}"
            _save_image(folder / (stem + ".png"), packet["rgb"])
            # Colors identify a6 (cyan), a37 (orange), road (green) and old scene (gray).
            colors = np.array([[0, .8, 1], [1, .35, 0], [.15, .8, .15], [.5, .5, .5]], np.float32)
            tags = sum(packet["weights"][key][..., None] * color for key, color in zip(COMPONENTS, colors)
                       if packet["weights"].get(key) is not None)
            _save_image(folder / (stem + "_contribution.png"), tags)


def evaluate(bank, render, output, raw_ids=None, *, groups=GROUPS, probe_path=None, save_images=True):
    """Evaluate a callback render(group, state, raw) against one frozen bank.

    The native baseline can use None for its unresolved a6 contribution and its
    absent a6-only rendering. Missing new states, tags or camera bindings fail.
    """
    groups = tuple(groups)
    if not groups or len(set(groups)) != len(groups) or set(groups) - set(GROUPS):
        raise ValueError("Unknown or duplicate comparison controls")
    raws = _selected_raws(bank, raw_ids)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    fixed_probes = FixedProbes(probe_path) if probe_path is not None else None
    metrics = Metrics()
    max_conservation = {}
    pending_temporal = {}
    evaluated_raws = set()
    selected = set(raws)
    for index, raw in enumerate(raws, 1):
        frame = bank.frame(raw)
        shape = frame["rgb"].shape[:2]
        probes = fixed_probes.for_frame(raw, frame) if fixed_probes else {}
        rendered = {}
        for group in groups:
            states = rendered.setdefault(group, {})
            for state in STATES:
                data = render(group, state, raw)
                if data is None and group == "strong_baseline" and state == "a6_only":
                    continue
                if data is None:
                    raise ValueError(f"Missing {group}/{state} at raw{raw}")
                packet = _packet(data, shape, group, state)
                states[state] = packet
                if "conservation_max_abs" in packet:
                    max_conservation[group] = max(max_conservation.get(group, 0), packet["conservation_max_abs"])
        for group, states in rendered.items():
            for state, packet in states.items():
                baseline = rendered.get("strong_baseline", {}).get(state)
                if group == "strong_baseline" or state.endswith("_only"):
                    baseline = None
                prior_full = states["unedited"] if state == "delete_a37" else None
                _record_render(metrics, raw, group, state, packet, frame, probes, prior_full, baseline)
        _record_common_depth(metrics, raw, frame, rendered)
        temporal = _temporal_frame(frame, probes, rendered)
        if temporal is not None:
            for other in (raw - 1, raw + 1):
                if other in pending_temporal:
                    pair = (pending_temporal[other], temporal) if other < raw else (temporal, pending_temporal[other])
                    _record_temporal(metrics, min(raw, other), *pair)
            evaluated_raws.add(raw)
            pending_temporal[raw] = temporal
            for other in (raw - 1, raw, raw + 1):
                if {other - 1, other + 1} & (selected - evaluated_raws):
                    continue
                pending_temporal.pop(other, None)
        if save_images:
            _save_frame(output, raw, frame, rendered)
        (output / "progress.json").write_text(json.dumps(dict(last_raw=raw, evaluated=index,
                                                              requested=len(raws))) + "\n", encoding="utf-8")
    return metrics.write(output, dict(
        schema="scene008.instance_comparison.v1", raw_ids=raws, groups=groups, states=STATES,
        observation_bank=str(bank.root) if hasattr(bank, "root") else None,
        observation_receipt_sha256=_sha256(bank.root / "receipt.json") if hasattr(bank, "root") else None,
        coverage_threshold=COVERAGE_THRESHOLD, max_group_conservation_error=max_conservation,
        probe_path=str(fixed_probes.path) if fixed_probes else None,
        probe_sha256=_sha256(fixed_probes.path) if fixed_probes else None,
        comparison_scope="Same-scene observed FRONT views. Unknown proxy metrics are diagnostics.",
        depth_contract="Camera-z metres, expected Gaussian depth after alpha normalization.",
        hidden_rgb_gt=False, hidden_geometry_gt=False,
    ))


def _load_packet(path):
    with np.load(path, allow_pickle=False) as data:
        packet = {key: data[key] for key in ("rgb", "depth", "alpha", "rgb_premultiplied") if key in data}
        packet["weights"] = {key: data[f"w_{key}"] if f"w_{key}" in data else None for key in COMPONENTS}
        packet["component_depth_m"] = {key: data[f"z_{key}"] for key in COMPONENTS if f"z_{key}" in data}
        return packet


def audit_history(bank, baseline_dir, dual_dir, output, *, raw_ids=None, probe_path=None):
    """Re-score historical images on the new bank, keeping 8-bit provenance explicit."""
    paths = {
        "strong_baseline_png": (Path(baseline_dir), "repaired_full.png", "repaired_delete37.png", "gt_opencv.png"),
        "dual_union_png": (Path(dual_dir), "composite.png", "delete_a37.png", "gt.png"),
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    metrics = Metrics()
    max_gt_delta = {}
    raws = _selected_raws(bank, raw_ids)
    probes = FixedProbes(probe_path) if probe_path is not None else None
    for raw in raws:
        frame = bank.frame(raw)
        if probes is not None:
            probes.for_frame(raw, frame)
        truth = frame["rgb"]
        for name, (root, full_name, delete_name, truth_name) in paths.items():
            images = []
            for file in (full_name, delete_name, truth_name):
                with Image.open(root / f"{raw:03d}" / file) as image:
                    images.append(np.asarray(image.convert("RGB"), dtype=np.float32) / 255)
            source_delta = float(np.abs(images[2] - truth).max())
            if source_delta > 2 / 255 + 1e-6:
                raise ValueError(f"Historical RGB observation differs at raw{raw}, {name}: {source_delta}")
            max_gt_delta[name] = max(max_gt_delta.get(name, 0), source_delta)
            for state, image in zip(STATES[:2], images):
                for region, mask in _observed_regions(frame).items():
                    if state == "delete_a37" and region == "a37_visible":
                        continue
                    metrics.add(raw, name, state, region, "rgb_mae", np.abs(image - truth).mean(2)[mask])
                    if state == "delete_a37":
                        metrics.add(raw, name, state, region, "edit_rgb_delta",
                                    np.abs(image - images[0]).mean(2)[mask])
            road_edit = _road_edit(frame)
            if road_edit is not None:
                metrics.add(raw, name, "delete_a37", "observed_recovered_road_proxy", "rgb_mae",
                            np.abs(images[1][road_edit] - frame["road_edit_rgb"][road_edit]).mean(1))
    return metrics.write(output, dict(
        schema="scene008.historical_rgb_audit.v1", raw_ids=raws, max_archived_observation_delta=max_gt_delta,
        observation_bank=str(bank.root) if hasattr(bank, "root") else None,
        observation_receipt_sha256=_sha256(bank.root / "receipt.json") if hasattr(bank, "root") else None,
        probe_path=str(probes.path) if probes else None,
        probe_sha256=_sha256(probes.path) if probes else None,
        roots={name: str(path[0]) for name, path in paths.items()},
        precision="Archived 8-bit rendered PNG vs common floating-point original RGB",
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=GROUPS)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--packets", type=Path, help="Packaged renders at <root>/<group>/<raw>/<state>.npz")
    source.add_argument("--history", nargs=2, metavar=("R3_BASELINE_PNGS", "DUAL_UNION_PNGS"))
    args = parser.parse_args()
    from object_observations import ObservationBank

    bank = ObservationBank(args.observations)
    if args.history:
        audit_history(bank, *args.history, args.output, raw_ids=PILOT_RAWS if args.pilot else None,
                      probe_path=args.probe)
        return

    def render(group, state, raw):
        path = args.packets / group / f"{raw:03d}" / f"{state}.npz"
        return _load_packet(path) if path.is_file() else None

    evaluate(bank, render, args.output, PILOT_RAWS if args.pilot else None,
             groups=args.groups, probe_path=args.probe, save_images=not args.no_images)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Zero-training Gaussian delete/move/insert-copy probe on official DGGT Waymo mode 2.

The Gaussian edit and RGB target association here are local experiment code, not
an author-provided DGGT editing CLI. No GT depth, pose, or dynamic mask is used
to predict geometry or choose the target. The official dataset's sky mask is
used in precisely the mode-2 non-sky test.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


class NeedsTargetSelection(RuntimeError):
    """The RGB-derived evidence did not identify one vehicle in every frame."""


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--external", type=Path, required=True, help="Official DGGT repository root")
    p.add_argument("--data-root", type=Path, required=True, help="Preprocessed Waymo root")
    p.add_argument("--scene", required=True, help="One Waymo scene directory, e.g. 001")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True, help="New run directory; existing path is rejected")
    p.add_argument("--sequence-length", type=int, default=4)
    p.add_argument("--edit-config", type=Path, help="JSON target selection and move configuration")
    p.add_argument("--official-trace", type=Path, help="Actual official mode-2 tensor trace; reuse its prediction and sky without another model forward")
    p.add_argument("--check-only", action="store_true", help="CPU contract check; do not import torch or create output")
    return p.parse_args()


def _config(path: Path | None) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8")) if path else {}
    if not isinstance(cfg, dict):
        raise ValueError("edit config must be a JSON object")
    cfg.setdefault("selection", {})
    cfg.setdefault("move", {})
    if not isinstance(cfg["selection"], dict) or not isinstance(cfg["move"], dict):
        raise ValueError("selection and move must be JSON objects")
    sel = cfg["selection"]
    sel.setdefault("source", "custom_masks")
    sel.setdefault("min_area_pixels", 100)
    sel.setdefault("max_area_fraction", 0.25)
    sel.setdefault("min_iou", 0.01)
    sel.setdefault("max_3d_distance_widths", 3.0)
    sel.setdefault("neighbor_radius_widths", 0.02)
    cfg["move"].setdefault("right_widths", 1.0)
    if sel["source"] not in ("custom_masks", "semantic_masks", "masks", "boxes"):
        raise ValueError("selection.source must be custom_masks, semantic_masks, masks, or boxes")
    if sel["source"] == "semantic_masks":
        sel.setdefault("labels", [13, 14, 15])  # RGB SegFormer car/truck/bus
    if sel["source"] == "custom_masks":
        sel.setdefault("labels", [40])  # official RGB-derived custom_masks vehicle
    if sel["source"] == "masks" and sel.get("provenance") != "rgb_derived":
        raise ValueError("explicit masks require selection.provenance='rgb_derived'; GT dynamic masks are forbidden")
    if sel["source"] == "boxes" and sel.get("provenance") != "rgb_manual":
        raise ValueError("explicit boxes require selection.provenance='rgb_manual'")
    if "anchor_normalized_xy" in sel:
        anchor = sel["anchor_normalized_xy"]
        if sel["source"] not in ("custom_masks", "semantic_masks") or not isinstance(anchor, list) or len(anchor) != 2 or not all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in anchor):
            raise ValueError("anchor_normalized_xy must be [x,y] in [0,1] for RGB semantic component selection")
    if "roi_box_normalized_xyxy" in sel:
        box = sel["roi_box_normalized_xyxy"]
        if sel["source"] not in ("custom_masks", "semantic_masks") or not isinstance(box, list) or len(box) != 4 or not all(isinstance(x, (int, float)) for x in box) or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
            raise ValueError("roi_box_normalized_xyxy must be [x0,y0,x1,y1] within [0,1] for RGB semantic masks")
    if float(sel["neighbor_radius_widths"]) < 0 or float(cfg["move"]["right_widths"]) <= 0:
        raise ValueError("neighbor radius must be nonnegative and move.right_widths positive")
    return cfg


def _image_files(data_root: Path, scene: str, count: int) -> list[Path]:
    folder = data_root / scene / "images"
    paths = sorted(p for p in folder.glob("*") if p.name.endswith(("_0.jpg", "_0.png")))
    if len(paths) < count:
        raise ValueError(f"scene {scene}: need {count} camera-0 RGB frames, found {len(paths)} in {folder}")
    return paths[:count]


def _target_size(image_path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as im:
        w, h = im.size
    resized_h = round(h * (518 / w) / 14) * 14
    return min(resized_h, 518), 518


def _load_label(path: Path, size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor companion to official 518-wide RGB resize/center crop."""
    from PIL import Image

    with Image.open(path) as im:
        # Palette PNGs store class IDs in pixel indices; convert('L') would
        # replace those IDs with palette luminance and silently change labels.
        if im.mode not in ("P", "L", "I", "I;16"):
            im = im.convert("L")
        target_h = round(im.height * (518 / im.width) / 14) * 14
        im = im.resize((518, target_h), Image.Resampling.NEAREST)
        if target_h > 518:
            top = (target_h - 518) // 2
            im = im.crop((0, top, 518, top + 518))
        arr = np.asarray(im)
    if arr.shape != size:
        raise ValueError(f"{path}: processed mask size {arr.shape} does not match RGB {size}")
    return arr


def _mask_paths(folder: Path, image_paths: list[Path]) -> list[Path]:
    paths = []
    for rgb in image_paths:
        candidates = [folder / (rgb.stem + ext) for ext in (".png", ".jpg", ".jpeg")]
        found = next((p for p in candidates if p.is_file()), None)
        if found is None:
            raise NeedsTargetSelection(f"no RGB-derived semantic mask for {rgb.name} under {folder}")
        paths.append(found)
    return paths


def _rgb_roi_mask(image_path: Path, size: tuple[int, int], box: list[float]) -> np.ndarray:
    """Map an original-RGB normalized box through the official resize/crop."""
    from PIL import Image

    with Image.open(image_path) as im:
        original_w, original_h = im.size
    resized_h = round(original_h * (518 / original_w) / 14) * 14
    crop_top = max(0, (resized_h - 518) // 2)
    h, w = size
    x0, y0, x1, y1 = box
    left = max(0, min(w, int(np.floor(x0 * w))))
    right = max(0, min(w, int(np.ceil(x1 * w))))
    top = max(0, min(h, int(np.floor(y0 * resized_h)) - crop_top))
    bottom = max(0, min(h, int(np.ceil(y1 * resized_h)) - crop_top))
    roi = np.zeros(size, dtype=bool)
    roi[top:bottom, left:right] = True
    return roi


def _rgb_anchor_pixel(image_path: Path, size: tuple[int, int], anchor: list[float]) -> tuple[int, int]:
    """Map an original-RGB normalized click to the model's resized/cropped RGB."""
    from PIL import Image

    with Image.open(image_path) as im:
        original_w, original_h = im.size
    resized_h = round(original_h * (518 / original_w) / 14) * 14
    crop_top = max(0, (resized_h - 518) // 2)
    x = min(size[1] - 1, int(anchor[0] * size[1]))
    y = int(anchor[1] * resized_h) - crop_top
    if not 0 <= y < size[0]:
        raise NeedsTargetSelection("needs_target_selection: RGB anchor falls outside official center crop")
    return x, y


def _label_maps(sel: dict, data_root: Path, scene: str, images: list[Path], size: tuple[int, int]) -> tuple[list[np.ndarray], list[str]]:
    source = sel["source"]
    if source in ("custom_masks", "semantic_masks"):
        folder = Path(sel.get("directory", data_root / scene / "custom_masks"))
        paths = _mask_paths(folder, images)
        labels = [int(x) for x in sel["labels"]]
        maps = [np.isin(_load_label(p, size), labels) for p in paths]
        if "roi_box_normalized_xyxy" in sel:
            maps = [mask & _rgb_roi_mask(rgb, size, sel["roi_box_normalized_xyxy"])
                    for mask, rgb in zip(maps, images)]
        return maps, [str(p) for p in paths]
    if source == "masks":
        paths = [Path(x) for x in sel.get("paths", [])]
        if len(paths) != len(images):
            raise ValueError("selection.paths must contain one RGB-derived binary mask per input frame")
        if any("dynamic_mask" in str(p).lower() for p in paths):
            raise ValueError("GT dynamic-mask paths are not legal edit selectors")
        return [_load_label(p, size) > 0 for p in paths], [str(p) for p in paths]
    boxes = sel.get("xyxy_normalized", [])
    if len(boxes) != len(images):
        raise ValueError("selection.xyxy_normalized must contain one RGB bbox per input frame")
    h, w = size
    masks = []
    for box in boxes:
        if len(box) != 4 or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
            raise ValueError("each normalized RGB box must be [x0,y0,x1,y1] within [0,1]")
        x0, y0, x1, y1 = box
        mask = np.zeros((h, w), dtype=bool)
        mask[int(y0 * h):max(int(y0 * h) + 1, int(np.ceil(y1 * h))), int(x0 * w):max(int(x0 * w) + 1, int(np.ceil(x1 * w)))] = True
        masks.append(mask)
    return masks, ["manual RGB bbox" for _ in masks]


def _components(binary: np.ndarray, min_area: int, max_area: int) -> list[np.ndarray]:
    from scipy import ndimage

    labels, n = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    out = []
    for i in range(1, n + 1):
        m = labels == i
        if min_area <= int(m.sum()) <= max_area:
            out.append(m)
    return out


def _center_3d(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    p = points[mask]
    p = p[np.isfinite(p).all(axis=-1)]
    if len(p) < max(30, int(mask.sum() * 0.8)):
        raise NeedsTargetSelection("too few finite predicted 3D points inside selected RGB target")
    return np.median(p, axis=0)


def _width_3d(points: np.ndarray, mask: np.ndarray, right: np.ndarray) -> float:
    p = points[mask]
    p = p[np.isfinite(p).all(axis=-1)]
    projection = p @ right
    width = float(np.quantile(projection, 0.9) - np.quantile(projection, 0.1))
    if not np.isfinite(width) or width <= 1e-6:
        raise NeedsTargetSelection("predicted target width is degenerate")
    return width


def _choose_target(raw_masks: list[np.ndarray], points: np.ndarray, right: np.ndarray, sel: dict,
                   anchor_pixel: tuple[int, int] | None = None) -> tuple[np.ndarray, dict]:
    h, w = raw_masks[0].shape
    min_area = int(sel["min_area_pixels"])
    max_area = int(h * w * float(sel["max_area_fraction"]))
    if min_area < 1 or max_area < min_area:
        raise ValueError("invalid target component area limits")
    if sel["source"] in ("masks", "boxes"):
        candidates = [[m] if min_area <= int(m.sum()) <= max_area else [] for m in raw_masks]
    else:
        candidates = [_components(m, min_area, max_area) for m in raw_masks]
    if any(not frame for frame in candidates):
        raise NeedsTargetSelection("needs_target_selection: no quality-gated RGB vehicle in every frame")
    anchor = sel.get("anchor_normalized_xy")
    if anchor is not None:
        x, y = anchor_pixel if anchor_pixel is not None else (min(w - 1, int(anchor[0] * w)), min(h - 1, int(anchor[1] * h)))
        anchored = [m for m in candidates[0] if m[y, x]]
        if not anchored:
            raise NeedsTargetSelection(f"needs_target_selection: RGB anchor {anchor} does not hit a quality-gated vehicle component")
        first = anchored[0]
    else:
        first = sorted(candidates[0], key=lambda m: (-int(m.sum()), np.linalg.norm(np.argwhere(m).mean(axis=0) - [h / 2, w / 2])))[0]
    width = _width_3d(points[0], first, right)
    chosen = [first]
    associations = [{"frame": 0, "area": int(first.sum()), "iou": None, "distance_widths": None}]
    prev_center = _center_3d(points[0], first)
    for t in range(1, len(candidates)):
        if sel["source"] in ("masks", "boxes"):
            mask = candidates[t][0]
            center = _center_3d(points[t], mask)
            intersection = int(np.logical_and(mask, chosen[-1]).sum())
            union = int(np.logical_or(mask, chosen[-1]).sum())
            associations.append({"frame": t, "area": int(mask.sum()), "iou": intersection / union if union else 0.0,
                                 "distance_widths": float(np.linalg.norm(center - prev_center) / width), "manual": True})
            chosen.append(mask)
            prev_center = center
            continue
        ranked = []
        for m in candidates[t]:
            intersection = int(np.logical_and(m, chosen[-1]).sum())
            union = int(np.logical_or(m, chosen[-1]).sum())
            iou = intersection / union if union else 0.0
            try:
                center = _center_3d(points[t], m)
            except NeedsTargetSelection:
                continue
            distance_widths = float(np.linalg.norm(center - prev_center) / width)
            if iou >= float(sel["min_iou"]) and distance_widths <= float(sel["max_3d_distance_widths"]):
                ranked.append((distance_widths - iou, -iou, m, center, distance_widths, iou))
        if not ranked:
            raise NeedsTargetSelection(f"needs_target_selection: target association failed in frame {t}; provide RGB masks or boxes")
        ranked.sort(key=lambda x: (x[0], x[1]))
        _, _, mask, prev_center, distance_widths, iou = ranked[0]
        chosen.append(mask)
        associations.append({"frame": t, "area": int(mask.sum()), "iou": iou, "distance_widths": distance_widths})
    return np.stack(chosen), {"target_width_model_units": width, "associations": associations,
                              "first_frame_selection": "rgb_anchor_component" if anchor is not None else ("explicit_rgb_region" if sel["source"] in ("masks", "boxes") else "largest_rgb_semantic_component"),
                              "anchor_normalized_xy": anchor,
                              "roi_box_normalized_xyxy": sel.get("roi_box_normalized_xyxy"),
                              "selection_provenance": "manual_RGB_ROI_and_anchor_on_RGB_derived_semantic_components" if anchor is not None and "roi_box_normalized_xyxy" in sel else ("manual_RGB_anchor_on_RGB_derived_semantic_components" if anchor is not None else sel["source"]) }


def _expand_in_3d(target: np.ndarray, points: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return target
    from scipy.spatial import cKDTree

    finite = np.isfinite(points).all(axis=-1)
    cloud = points[target & finite]
    if len(cloud) == 0:
        return target
    tree = cKDTree(cloud.reshape(-1, 3))
    flat_points = points.reshape(-1, 3)
    valid = finite.reshape(-1)
    expanded = target.reshape(-1).copy()
    valid_index = np.flatnonzero(valid)
    for start in range(0, len(valid_index), 50000):
        idx = valid_index[start:start + 50000]
        near = tree.query(flat_points[idx], k=1, distance_upper_bound=radius, workers=1)[0] <= radius
        expanded[idx] |= near
    return expanded.reshape(target.shape)


def _alpha_t(t, t0, alpha, gamma0, torch):
    # Exact official mode-2 formula (inference.py 32-36), including default gamma1.
    sigma = torch.log(torch.tensor(0.1)).to(gamma0.device) / (gamma0 ** 2 + 1e-6)
    conf = torch.exp(sigma * (t0 - t) ** 2)
    return (alpha * conf).float()


def _rgb_u8(tensor) -> np.ndarray:
    return (tensor.detach().float().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)


def _save_sequence(folder: Path, name: str, frames: np.ndarray) -> None:
    import imageio.v2 as imageio
    from PIL import Image

    out = folder / name
    out.mkdir()
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(out / f"{i:03d}.png")
    imageio.mimwrite(folder / f"{name}.mp4", list(frames), fps=8, codec="libx264")


def _heatmap(values: np.ndarray) -> np.ndarray:
    v = np.clip(values, 0, 1)
    return np.stack([v, np.zeros_like(v), 1 - v], axis=-1).__mul__(255).round().astype(np.uint8)


def _save_visuals(out: Path, images, predictions: dict, selected: np.ndarray, outputs: dict) -> None:
    from PIL import Image

    raw = _rgb_u8(images[0].permute(0, 2, 3, 1))
    _save_sequence(out, "input_rgb", raw)
    _save_sequence(out, "predicted_dynamic", _heatmap(predictions["dynamic_conf"][0].squeeze(-1).sigmoid().detach().cpu().numpy()))
    if "semantic_logits" in predictions:
        logits = predictions["semantic_logits"][0].detach().cpu()
        if logits.shape[-1] <= 256:
            classes = logits.argmax(dim=-1).numpy()
            hues = np.stack([(classes * 37) % 256, (classes * 73) % 256, (classes * 131) % 256], axis=-1).astype(np.uint8)
            _save_sequence(out, "predicted_semantic_diagnostic", hues)
    overlays = []
    for frame, mask in zip(raw, selected):
        im = Image.fromarray(frame).convert("RGBA")
        layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
        layer.putalpha(Image.fromarray((mask * 105).astype(np.uint8)))
        tint = Image.new("RGBA", im.size, (255, 30, 30, 0))
        tint.putalpha(layer.getchannel("A"))
        im.alpha_composite(tint)
        overlays.append(np.asarray(im.convert("RGB")))
    _save_sequence(out, "selected_target_overlay", np.stack(overlays))
    for name, data in outputs.items():
        _save_sequence(out, name, _rgb_u8(data["rgb"]))
        _save_sequence(out, name + "_alpha", np.repeat(_rgb_u8(data["alpha"])[..., None], 3, axis=-1))
    hole = (outputs["noop"]["alpha"] - outputs["delete"]["alpha"]).detach().float().cpu().clamp(0, 1).numpy()
    _save_sequence(out, "delete_alpha_loss_hole", np.repeat((hole * 255).round().astype(np.uint8)[..., None], 3, axis=-1))
    np.save(out / "noop_rgb_float.npy", outputs["noop"]["rgb"].detach().float().cpu().numpy())


def main() -> int:
    args = _args()
    if args.sequence_length < 2:
        raise ValueError("official mode 2 timestamp normalization requires sequence-length >= 2")
    cfg = _config(args.edit_config)
    external, data_root = args.external.resolve(), args.data_root.resolve()
    if not (external / "dggt" / "models" / "vggt.py").is_file():
        raise ValueError(f"no official DGGT model at {external}")
    if not (external / "datasets" / "dataset.py").is_file():
        raise ValueError(f"no official Waymo dataset loader at {external}")
    images_paths = _image_files(data_root, args.scene, args.sequence_length)
    size = _target_size(images_paths[0])
    raw_masks, selector_paths = _label_maps(cfg["selection"], data_root, args.scene, images_paths, size)
    if any(not m.any() for m in raw_masks):
        raise NeedsTargetSelection("needs_target_selection: RGB selector empty in at least one frame")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    if not args.check_only and not args.official_trace and not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.official_trace and not args.official_trace.is_file():
        raise FileNotFoundError(args.official_trace)
    if args.check_only:
        print(json.dumps({"status": "cpu_contract_ok", "scene": args.scene, "frames": len(images_paths), "processed_size": size, "selection_source": cfg["selection"]["source"], "selector_paths": selector_paths}, ensure_ascii=False))
        return 0

    # Lazy official mode-2 only imports: no TAPIP3D, pointops, Difix, or GT camera/depth.
    sys.path.insert(0, str(external))
    import torch
    from datasets.dataset import WaymoOpenDataset
    from dggt.utils.geometry import unproject_depth_map_to_point_map
    from dggt.utils.gs import concat_list, get_split_gs
    from dggt.utils.pose_enc import pose_encoding_to_extri_intri
    from gsplat.rendering import rasterization

    if not torch.cuda.is_available():
        raise RuntimeError("DGGT + gsplat render needs CUDA; use --check-only for CPU contract")
    device = "cuda"
    dataset = WaymoOpenDataset(str(data_root), scene_names=[args.scene], sequence_length=args.sequence_length, start_idx=0, mode=2, views=1)
    sample = dataset[0]
    if [Path(p).resolve() for p in sample["image_paths"]] != [p.resolve() for p in images_paths]:
        raise RuntimeError("official loader RGB frame order differs from selector order")
    images = sample["images"].unsqueeze(0).to(device)
    sky_mask = sample["masks"].unsqueeze(0).to(device).permute(0, 1, 3, 4, 2)
    bg_mask = (sky_mask == 0).any(dim=-1)
    timestamps = torch.as_tensor(sample["timestamps"], device=device)
    if tuple(images.shape[-2:]) != size:
        raise RuntimeError("official RGB preprocessing differs from selector preprocessing")
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    if args.official_trace:
        trace = torch.load(args.official_trace, map_location=device, weights_only=False)
        required = ("rgb", "input_rgb", "extrinsic", "intrinsic", "point_map", "gs_map", "gs_conf", "dy_map", "bg_mask", "timestamps", "bg_render")
        missing = [key for key in required if key not in trace]
        if missing:
            raise ValueError(f"official trace missing {missing}; export mode-2 tensors including bg_render")
        load_seconds = time.perf_counter() - start
    else:
        from dggt.models.vggt import VGGT

        model = VGGT().to(device)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint, strict=True)
        model.eval()
        load_seconds = time.perf_counter() - start
    with torch.no_grad():
        predict_start = time.perf_counter()
        h, w = images.shape[-2:]
        if args.official_trace:
            trace_input = trace["input_rgb"]
            if trace_input.ndim == 4:
                trace_input = trace_input.unsqueeze(0)
            if trace_input.shape != images.shape or not torch.allclose(trace_input, images, atol=1e-7, rtol=0):
                raise RuntimeError("official trace input RGB differs from official loader input")
            extrinsic, intrinsic = trace["extrinsic"], trace["intrinsic"]
            point_map, gs_map, gs_conf = trace["point_map"], trace["gs_map"], trace["gs_conf"]
            dy_map, sky = trace["dy_map"], trace["bg_render"]
            if not torch.equal(trace["bg_mask"].bool(), bg_mask):
                raise RuntimeError("official trace sky/background mask differs from loader")
            if not torch.allclose(trace["timestamps"], timestamps, atol=1e-7, rtol=0):
                raise RuntimeError("official trace timestamps differ from loader")
            if sky.ndim == 5 and sky.shape[0] == 1:
                sky = sky[0]
            predictions = {"dynamic_conf": dy_map.unsqueeze(-1)}
            if "semantic_logits" in trace:
                predictions["semantic_logits"] = trace["semantic_logits"]
        else:
            predictions = model(images)  # Official FP32; no autocast to lower precision.
            extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions["pose_enc"], (h, w))
            extrinsic = extrinsics[0]
            bottom = torch.tensor([0., 0., 0., 1.], device=extrinsic.device).view(1, 1, 4).expand(extrinsic.shape[0], 1, 4)
            extrinsic = torch.cat([extrinsic, bottom], dim=1)
            intrinsic = intrinsics[0]
            depth_map = predictions["depth"][0]
            point_map = torch.from_numpy(unproject_depth_map_to_point_map(depth_map, extrinsics[0], intrinsics[0])[None, ...]).to(device).float()
            gs_map = predictions["gs_map"]
            gs_conf = predictions["gs_conf"]
            dy_map = predictions["dynamic_conf"].squeeze(-1)
            sky = model.sky_model(images, extrinsic, intrinsic)
            sky = (sky - sky.min()) / (sky.max() - sky.min() + 1e-8)
        torch.cuda.synchronize()
        predict_seconds = time.perf_counter() - predict_start

        points_cpu = point_map[0].detach().cpu().numpy()
        camera0_right = extrinsic[0, :3, :3].detach().float().cpu().numpy().T @ np.array([1., 0., 0.])
        camera0_right /= np.linalg.norm(camera0_right)
        anchor = cfg["selection"].get("anchor_normalized_xy")
        anchor_pixel = _rgb_anchor_pixel(images_paths[0], size, anchor) if anchor is not None else None
        selected, selection_meta = _choose_target(raw_masks, points_cpu, camera0_right, cfg["selection"], anchor_pixel)
        selection_meta["anchor_processed_xy"] = anchor_pixel
        width = selection_meta["target_width_model_units"]
        selected_expanded = _expand_in_3d(selected, points_cpu, width * float(cfg["selection"]["neighbor_radius_widths"]))
        selected_t = torch.from_numpy(selected_expanded).to(device)
        delta = torch.as_tensor(camera0_right * width * float(cfg["move"]["right_widths"]), device=device, dtype=point_map.dtype)

        # Official inference.py mode 2, lines 178-228: exact masks/sigmoids and
        # provenance retained before flattening. Edit mask spans all input frames.
        static_mask = bg_mask & (dy_map < 0.5)
        static_points = point_map[static_mask].reshape(-1, 3)
        gs_dynamic_list = dy_map[static_mask].sigmoid()
        static_rgbs, static_opacity, static_scales, static_rotations = get_split_gs(gs_map, static_mask)
        static_opacity = static_opacity * (1 - gs_dynamic_list)
        static_gs_conf = gs_conf[static_mask]
        static_indices = torch.nonzero(static_mask, as_tuple=False)
        gs_timestamps = timestamps[static_indices[:, 1]]
        selected_static = selected_t[static_indices[:, 1], static_indices[:, 2], static_indices[:, 3]]

        dynamic_gs = []
        selected_dynamic = []
        for i in range(dy_map.shape[1]):
            frame_bg = bg_mask[:, i]
            frame_points = point_map[:, i][frame_bg].reshape(-1, 3)
            frame_rgb, frame_opacity, frame_scales, frame_rotation = get_split_gs(gs_map[:, i], frame_bg)
            frame_opacity = frame_opacity * dy_map[:, i][frame_bg].sigmoid()
            dynamic_gs.append([frame_points, frame_rgb, frame_opacity, frame_scales, frame_rotation])
            selected_dynamic.append(selected_t[i][frame_bg[0]])
        if not bool(selected_static.any()) and not any(bool(x.any()) for x in selected_dynamic):
            raise NeedsTargetSelection("needs_target_selection: selected RGB vehicle has no non-sky Gaussian provenance")

        def render_branch(kind: str) -> dict:
            start_branch = time.perf_counter()
            renders, alphas = [], []
            for idx in range(dy_map.shape[1]):
                t0 = timestamps[idx]
                static_opacity_t = _alpha_t(gs_timestamps, t0, static_opacity, static_gs_conf, torch)
                static_gs = [static_points, static_rgbs, static_opacity_t, static_scales, static_rotations]
                dynamic = dynamic_gs[idx]
                if kind == "delete":
                    static_gs = [x[~selected_static] for x in static_gs]
                    dynamic = [x[~selected_dynamic[idx]] for x in dynamic]
                elif kind == "move":
                    static_gs = [x.clone() if j == 0 else x for j, x in enumerate(static_gs)]
                    dynamic = [x.clone() if j == 0 else x for j, x in enumerate(dynamic)]
                    static_gs[0][selected_static] += delta
                    dynamic[0][selected_dynamic[idx]] += delta
                world_points, rgbs, opacity, scales, rotation = concat_list(static_gs, dynamic)
                if kind == "insert_copy":
                    # Retain every original Gaussian, then append a translated
                    # same-scene instance copy with original attributes/time.
                    copy_static = [x[selected_static] for x in static_gs]
                    copy_dynamic = [x[selected_dynamic[idx]] for x in dynamic]
                    copied = concat_list(copy_static, copy_dynamic)
                    world_points = torch.cat([world_points, copied[0] + delta], dim=0)
                    rgbs = torch.cat([rgbs, copied[1]], dim=0)
                    opacity = torch.cat([opacity, copied[2]], dim=0)
                    scales = torch.cat([scales, copied[3]], dim=0)
                    rotation = torch.cat([rotation, copied[4]], dim=0)
                render, alpha, _ = rasterization(means=world_points, quats=rotation, scales=scales,
                                                  opacities=opacity, colors=rgbs,
                                                  viewmats=extrinsic[idx:idx + 1], Ks=intrinsic[idx:idx + 1],
                                                  width=w, height=h, render_mode="RGB+ED")
                renders.append(render)
                alphas.append(alpha)
            rgba_depth = torch.cat(renders, dim=0)
            alpha = torch.cat(alphas, dim=0)
            rgb = alpha * rgba_depth[..., :-1] + (1 - alpha) * sky
            torch.cuda.synchronize()
            return {"rgb": rgb, "alpha": alpha.squeeze(-1), "seconds": time.perf_counter() - start_branch}

        outputs = {name: render_branch(name) for name in ("noop", "delete", "move", "insert_copy")}
        official_comparison = None
        if args.official_trace:
            official_rgb = trace["rgb"]
            if official_rgb.ndim == 5 and official_rgb.shape[0] == 1:
                official_rgb = official_rgb[0]
            if official_rgb.ndim == 4 and official_rgb.shape[1] == 3:
                official_rgb = official_rgb.permute(0, 2, 3, 1)
            if official_rgb.shape != outputs["noop"]["rgb"].shape:
                raise RuntimeError("official RGB trace shape differs from no-op render")
            diff = (official_rgb - outputs["noop"]["rgb"]).abs()
            official_comparison = {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
                                   "same_8bit_quantization": bool(np.array_equal(_rgb_u8(official_rgb), _rgb_u8(outputs["noop"]["rgb"]))) }
            if official_comparison["max_abs"] > 1e-4:
                raise RuntimeError(f"no-op diverges from actual official mode-2 RGB: {official_comparison}")
        out = args.output_dir.resolve()
        out.mkdir(parents=True, exist_ok=False)
        _save_visuals(out, images, predictions, selected, outputs)
        if cfg.get("save_gaussians", False):
            torch.save({"static": [x.detach().cpu() for x in [static_points, static_rgbs, static_opacity, static_scales, static_rotations]],
                        "static_selected": selected_static.cpu(), "static_timestamps": gs_timestamps.cpu(), "static_conf": static_gs_conf.cpu(),
                        "dynamic": [[x.detach().cpu() for x in frame] for frame in dynamic_gs],
                        "dynamic_selected": [x.cpu() for x in selected_dynamic],
                        "extrinsic": extrinsic.cpu(), "intrinsic": intrinsic.cpu(), "sky": sky.cpu(), "delta": delta.cpu()}, out / "gaussian_state.pt")
        manifest = {
            "status": "completed", "method": "local_zero_training_gaussian_edit_not_official_cli",
            "official_mode": 2, "views": 1, "scene": args.scene, "frames": args.sequence_length,
            "official_trace": str(args.official_trace.resolve()) if args.official_trace else None,
            "official_noop_comparison": official_comparison,
            "predicted_semantic_exported": "semantic_logits" in predictions,
            "rgb_paths": [str(p) for p in images_paths], "selector_paths": selector_paths,
            "selection_config": cfg["selection"], "move_config": cfg["move"],
            "selection": {**selection_meta, "direct_pixels": [int(m.sum()) for m in selected],
                          "expanded_pixels": [int(m.sum()) for m in selected_expanded],
                          "delta_world_model_units": delta.detach().cpu().tolist()},
            "gaussian_counts": {"static_total": len(static_points), "static_selected": int(selected_static.sum()),
                                "dynamic_total_per_frame": [len(x[0]) for x in dynamic_gs],
                                "dynamic_selected_per_frame": [int(x.sum()) for x in selected_dynamic],
                                "insert_copy_added_per_frame": [int(selected_static.sum()) + int(x.sum()) for x in selected_dynamic]},
            "edit_roles": {"delete": "filter selected original Gaussians", "move": "translate selected original Gaussian means",
                           "insert_copy": "append translated copy of same-scene selected Gaussians; no cross-scene transfer"},
            "seconds": {"load": load_seconds, "predict_and_sky": predict_seconds,
                        **{name + "_render": outputs[name]["seconds"] for name in outputs}},
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "interface_source": {"dataset": "datasets/dataset.py WaymoOpenDataset mode=2", "model": "dggt/models/vggt.py VGGT",
                                 "prediction_geometry": "inference.py 157-175", "gaussian_assembly": "inference.py 178-228",
                                 "render_and_sky": "inference.py 258-293", "selection": "this script; RGB-derived masks/boxes + predicted 3D association"},
            "limits": [f"No-op was numerically compared with actual official mode-2 RGB (max_abs={official_comparison['max_abs']:.8g})." if official_comparison else "No official mode-2 tensor trace was supplied, so numerical parity was not checked.",
                       "Masks and boxes are edit controls only; predicted DGGT semantic logits are diagnostic and not used for target selection.",
                       "Delete exposes existing Gaussian/sky content; no inpainting or Difix is applied."]}
        (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"status": "completed", "output_dir": str(out), "gaussian_counts": manifest["gaussian_counts"], "seconds": manifest["seconds"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except NeedsTargetSelection as exc:
        print(json.dumps({"status": "needs_target_selection", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)

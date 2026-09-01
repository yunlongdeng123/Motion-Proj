"""Deterministic AV2 RGB, camera-depth, and video evidence for WorldSim V7 P3-B."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFont


_CUBOID_EDGES = (
    (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
)


def _local_to_ego(points: np.ndarray, diagnostics: Mapping[str, Any]) -> np.ndarray:
    rotation = np.asarray(diagnostics["query_actor_rotation_ego"], dtype=np.float32)
    center = np.asarray(diagnostics["query_actor_center_ego"], dtype=np.float32)
    return np.asarray(points, dtype=np.float32) @ rotation.T + center[None, :]


def _project(
    loader: Any,
    points_ego: np.ndarray,
    camera_name: str,
    camera_timestamp_ns: int,
    lidar_timestamp_ns: int,
    log_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points_ego) == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty(0, dtype=bool),
        )
    uv, points_camera, valid = loader.project_ego_to_img_motion_compensated(
        points_lidar_time=np.asarray(points_ego, dtype=np.float32),
        cam_name=camera_name,
        cam_timestamp_ns=camera_timestamp_ns,
        lidar_timestamp_ns=lidar_timestamp_ns,
        log_id=log_id,
    )
    if uv is None or points_camera is None or valid is None:
        return (
            np.empty((len(points_ego), 2), dtype=np.float32),
            np.empty((len(points_ego), 3), dtype=np.float32),
            np.zeros(len(points_ego), dtype=bool),
        )
    return (
        np.asarray(uv, dtype=np.float32),
        np.asarray(points_camera, dtype=np.float32),
        np.asarray(valid, dtype=bool),
    )


def select_camera_by_query_visibility(
    loader: Any,
    query_ego: np.ndarray,
    log_id: str,
    lidar_timestamp_ns: int,
    camera_names: list[str],
) -> dict[str, Any]:
    """Select a camera without decoding RGB or reading compiled-result quality."""
    candidates = []
    for rank, camera_name in enumerate(camera_names):
        image_path = loader.get_closest_img_fpath(
            log_id=log_id,
            cam_name=camera_name,
            lidar_timestamp_ns=lidar_timestamp_ns,
        )
        if image_path is None or not image_path.is_file():
            candidates.append(
                {
                    "camera_name": camera_name,
                    "camera_rank": rank,
                    "image_path": None,
                    "camera_timestamp_ns": None,
                    "visible_query_points": -1,
                }
            )
            continue
        camera_timestamp_ns = int(image_path.stem)
        _, _, valid = _project(
            loader,
            query_ego,
            camera_name,
            camera_timestamp_ns,
            lidar_timestamp_ns,
            log_id,
        )
        candidates.append(
            {
                "camera_name": camera_name,
                "camera_rank": rank,
                "image_path": image_path,
                "camera_timestamp_ns": camera_timestamp_ns,
                "visible_query_points": int(np.count_nonzero(valid)),
            }
        )
    selected = max(
        candidates,
        key=lambda row: (int(row["visible_query_points"]), -int(row["camera_rank"])),
    )
    if selected["image_path"] is None:
        raise RuntimeError(f"no synchronized camera image for {log_id}@{lidar_timestamp_ns}")
    return {**selected, "candidates": candidates}


def _crop_bounds(
    uv: np.ndarray,
    valid: np.ndarray,
    width: int,
    height: int,
    config: Mapping[str, Any],
) -> tuple[int, int, int, int]:
    visible = uv[valid]
    if len(visible) == 0:
        return 0, 0, width, height
    minimum_width = min(int(config["minimum_crop_width_px"]), width)
    minimum_height = min(int(config["minimum_crop_height_px"]), height)
    span = np.maximum(np.ptp(visible, axis=0), 1.0)
    crop_width = min(
        max(float(span[0]) * (1.0 + 2.0 * float(config["crop_padding_fraction"])), minimum_width),
        width,
    )
    crop_height = min(
        max(float(span[1]) * (1.0 + 2.0 * float(config["crop_padding_fraction"])), minimum_height),
        height,
    )
    center = np.mean(visible, axis=0)
    x0 = int(np.clip(round(center[0] - crop_width / 2.0), 0, width - crop_width))
    y0 = int(np.clip(round(center[1] - crop_height / 2.0), 0, height - crop_height))
    return x0, y0, int(round(x0 + crop_width)), int(round(y0 + crop_height))


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _draw_points(
    image: Image.Image,
    uv: np.ndarray,
    valid: np.ndarray,
    crop: tuple[int, int, int, int],
    color: tuple[int, int, int],
    radius: int,
    cross: bool = False,
) -> None:
    draw = ImageDraw.Draw(image)
    x0, y0, _, _ = crop
    for u, v in uv[valid]:
        x, y = float(u) - x0, float(v) - y0
        if cross:
            draw.line((x - radius, y - radius, x + radius, y + radius), fill=color, width=2)
            draw.line((x - radius, y + radius, x + radius, y - radius), fill=color, width=2)
        else:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def _cuboid_corners(size_lwh_m: np.ndarray) -> np.ndarray:
    half = np.asarray(size_lwh_m, dtype=np.float32) * 0.5
    return np.asarray(
        [[sx * half[0], sy * half[1], sz * half[2]]
         for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        dtype=np.float32,
    )


def _draw_cuboid(
    image: Image.Image,
    uv: np.ndarray,
    points_camera: np.ndarray,
    crop: tuple[int, int, int, int],
    color: tuple[int, int, int],
) -> None:
    draw = ImageDraw.Draw(image)
    x0, y0, _, _ = crop
    for first, second in _CUBOID_EDGES:
        if points_camera[first, 2] <= 0.0 or points_camera[second, 2] <= 0.0:
            continue
        draw.line(
            (
                float(uv[first, 0]) - x0,
                float(uv[first, 1]) - y0,
                float(uv[second, 0]) - x0,
                float(uv[second, 1]) - y0,
            ),
            fill=color,
            width=3,
        )


def _titled(panel: Image.Image, title: str, output_size: tuple[int, int]) -> Image.Image:
    width, height = output_size
    resized = panel.resize((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height + 38), (18, 18, 18))
    canvas.paste(resized, (0, 38))
    ImageDraw.Draw(canvas).text((12, 8), title, fill=(242, 242, 242), font=_font(18))
    return canvas


def _sparse_depth_panel(
    lidar_uv: np.ndarray,
    lidar_camera: np.ndarray,
    lidar_valid: np.ndarray,
    query_uv: np.ndarray,
    query_valid: np.ndarray,
    crop: tuple[int, int, int, int],
    maximum_depth_m: float,
) -> tuple[Image.Image, int]:
    x0, y0, x1, y1 = crop
    width, height = x1 - x0, y1 - y0
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    valid = lidar_valid.copy()
    valid &= lidar_uv[:, 0] >= x0
    valid &= lidar_uv[:, 0] < x1
    valid &= lidar_uv[:, 1] >= y0
    valid &= lidar_uv[:, 1] < y1
    indices = np.flatnonzero(valid)
    if len(indices):
        u = np.rint(lidar_uv[indices, 0]).astype(np.int32) - x0
        v = np.rint(lidar_uv[indices, 1]).astype(np.int32) - y0
        z = np.clip(lidar_camera[indices, 2], 0.0, maximum_depth_m)
        scalar = np.rint((1.0 - z / maximum_depth_m) * 255.0).astype(np.uint8)
        colors = cv2.applyColorMap(scalar[:, None], cv2.COLORMAP_TURBO)[:, 0, ::-1]
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            uu = np.clip(u + dx, 0, width - 1)
            vv = np.clip(v + dy, 0, height - 1)
            canvas[vv, uu] = colors
    image = Image.fromarray(canvas)
    _draw_points(image, query_uv, query_valid, crop, (255, 255, 255), radius=3)
    return image, len(indices)


def render_camera_evidence(
    loader: Any,
    diagnostics: Mapping[str, Any],
    evidence: Mapping[str, Any],
    log_dir: Path,
    camera_config: Mapping[str, Any],
    output_path: Path,
) -> tuple[dict[str, Any], dict[str, Image.Image]]:
    log_id = log_dir.name
    lidar_timestamp_ns = int(diagnostics["query_timestamp_ns"])
    query_ego = _local_to_ego(diagnostics["query"], diagnostics)
    selected = select_camera_by_query_visibility(
        loader,
        query_ego,
        log_id,
        lidar_timestamp_ns,
        [str(value) for value in camera_config["candidate_cameras"]],
    )
    camera_name = str(selected["camera_name"])
    camera_timestamp_ns = int(selected["camera_timestamp_ns"])
    image_path = Path(selected["image_path"])
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    layers_local = {
        "query": diagnostics["query"],
        "ghost": diagnostics["ghost"],
        "duplicate": diagnostics["duplicate"],
        "flicker": diagnostics["flicker"],
        "kept": diagnostics["kept"],
        "projected": diagnostics["projected"],
        "completed": diagnostics["completed"],
        "unknown": diagnostics["unknown_query"],
    }
    projected_layers: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, points_local in layers_local.items():
        points_ego = _local_to_ego(points_local, diagnostics)
        projected_layers[name] = _project(
            loader,
            points_ego,
            camera_name,
            camera_timestamp_ns,
            lidar_timestamp_ns,
            log_id,
        )
    query_uv, query_camera, query_valid = projected_layers["query"]
    crop = _crop_bounds(query_uv, query_valid, width, height, camera_config)

    cuboid_ego = _local_to_ego(
        _cuboid_corners(np.asarray(diagnostics["track"].size_lwh_m)), diagnostics
    )
    cuboid_uv, cuboid_camera, _ = _project(
        loader,
        cuboid_ego,
        camera_name,
        camera_timestamp_ns,
        lidar_timestamp_ns,
        log_id,
    )
    base_crop = image.crop(crop)
    dim_crop = ImageEnhance.Brightness(base_crop).enhance(0.62)

    context = base_crop.copy()
    _draw_cuboid(context, cuboid_uv, cuboid_camera, crop, (255, 255, 255))
    _draw_points(context, query_uv, query_valid, crop, (0, 238, 255), radius=3)

    before = dim_crop.copy()
    _draw_cuboid(before, cuboid_uv, cuboid_camera, crop, (255, 255, 255))
    _draw_points(before, query_uv, query_valid, crop, (220, 220, 220), radius=2)
    for name, color in (
        ("ghost", (255, 45, 45)),
        ("duplicate", (255, 151, 38)),
        ("flicker", (191, 105, 255)),
    ):
        uv, _, valid = projected_layers[name]
        _draw_points(before, uv, valid, crop, color, radius=4)

    after = dim_crop.copy()
    _draw_cuboid(after, cuboid_uv, cuboid_camera, crop, (255, 255, 255))
    for name, color, radius, cross in (
        ("kept", (190, 190, 190), 2, False),
        ("projected", (0, 218, 255), 4, False),
        ("completed", (79, 220, 100), 4, False),
        ("unknown", (255, 205, 55), 4, True),
    ):
        uv, _, valid = projected_layers[name]
        _draw_points(after, uv, valid, crop, color, radius=radius, cross=cross)

    lidar_path = log_dir / "sensors" / "lidar" / f"{lidar_timestamp_ns}.feather"
    lidar_frame = pd.read_feather(lidar_path, columns=["x", "y", "z"])
    lidar_ego = lidar_frame[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=True)
    lidar_uv, lidar_camera, lidar_valid = _project(
        loader,
        lidar_ego,
        camera_name,
        camera_timestamp_ns,
        lidar_timestamp_ns,
        log_id,
    )
    depth, depth_points = _sparse_depth_panel(
        lidar_uv,
        lidar_camera,
        lidar_valid,
        query_uv,
        query_valid,
        crop,
        float(camera_config["maximum_depth_m"]),
    )
    _draw_cuboid(depth, cuboid_uv, cuboid_camera, crop, (255, 255, 255))

    panel_size = (
        int(camera_config["panel_width_px"]),
        int(camera_config["panel_height_px"]),
    )
    titled = [
        _titled(context, "RGB + observed AV2 query returns", panel_size),
        _titled(before, "BEFORE: paired ghost / duplicate / flicker", panel_size),
        _titled(after, "AFTER: KEEP / PROJECT / COMPLETE / UNKNOWN", panel_size),
        _titled(depth, "Motion-compensated sparse camera depth", panel_size),
    ]
    footer_height = 58
    canvas = Image.new(
        "RGB",
        (sum(panel.width for panel in titled), titled[0].height + footer_height),
        (18, 18, 18),
    )
    offset = 0
    for panel in titled:
        canvas.paste(panel, (offset, 0))
        offset += panel.width
    footer = (
        f"{log_id[:8]} | {str(evidence['track_id'])[:8]} | {camera_name} | "
        f"camera selected by query visibility only | hazard={int(bool(evidence['hazardous']))} kept | "
        f"FREE {float(evidence['free_space_violation_rate_before']):.2f}→"
        f"{float(evidence['free_space_violation_rate_after']):.2f} | "
        f"CD {float(evidence['symmetric_chamfer_before_m']):.3f}→"
        f"{float(evidence['symmetric_chamfer_after_m']):.3f} m"
    )
    ImageDraw.Draw(canvas).text(
        (14, titled[0].height + 15), footer, fill=(235, 235, 235), font=_font(17)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

    metadata = {
        "camera_name": camera_name,
        "camera_rank": int(selected["camera_rank"]),
        "camera_timestamp_ns": camera_timestamp_ns,
        "lidar_timestamp_ns": lidar_timestamp_ns,
        "visible_query_points": int(selected["visible_query_points"]),
        "query_point_count": len(query_ego),
        "query_visibility_fraction": int(selected["visible_query_points"])
        / max(len(query_ego), 1),
        "camera_visibility_counts": {
            str(row["camera_name"]): max(int(row["visible_query_points"]), 0)
            for row in selected["candidates"]
        },
        "crop_xyxy": list(crop),
        "depth_points_in_crop": int(depth_points),
        "synthetic_artifact_overlay": True,
        "photorealistic_reconstruction": False,
        "selection_read_rgb_appearance": False,
        "selection_read_method_quality": False,
    }
    return metadata, {"context": context, "before": before, "after": after}


def write_evidence_video(
    panels: Mapping[str, Image.Image],
    output_path: Path,
    evidence: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    width = int(config["panel_width_px"])
    height = int(config["panel_height_px"])
    fps = int(config["fps"])
    frames_per_phase = int(config["frames_per_phase"])
    context = np.asarray(panels["context"].resize((width, height), Image.Resampling.LANCZOS))
    before = np.asarray(panels["before"].resize((width, height), Image.Resampling.LANCZOS))
    after = np.asarray(panels["after"].resize((width, height), Image.Resampling.LANCZOS))
    frame_size = (width * 2, height + 62)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {output_path}")
    try:
        for phase in range(4):
            for frame_index in range(frames_per_phase):
                alpha = frame_index / max(frames_per_phase - 1, 1)
                if phase == 0:
                    left, right = context, context
                    state = "Observed AV2 frame"
                elif phase == 1:
                    left = cv2.addWeighted(context, 1.0 - alpha, before, alpha, 0.0)
                    right = context
                    state = "Inject paired ray/depth artifacts"
                elif phase == 2:
                    pulse = 0.30 if frame_index % 2 == 0 else 1.0
                    left = cv2.addWeighted(context, 1.0 - pulse, before, pulse, 0.0)
                    right = cv2.addWeighted(context, 1.0 - alpha, after, alpha, 0.0)
                    state = "Flicker exposed; compiler actions applied"
                else:
                    left, right = before, after
                    state = "Ray-certified result; Actor and hazard state retained"
                frame = np.zeros((height + 62, width * 2, 3), dtype=np.uint8)
                frame[:height, :width] = left[:, :, ::-1]
                frame[:height, width:] = right[:, :, ::-1]
                cv2.putText(frame, "BEFORE", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2)
                cv2.putText(frame, "AFTER", (width + 14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2)
                cv2.putText(frame, state, (14, height + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1)
                claim = (
                    f"paired synthetic artifacts | hazard={int(bool(evidence['hazardous']))} kept | "
                    "evidence overlay, not photorealistic reconstruction"
                )
                cv2.putText(frame, claim, (14, height + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 210, 225), 1)
                writer.write(frame)
    finally:
        writer.release()

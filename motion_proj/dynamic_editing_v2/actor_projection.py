"""nuScenes global box 到相机像素的显式变换与边界裁剪。"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def wlh_to_lwh(size_wlh: list[float]) -> list[float]:
    if len(size_wlh) != 3 or any(not math.isfinite(float(x)) or float(x) <= 0 for x in size_wlh):
        raise ValueError(f"无效 wlh: {size_wlh}")
    w, length, height = [float(x) for x in size_wlh]
    return [length, w, height]


def lwh_to_wlh(size_lwh: list[float]) -> list[float]:
    if len(size_lwh) != 3 or any(not math.isfinite(float(x)) or float(x) <= 0 for x in size_lwh):
        raise ValueError(f"无效 lwh: {size_lwh}")
    length, w, height = [float(x) for x in size_lwh]
    return [w, length, height]


def quaternion_matrix(quaternion_wxyz: list[float]) -> np.ndarray:
    if len(quaternion_wxyz) != 4:
        raise ValueError("四元数必须为 wxyz")
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    if not np.all(np.isfinite(q)):
        raise ValueError("四元数非 finite")
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        raise ValueError("四元数范数为零")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_matrix(translation: list[float], rotation_wxyz: list[float]) -> np.ndarray:
    vector = np.asarray(translation, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("平移向量无效")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_matrix(rotation_wxyz)
    matrix[:3, 3] = vector
    return matrix


def box_corners_global(annotation: dict[str, Any]) -> np.ndarray:
    length, width, height = wlh_to_lwh(annotation["size_wlh"])
    local = np.asarray(
        [
            [length / 2, width / 2, height / 2],
            [length / 2, -width / 2, height / 2],
            [-length / 2, -width / 2, height / 2],
            [-length / 2, width / 2, height / 2],
            [length / 2, width / 2, -height / 2],
            [length / 2, -width / 2, -height / 2],
            [-length / 2, -width / 2, -height / 2],
            [-length / 2, width / 2, -height / 2],
        ],
        dtype=np.float64,
    )
    rotation = quaternion_matrix(annotation["rotation_quaternion"])
    translation = np.asarray(annotation["translation_global"], dtype=np.float64)
    return (rotation @ local.T).T + translation


def convex_hull(points: list[list[float]]) -> list[list[float]]:
    unique = sorted({(float(x), float(y)) for x, y in points})
    if len(unique) <= 1:
        return [list(point) for point in unique]

    def cross(origin, a, b):
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return [list(point) for point in lower[:-1] + upper[:-1]]


def clip_polygon_to_image(polygon: list[list[float]], width: int, height: int) -> list[list[float]]:
    def clip(points, inside, intersect):
        if not points:
            return []
        output = []
        previous = points[-1]
        previous_inside = inside(previous)
        for current in points:
            current_inside = inside(current)
            if current_inside != previous_inside:
                output.append(intersect(previous, current))
            if current_inside:
                output.append(current)
            previous, previous_inside = current, current_inside
        return output

    xmax, ymax = float(width - 1), float(height - 1)
    result = [list(map(float, point)) for point in polygon]
    boundaries = [
        (lambda p: p[0] >= 0.0, lambda a, b: [0.0, a[1] + (b[1] - a[1]) * (0.0 - a[0]) / (b[0] - a[0])]),
        (lambda p: p[0] <= xmax, lambda a, b: [xmax, a[1] + (b[1] - a[1]) * (xmax - a[0]) / (b[0] - a[0])]),
        (lambda p: p[1] >= 0.0, lambda a, b: [a[0] + (b[0] - a[0]) * (0.0 - a[1]) / (b[1] - a[1]), 0.0]),
        (lambda p: p[1] <= ymax, lambda a, b: [a[0] + (b[0] - a[0]) * (ymax - a[1]) / (b[1] - a[1]), ymax]),
    ]
    for inside, intersect in boundaries:
        result = clip(result, inside, intersect)
    return result


def polygon_area(polygon: list[list[float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    return abs(
        sum(
            polygon[i][0] * polygon[(i + 1) % len(polygon)][1]
            - polygon[(i + 1) % len(polygon)][0] * polygon[i][1]
            for i in range(len(polygon))
        )
    ) / 2.0


def project_box(
    annotation: dict[str, Any],
    calibrated_sensor: dict[str, Any],
    ego_pose: dict[str, Any],
    image_width: int,
    image_height: int,
    near_plane: float = 0.1,
) -> dict[str, Any]:
    intrinsic = np.asarray(calibrated_sensor["camera_intrinsic"], dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.all(np.isfinite(intrinsic)):
        raise ValueError("相机内参无效")
    t_global_ego = transform_matrix(ego_pose["translation"], ego_pose["rotation"])
    t_ego_camera = transform_matrix(calibrated_sensor["translation"], calibrated_sensor["rotation"])
    t_camera_global = np.linalg.inv(t_global_ego @ t_ego_camera)
    corners_global = box_corners_global(annotation)
    homogeneous = np.concatenate([corners_global, np.ones((8, 1))], axis=1)
    corners_camera = (t_camera_global @ homogeneous.T).T[:, :3]
    center_global = np.asarray([*annotation["translation_global"], 1.0], dtype=np.float64)
    center_camera = (t_camera_global @ center_global)[:3]

    clipped_3d = []
    for first, second in BOX_EDGES:
        a, b = corners_camera[first], corners_camera[second]
        if a[2] >= near_plane:
            clipped_3d.append(a)
        if b[2] >= near_plane:
            clipped_3d.append(b)
        if (a[2] >= near_plane) != (b[2] >= near_plane):
            alpha = (near_plane - a[2]) / (b[2] - a[2])
            clipped_3d.append(a + alpha * (b - a))
    if not clipped_3d:
        return {
            "valid": False,
            "invalid_reason": "box_behind_near_plane",
            "T_camera_global": t_camera_global.tolist(),
            "box_corners_camera": corners_camera.tolist(),
            "polygon_before_clip": [],
            "polygon_after_clip": [],
            "visible_area_px": 0.0,
            "center_camera": center_camera.tolist(),
            "center_pixel": None,
            "center_inside_image": False,
            "image_size_wh": [image_width, image_height],
            "near_plane": near_plane,
        }
    points = np.asarray(clipped_3d)
    projected = (intrinsic @ points.T).T
    pixels = projected[:, :2] / projected[:, 2:3]
    polygon_before = convex_hull(pixels.tolist())
    polygon_after = clip_polygon_to_image(polygon_before, image_width, image_height)
    visible_area = polygon_area(polygon_after)
    if center_camera[2] > near_plane:
        center_h = intrinsic @ center_camera
        center_pixel = (center_h[:2] / center_h[2]).tolist()
    else:
        center_pixel = None
    center_inside = bool(
        center_pixel is not None
        and 0 <= center_pixel[0] < image_width
        and 0 <= center_pixel[1] < image_height
    )
    valid = len(polygon_after) >= 3 and visible_area > 0
    return {
        "valid": valid,
        "invalid_reason": None if valid else "projected_polygon_outside_image",
        "T_camera_global": t_camera_global.tolist(),
        "box_corners_camera": corners_camera.tolist(),
        "polygon_before_clip": polygon_before,
        "polygon_after_clip": polygon_after,
        "visible_area_px": visible_area,
        "center_camera": center_camera.tolist(),
        "center_pixel": center_pixel,
        "center_inside_image": center_inside,
        "image_size_wh": [image_width, image_height],
        "near_plane": near_plane,
    }

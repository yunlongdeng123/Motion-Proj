"""检查sim角点K、原生投影与mask所属cell；只使用合成几何，不读取特权数据。"""
import argparse
import ast
import json
from pathlib import Path
import re
from types import SimpleNamespace
from tempfile import TemporaryDirectory

import numpy as np

from adapt_scene import actor_indices


def camera_points(uv, k, depth=2.0):
    uv = np.asarray(uv, dtype=np.float64)
    return np.column_stack(((uv[:, 0] - k[0, 2]) / k[0, 0] * depth,
                            (uv[:, 1] - k[1, 2]) / k[1, 1] * depth,
                            np.full(len(uv), depth)))


def native_projection_function(source):
    # 执行真实原生投影函数体，numpy代替torch.zeros；不导入Camera或创建CUDA context。
    tree = ast.parse((source / "utils/graphics_utils.py").read_text(encoding="utf-8-sig"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "getProjectionMatrixCenterShift")
    namespace = {"torch": SimpleNamespace(zeros=lambda *shape: np.zeros(shape))}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "native_getProjectionMatrixCenterShift", "exec"), namespace)
    return namespace[function.name]


def native_calibration_function(source):
    tree = ast.parse((source / "scene/waymo_calibration.py").read_text(encoding="utf-8-sig"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "read_named_camera_calibration")
    namespace = {"np": np, "Path": Path}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "native_read_named_camera_calibration", "exec"), namespace)
    return namespace[function.name]


def native_ndc_to_pixel(auxiliary):
    expression = "((v + 1.0) * S - 1.0) * 0.5"
    if auxiliary is not None:
        text = auxiliary.read_text(encoding="utf-8-sig")
        match = re.search(r"float\s+ndc2Pix\s*\([^)]*\)\s*\{\s*return\s+([^;]+);", text)
        if not match:
            raise ValueError("未找到真实rasterizer ndc2Pix函数")
        expression = match.group(1)
    parsed = ast.parse(expression, mode="eval")
    allowed = (ast.Expression, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Constant, ast.Name, ast.Load)
    if any(not isinstance(node, allowed) or isinstance(node, ast.Name) and node.id not in {"v", "S"} for node in ast.walk(parsed)):
        raise ValueError("ndc2Pix表达式超出简单算术范围")
    code = compile(parsed, "native_ndc2Pix", "eval")
    return lambda v, size: eval(code, {"__builtins__": {}}, {"v": v, "S": size})


def check(native_source, auxiliary=None):
    k = np.asarray([[8.0, 0.0, 2.5], [0.0, 8.0, 2.0], [0.0, 0.0, 1.0]])
    original_k = k.copy()
    height, width = 4, 5
    yy, xx = np.meshgrid(np.arange(height) + .5, np.arange(width) + .5, indexing="ij")
    uv = np.column_stack((xx.ravel(), yy.ravel()))
    points = camera_points(uv, k)
    body = np.zeros((height, width), dtype=bool)
    body[1, 2] = True
    selected, projected, before_component = actor_indices(points, points, k, body)
    assert selected.tolist() == [1 * width + 2], (selected, projected, before_component)
    # 旧round查找会把另一个像素的中心射线当作该mask像素，不只是丢边界点。
    legacy = np.floor(uv + .5).astype(int)
    legacy_hit = (legacy[:, 0] == 2) & (legacy[:, 1] == 1)
    assert np.flatnonzero(legacy_hit).tolist() != selected.tolist()
    all_body = np.ones((height, width), dtype=bool)
    selected_all, _, _ = actor_indices(points, points, k, all_body)
    assert np.array_equal(selected_all, np.arange(height * width)), "右/下边界中心射线被错误丢弃"
    assert np.array_equal(k, original_k), "修复不得改动K"

    # 明确检查[0,W)x[0,H)所属cell，负坐标不得因round被拉进图内。
    boundary_uv = [[-.1, .5], [.1, .5], [4.9, .5], [5., .5], [.5, -.1], [.5, .1], [.5, 3.9], [.5, 4.]]
    boundary_k = np.asarray([[128., 0., 2.5], [0., 128., 2.], [0., 0., 1.]])
    boundary_points = camera_points(boundary_uv, boundary_k)
    selected_boundary, _, _ = actor_indices(boundary_points, boundary_points, boundary_k, all_body)
    assert selected_boundary.tolist() == [1, 2, 5, 6], selected_boundary

    with TemporaryDirectory(prefix="v43_pixel_fixture_") as temporary:
        calibration = Path(temporary) / "calib.txt"
        projection_record = np.column_stack((k, np.zeros(3)))
        transform_record = np.eye(4)[:3]
        calibration.write_text("P0: " + " ".join(map(str, projection_record.ravel())) + "\n"
                               + "Tr_velo_to_cam_0: " + " ".join(map(str, transform_record.ravel())) + "\n", encoding="ascii")
        native_k, native_transform = native_calibration_function(native_source)(calibration, [0])
        assert np.array_equal(native_k[0], k), "原生calib读取改变了K"
        assert np.array_equal(native_transform[0], np.eye(4))

    # 真实原生calib读取不偏移K；真实投影函数+CUDA ndc2Pix应将sim中心射线映射到整数索引。
    projection = native_projection_function(native_source)(.01, 1000., k[0, 2], k[1, 2], k[0, 0], k[1, 1], width, height)
    clip = np.column_stack((points, np.ones(len(points)))) @ projection.T
    ndc = clip[:, :2] / clip[:, 3:4]
    to_pixel = native_ndc_to_pixel(auxiliary)
    raster_xy = np.column_stack((to_pixel(ndc[:, 0], width), to_pixel(ndc[:, 1], height)))
    expected_indices = uv - .5
    error = float(np.max(np.abs(raster_xy - expected_indices)))
    assert error < 1e-12, error
    shifted_k_projection = native_projection_function(native_source)(.01, 1000., k[0, 2] - .5, k[1, 2] - .5, k[0, 0], k[1, 1], width, height)
    shifted_clip = np.column_stack((points, np.ones(len(points)))) @ shifted_k_projection.T
    shifted_ndc = shifted_clip[:, :2] / shifted_clip[:, 3:4]
    shifted_xy = np.column_stack((to_pixel(shifted_ndc[:, 0], width), to_pixel(shifted_ndc[:, 1], height)))
    assert np.allclose(shifted_xy - expected_indices, -.5), "负对照应显示误减K主点产生额外半像素偏移"
    return {"status": "PASS_PIXEL_COORDINATE_FIXTURE", "mask_center_identity": "passed",
            "all_pixel_centers_and_boundaries": "passed", "K_unchanged": True,
            "native_calibration_K_passthrough": "passed",
            "native_projection_max_error_px": error, "wrong_K_minus_half_pixel_negative_control": "passed",
            "native_source": str(native_source), "rasterizer_auxiliary": None if auxiliary is None else str(auxiliary),
            "rasterizer_formula_origin": "declared native formula" if auxiliary is None else "actual auxiliary.h function expression",
            "scope": "synthetic geometry fixture; no privileged input, training, CUDA context, or experiment"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-source", type=Path, default=Path(__file__).resolve().parents[2] / "v1")
    parser.add_argument("--rasterizer-auxiliary", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.native_source, args.rasterizer_auxiliary), ensure_ascii=False))


if __name__ == "__main__":
    main()

"""场景无关的连贯高斯路块运输；全部几何单位为世界米。

sample_target(uv[N,2]) -> dict(position[N,3], jacobian[N,3,2],
    normal[N,3] optional, valid[N] optional, normal_jacobian[N,3,2] optional).
normal_jacobian仅在显式携带源中心normal residual时必需。

中心默认落到phi上；协方差使用保留法线厚度的局部normal extension。
单归属作用于高斯中心所在的世界网格，保留获胜块的完整行，不混色。
网格归属和3sigma替换范围不构成渲染器的逐像素保护保证。
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

_EPS = 1e-12


def _array(value, shape, name, *, dtype=np.float64):
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name}须为有限数组，形状{shape}，实际{result.shape}")
    return result


def _unit(value, name):
    length = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(length <= _EPS):
        raise ValueError(name + "含零向量")
    return value / length


def quaternion_wxyz_to_matrix(quaternion):
    """输入/输出保留前导维度；不改变调用者的四元数数组。"""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape[-1:] != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("world quaternion须为有限wxyz数组")
    shape = quaternion.shape[:-1]
    normalized = _unit(quaternion.reshape(-1, 4), "quaternion")
    return Rotation.from_quat(normalized[:, [1, 2, 3, 0]]).as_matrix().reshape(shape + (3, 3))


def matrix_to_quaternion_wxyz(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation matrix须为有限3x3数组")
    shape = matrix.shape[:-2]
    flat = matrix.reshape(-1, 3, 3)
    if not np.allclose(np.swapaxes(flat, 1, 2) @ flat, np.eye(3), atol=1e-9, rtol=1e-9) or np.any(np.linalg.det(flat) < 0):
        raise ValueError("rotation matrix须为右手正交阵")
    quaternion = Rotation.from_matrix(flat).as_quat()[:, [3, 0, 1, 2]]
    # q与-q同义，使用首个非零分量为正的确定性表示。
    for row in quaternion:
        first = np.flatnonzero(np.abs(row) > 1e-14)
        if len(first) and row[first[0]] < 0:
            row *= -1
    return quaternion.reshape(shape + (4,))


def covariance_from_sigma_quaternion(sigma_m, rotation_world_wxyz):
    sigma = np.asarray(sigma_m, dtype=np.float64)
    if sigma.shape[-1:] != (3,) or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("sigma_m须为正有限三轴标准差")
    rotation = quaternion_wxyz_to_matrix(rotation_world_wxyz)
    if rotation.shape[:-2] != sigma.shape[:-1]:
        raise ValueError("sigma与quaternion前导维度不一致")
    return (rotation * sigma[..., None, :] ** 2) @ np.swapaxes(rotation, -1, -2)


def normal_preserving_decomposition(covariance_world, normals_world, *, rtol=1e-8, atol=1e-14):
    """切平面2x2特征分解，第三轴严格保留给指定normal。

    若协方差有normal/tangent耦合则拒绝，避免静默丢弃完整协方差。
    切向sigma按大到小排序，输出wxyz与可重建的旋转矩阵。
    """
    covariance = np.asarray(covariance_world, dtype=np.float64)
    if covariance.ndim != 3 or covariance.shape[1:] != (3, 3) or not np.isfinite(covariance).all():
        raise ValueError("covariance_world须为Nx3x3")
    n = len(covariance)
    normals = _unit(_array(normals_world, (n, 3), "normals_world"), "normals_world")
    if not np.allclose(covariance, covariance.transpose(0, 2, 1), rtol=rtol, atol=atol):
        raise ValueError("协方差不对称")
    eigenvalues = np.linalg.eigvalsh(covariance)
    if np.any(eigenvalues <= 0):
        raise ValueError("协方差须正定")
    normal_variance = np.einsum("ni,nij,nj->n", normals, covariance, normals)
    coupling = (covariance @ normals[..., None])[..., 0] - normal_variance[:, None] * normals
    limit = atol + rtol * np.linalg.norm(covariance, axis=(1, 2))
    if np.any(np.linalg.norm(coupling, axis=1) > limit):
        raise ValueError("完整协方差含normal/tangent耦合，不能保normal降为切平面分解")
    reference = np.eye(3)[np.argmin(np.abs(normals), axis=1)]
    tangent0 = _unit(np.cross(reference, normals), "tangent")
    tangent1 = np.cross(normals, tangent0)
    tangent = np.stack((tangent0, tangent1), axis=2)
    local = tangent.transpose(0, 2, 1) @ covariance @ tangent
    values, vectors = np.linalg.eigh(local)
    values, vectors = values[:, ::-1], vectors[:, :, ::-1]
    axes = tangent @ vectors
    handedness = np.einsum("ni,ni->n", np.cross(axes[:, :, 0], axes[:, :, 1]), normals)
    axes[:, :, 1] *= np.where(handedness < 0, -1., 1.)[:, None]
    rotation = np.concatenate((axes, normals[:, :, None]), axis=2)
    sigma = np.sqrt(np.concatenate((values, normal_variance[:, None]), axis=1))
    quaternion = matrix_to_quaternion_wxyz(rotation)
    reconstructed = covariance_from_sigma_quaternion(sigma, quaternion)
    if not np.allclose(reconstructed, covariance, rtol=5 * rtol, atol=5 * atol):
        raise ValueError("normal-preserving协方差重建未通过")
    return {"gaussian_sigma_m": sigma, "gaussian_rotation_world_wxyz": quaternion,
            "rotation_matrix_world": rotation,
            "reconstruction_max_abs": np.max(np.abs(reconstructed - covariance), axis=(1, 2))}


def _sample_target(callback, uv):
    sample = callback(uv.copy())
    if not isinstance(sample, dict):
        raise ValueError("sample_target须返回dict(position,jacobian,...) ")
    n = len(uv)
    position = np.asarray(sample["position"], dtype=np.float64)
    jacobian = np.asarray(sample["jacobian"], dtype=np.float64)
    if position.shape != (n, 3) or jacobian.shape != (n, 3, 2):
        raise ValueError("target position/jacobian形状错误")
    valid = np.asarray(sample.get("valid", np.ones(n, bool)), dtype=bool)
    if valid.shape != (n,):
        raise ValueError("target valid须为N布尔数组")
    valid = valid & np.isfinite(position).all(1) & np.isfinite(jacobian).all((1, 2))
    cross = np.cross(jacobian[:, :, 0], jacobian[:, :, 1])
    length = np.linalg.norm(cross, axis=1)
    valid &= np.isfinite(length) & (length > _EPS)
    normals = np.full((n, 3), np.nan)
    normals[valid] = cross[valid] / length[valid, None]
    if "normal" in sample:
        provided = np.asarray(sample["normal"], dtype=np.float64)
        if provided.shape != (n, 3):
            raise ValueError("target normal须为Nx3")
        valid &= np.isfinite(provided).all(1) & (np.linalg.norm(provided, axis=1) > _EPS)
        normals[valid] = _unit(provided[valid], "target normal")
        orthogonality = np.einsum("ni,nij->nj", normals[valid], jacobian[valid])
        if not np.allclose(orthogonality, 0, atol=1e-8, rtol=1e-8):
            raise ValueError("target normal与surface Jacobian不正交")
    return position, jacobian, normals, valid, sample


def transport_patch(centers_world, sigma_m, rotation_world_wxyz, rgb, opacity, source_row_ids, *,
                    source_origin, source_frame, sample_target: Callable, patch_id,
                    rotation_rad=0., translation_uv=(0., 0.), scale=1., normal_residual=False,
                    source_uv=None, source_chart_jacobian=None, invalid="raise", source_asset_id=None):
    """连贯运输一个patch；scale默认1，RGB/opacity不做均值、重排或拟合。

    source_frame为列向量[tangent0,tangent1,normal]的右手正交阵。
    可用source_uv和source_chart_jacobian提供非平面源的精确chart。
    默认chart导数在每个源Gaussian的切平面上求值；normal extension保留
    第三轴厚度。normal_residual=True要求源normal与source_frame[:,2]
    一致及callback提供normal_jacobian，以包含偏置曲面的曲率项。
    invalid='drop'时返回有效行并在trace保留原输入valid和丢弃行；默认拒绝。
    """
    centers = np.asarray(centers_world, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or not np.isfinite(centers).all():
        raise ValueError("centers_world须为有限Nx3")
    n = len(centers)
    if n == 0:
        raise ValueError("源patch不能为空")
    if not isinstance(patch_id, (int, str, np.integer)):
        raise ValueError("patch_id须为整数或字符串")
    sigma = _array(sigma_m, (n, 3), "sigma_m")
    if np.any(sigma <= 0):
        raise ValueError("sigma_m须为正")
    source_rotation = quaternion_wxyz_to_matrix(_array(rotation_world_wxyz, (n, 4), "rotation_world_wxyz"))
    source_normals = source_rotation[:, :, 2]
    source_covariance = covariance_from_sigma_quaternion(sigma, rotation_world_wxyz)
    colors = np.asarray(rgb)
    opacities = np.asarray(opacity)
    if colors.shape != (n, 3) or opacities.shape != (n,) or not np.isfinite(colors).all() or not np.isfinite(opacities).all():
        raise ValueError("rgb/opacity须为有限Nx3/N数组")
    if np.any((colors < 0) | (colors > 1)) or np.any((opacities < 0) | (opacities > 1)):
        raise ValueError("rgb/opacity须在[0,1]内")
    row_ids = np.asarray(source_row_ids)
    if row_ids.shape != (n,) or not np.issubdtype(row_ids.dtype, np.integer):
        raise ValueError("source_row_ids须为N整数数组")
    origin = _array(source_origin, (3,), "source_origin")
    frame = _array(source_frame, (3, 3), "source_frame")
    if not np.allclose(frame.T @ frame, np.eye(3), atol=1e-9) or np.linalg.det(frame) < 0:
        raise ValueError("source_frame须为右手正交阵")
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(rotation_rad):
        raise ValueError("scale须为正且rotation_rad有限")
    shift = _array(translation_uv, (2,), "translation_uv")
    cosine, sine = np.cos(rotation_rad), np.sin(rotation_rad)
    matrix = float(scale) * np.array([[cosine, -sine], [sine, cosine]])
    local = (centers - origin) @ frame
    uv = local[:, :2].copy() if source_uv is None else _array(source_uv, (n, 2), "source_uv")
    normal_offset = local[:, 2].copy()
    projection = np.eye(3)[None] - source_normals[:, :, None] * source_normals[:, None, :]
    chart = np.einsum("ab,nbc->nac", frame[:, :2].T, projection) if source_chart_jacobian is None else _array(source_chart_jacobian, (n, 2, 3), "source_chart_jacobian")
    if not np.allclose(np.einsum("nij,nj->ni", chart, source_normals), 0, atol=1e-9):
        raise ValueError("source chart导数必须消去源normal方向")
    if np.any(np.linalg.svd(chart, compute_uv=False)[:, -1] <= _EPS):
        raise ValueError("源chart局部退化")
    target_uv = uv @ matrix.T + shift
    position, surface_jacobian, normals, valid, samples = _sample_target(sample_target, target_uv)
    surface = surface_jacobian.copy()
    if normal_residual:
        if not np.allclose(source_normals, frame[:, 2], atol=1e-8):
            raise ValueError("携带normal residual要求源frame normal与每个Gaussian normal一致")
        if "normal_jacobian" not in samples:
            raise ValueError("携带normal residual必须提供target normal_jacobian")
        derivative = np.asarray(samples["normal_jacobian"], dtype=np.float64)
        if derivative.shape != (n, 3, 2):
            raise ValueError("normal_jacobian须为Nx3x2")
        valid &= np.isfinite(derivative).all((1, 2))
        if not np.allclose(np.einsum("ni,nij->nj", normals[valid], derivative[valid]), 0, atol=1e-8):
            raise ValueError("normal_jacobian必须与normal正交")
        position = position + normal_offset[:, None] * normals
        surface = surface + normal_offset[:, None, None] * derivative
    if invalid not in ("raise", "drop"):
        raise ValueError("invalid须为raise或drop")
    if invalid == "raise" and not valid.all():
        raise ValueError(f"目标surface无效行数: {int((~valid).sum())}")
    selection = np.flatnonzero(valid)
    if not len(selection):
        raise ValueError("目标surface没有有效行")
    jacobian = np.einsum("nij,jk,nkl->nil", surface[valid], matrix, chart[valid])
    jacobian += normals[valid, :, None] * source_normals[valid, None, :]
    if np.any(np.linalg.svd(jacobian, compute_uv=False)[:, -1] <= _EPS):
        raise ValueError("运输Jacobian退化")
    covariance = jacobian @ source_covariance[valid] @ jacobian.transpose(0, 2, 1)
    decomposition = normal_preserving_decomposition(covariance, normals[valid])
    return {"world_xyz": position[valid].copy(), "normal_world": normals[valid].copy(),
            "covariance_world": covariance, "gaussian_sigma_m": decomposition["gaussian_sigma_m"],
            "gaussian_rotation_world_wxyz": decomposition["gaussian_rotation_world_wxyz"],
            "gaussian_rotation_world_override": np.ones(len(selection), bool),
            "reference_srgb": colors[valid].copy(), "road_opacity": opacities[valid].copy(),
            "source_row_id": row_ids[valid].copy(), "patch_id": np.full(len(selection), patch_id),
            "source_uv": uv[valid].copy(), "target_uv": target_uv[valid].copy(),
            "source_normal_residual_m": normal_offset[valid].copy(), "jacobian_world": jacobian,
            "valid": np.ones(len(selection), bool), "source_input_index": selection,
            "trace": {"source_origin": origin.copy(), "source_frame": frame.copy(), "mapping_2d": matrix,
                      "translation_uv": shift.copy(), "scale": float(scale), "rotation_rad": float(rotation_rad),
                      "normal_residual_carried": bool(normal_residual), "normal_extension_scale": 1.,
                      "input_valid": valid.copy(), "dropped_input_indices": np.flatnonzero(~valid),
                      "source_row_ids_all": row_ids.copy(), "source_covariance_world": source_covariance,
                      "source_rotation_world_wxyz": np.asarray(rotation_world_wxyz).copy(),
                      "source_sigma_m": sigma.copy(), "covariance_reconstruction_max_abs": decomposition["reconstruction_max_abs"],
                      "source_kind": "similar_gaussian_patch", "source_asset_id": source_asset_id,
                      "rotation_convention": "world_wxyz_columns_tangent0_tangent1_normal", "world_units": "meter"}}


def world_grid_ownership(points, patch_ids, *, cell_size, unary_cost=None, grid_origin=None,
                         smoothness=0., max_iterations=8):
    """单cell单patch，保留获胜patch的全部候选行。

    points可为任意共同的2D/3D米制world/chart坐标。unary_cost按cell/patch
    求均值，避免高斯数目改变代价。smoothness>0使用确定性Potts ICM；
    它是局部最小误差归属，不声称全局graphcut最优。
    """
    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] not in (2, 3) or not len(coordinates) or not np.isfinite(coordinates).all():
        raise ValueError("ownership points须为非空有限Nx2/Nx3")
    n, dimension = coordinates.shape
    labels = np.asarray(patch_ids)
    if labels.shape != (n,):
        raise ValueError("patch_ids须为N数组")
    if labels.dtype.kind not in "iuUS":
        raise ValueError("patch_ids须为整数或字符串标签")
    size = np.broadcast_to(np.asarray(cell_size, dtype=float), (dimension,)).copy()
    if not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("cell_size须为正")
    origin = np.zeros(dimension) if grid_origin is None else _array(grid_origin, (dimension,), "grid_origin")
    cost = np.zeros(n) if unary_cost is None else _array(unary_cost, (n,), "unary_cost")
    if not np.isfinite(smoothness) or smoothness < 0 or max_iterations < 0:
        raise ValueError("smoothness/max_iterations范围错误")
    keys, inverse = np.unique(np.floor((coordinates - origin) / size).astype(np.int64), axis=0, return_inverse=True)
    # first-occurrence顺序作为稳定tie-break，不按随机或数值颜色排序。
    label_values, first, label_inverse_sorted = np.unique(labels, return_index=True, return_inverse=True)
    order = np.argsort(first)
    remap = np.empty(len(order), dtype=int); remap[order] = np.arange(len(order))
    label_inverse = remap[label_inverse_sorted]
    label_values = label_values[order]
    options = [dict() for _ in keys]
    for cell, label, value in zip(inverse, label_inverse, cost):
        total, count = options[cell].get(int(label), (0., 0))
        options[cell][int(label)] = (total + float(value), count + 1)
    unaries = [{label: value / count for label, (value, count) in item.items()} for item in options]
    winning = np.array([min(item, key=lambda label: (item[label], label)) for item in unaries])
    lookup = {tuple(key): i for i, key in enumerate(keys)}
    neighbours = [[] for _ in keys]
    for i, key in enumerate(keys):
        for axis in range(dimension):
            for sign in (-1, 1):
                neighbour = key.copy(); neighbour[axis] += sign
                if tuple(neighbour) in lookup:
                    neighbours[i].append(lookup[tuple(neighbour)])
    def energy():
        unary = sum(unaries[i][int(label)] for i, label in enumerate(winning))
        boundary = sum(winning[i] != winning[j] for i, row in enumerate(neighbours) for j in row if j > i)
        return float(unary + smoothness * boundary)
    history = [energy()]
    for _ in range(max_iterations if smoothness else 0):
        changed = False
        for i, available in enumerate(unaries):
            current = int(winning[i])
            values = {label: unary + smoothness * sum(winning[j] != label for j in neighbours[i]) for label, unary in available.items()}
            best = min(values, key=lambda label: (values[label], label != current, label))
            if values[best] < values[current] - 1e-12:
                winning[i] = best; changed = True
        history.append(energy())
        if not changed:
            break
    return {"keep_mask": label_inverse == winning[inverse], "cell_keys": keys,
            "cell_patch_id": label_values[winning], "candidate_cell_index": inverse,
            "cell_size": size, "grid_origin": origin, "energy_history": np.asarray(history),
            "method": "world_grid_mean_unary_potts_ICM" if smoothness else "world_grid_minimum_mean_unary",
            "global_graphcut_optimum": False}


def merge_transported_patches(patches: Sequence[dict], *, cell_size, unary_cost=None,
                              ownership_coordinates=None, grid_origin=None, smoothness=0., max_iterations=8):
    """合并共同chart中的多个运输结果；只选择行，不改变任何颜色/协方差。"""
    if not patches:
        raise ValueError("patches不能为空")
    patch_labels = [np.unique(patch["patch_id"]) for patch in patches]
    if any(len(labels) != 1 for labels in patch_labels) or len(set(str(labels[0]) for labels in patch_labels)) != len(patches):
        raise ValueError("每个输入patch须有独立且恒定的patch_id")
    keys = ("world_xyz", "normal_world", "covariance_world", "gaussian_sigma_m", "gaussian_rotation_world_wxyz",
            "gaussian_rotation_world_override",
            "reference_srgb", "road_opacity", "source_row_id", "patch_id", "source_uv", "target_uv",
            "source_normal_residual_m", "jacobian_world", "source_input_index", "valid")
    merged = {key: np.concatenate([patch[key] for patch in patches], axis=0) for key in keys}
    coordinates = merged["world_xyz"] if ownership_coordinates is None else ownership_coordinates
    ownership = world_grid_ownership(coordinates, merged["patch_id"], cell_size=cell_size, unary_cost=unary_cost,
                                     grid_origin=grid_origin, smoothness=smoothness, max_iterations=max_iterations)
    keep = ownership["keep_mask"]
    result = {key: value[keep].copy() for key, value in merged.items()}
    result["trace"] = {"patches": [patch["trace"] for patch in patches], "ownership": ownership,
                       "kept_candidate_indices": np.flatnonzero(keep), "discarded_candidate_indices": np.flatnonzero(~keep),
                       "candidate_source_row_id": merged["source_row_id"], "candidate_patch_id": merged["patch_id"]}
    return result


def target_replacement_mask(centers_world, covariance_world, signed_distance: Callable, *,
                            protected_mask=None, support_sigma=3.):
    """仅选择core内包含整个support球的旧分量；正距离表示core内部。

    signed_distance须为米制、1-Lipschitz距离函数。包围球半径由最大
    特征值给出，保守于椭球；Gaussian无限尾部和真实K贡献由renderer另验。
    本函数返回选择与trace，调用者决定active退役，不修改输入资产。
    """
    centers = np.asarray(centers_world, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3 or not np.isfinite(centers).all():
        raise ValueError("centers_world须为有限Nx3")
    n = len(centers)
    covariance = _array(covariance_world, (n, 3, 3), "covariance_world")
    if not np.allclose(covariance, covariance.transpose(0, 2, 1), atol=1e-14):
        raise ValueError("协方差不对称")
    eigenvalues = np.linalg.eigvalsh(covariance)
    if np.any(eigenvalues <= 0) or not np.isfinite(support_sigma) or support_sigma <= 0:
        raise ValueError("协方差/支持半径非法")
    distance = _array(signed_distance(centers.copy()), (n,), "signed_distance")
    protected = np.zeros(n, bool) if protected_mask is None else np.asarray(protected_mask, dtype=bool)
    if protected.shape != (n,):
        raise ValueError("protected_mask须为N数组")
    radius = float(support_sigma) * np.sqrt(eigenvalues[:, -1])
    return {"replace_mask": (distance >= radius) & ~protected, "core_distance_m": distance,
            "support_radius_m": radius, "protected_mask": protected.copy(), "support_sigma": float(support_sigma),
            "pixel_protection_proof": False}


def deduplicate_centers(centers_world, *, tolerance=1e-6, priority=None, fixed_centers=None):
    """稳定贪心去重复中心；不平均属性，不通过传递链合并远距离纹理点。

    priority越小越优先，平手按原输入行；fixed_centers优先保留。
    representative为原候选行号，-1表示撞到固定旧行，fixed_row给出旧行。
    """
    centers = np.asarray(centers_world, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3 or not np.isfinite(centers).all():
        raise ValueError("centers_world须为有限Nx3")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance须为非负米数")
    n = len(centers)
    order_cost = np.zeros(n) if priority is None else _array(priority, (n,), "priority")
    representative = np.arange(n, dtype=np.int64)
    fixed_row = np.full(n, -1, dtype=np.int64)
    keep = np.ones(n, bool)
    if fixed_centers is not None:
        fixed = np.asarray(fixed_centers, dtype=float)
        if fixed.ndim != 2 or fixed.shape[1] != 3 or not np.isfinite(fixed).all():
            raise ValueError("fixed_centers须为有限Mx3")
        if len(fixed) and n:
            distance, nearest = cKDTree(fixed).query(centers)
            collided = distance <= tolerance
            keep[collided] = False; representative[collided] = -1; fixed_row[collided] = nearest[collided]
    if n:
        tree = cKDTree(centers)
        for index in np.lexsort((np.arange(n), order_cost)):
            if not keep[index]:
                continue
            neighbours = np.asarray(tree.query_ball_point(centers[index], tolerance), dtype=int)
            duplicates = neighbours[keep[neighbours] & (neighbours != index)]
            keep[duplicates] = False; representative[duplicates] = index
    return {"keep_mask": keep, "representative": representative, "fixed_row": fixed_row,
            "kept_indices": np.flatnonzero(keep), "discarded_indices": np.flatnonzero(~keep),
            "tolerance_m": float(tolerance), "method": "stable_greedy_no_attribute_averaging"}

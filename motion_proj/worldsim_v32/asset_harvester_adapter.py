"""把 NVIDIA Asset Harvester PLY 转成 V3.2 actor-local Gaussian schema。"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np

from motion_proj.worldsim_v32.actor_asset_schema import (
    SH_C0,
    fit_gaussians_to_actor_box,
    normalize_quaternions,
    validate_actor_asset,
)


PLY_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def _read_header(handle: BinaryIO) -> tuple[str, int, list[tuple[str, str]]]:
    first = handle.readline().decode("ascii").strip()
    if first != "ply":
        raise ValueError("不是 PLY 文件")
    format_name = ""
    vertex_count = -1
    properties: list[tuple[str, str]] = []
    in_vertex = False
    while True:
        raw = handle.readline()
        if not raw:
            raise ValueError("PLY header 未结束")
        line = raw.decode("ascii").strip()
        parts = line.split()
        if not parts or parts[0] in {"comment", "obj_info"}:
            continue
        if parts[0] == "format":
            format_name = parts[1]
        elif parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
        elif parts[0] == "property" and in_vertex:
            if parts[1] == "list":
                raise ValueError("vertex list property 不受支持")
            properties.append((parts[2], parts[1]))
        elif parts[0] == "end_header":
            break
    if format_name not in {"binary_little_endian", "ascii"}:
        raise ValueError(f"不支持 PLY format: {format_name}")
    if vertex_count <= 0 or not properties:
        raise ValueError("PLY vertex schema 为空")
    return format_name, vertex_count, properties


def read_vertex_ply(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        format_name, count, properties = _read_header(handle)
        if format_name == "binary_little_endian":
            dtype = np.dtype(
                [(name, "<" + PLY_DTYPES[type_name]) for name, type_name in properties]
            )
            rows = np.fromfile(handle, dtype=dtype, count=count)
        else:
            values = np.loadtxt(handle, dtype=np.float64, max_rows=count)
            if values.ndim == 1:
                values = values[None]
            if values.shape != (count, len(properties)):
                raise ValueError("ASCII PLY vertex 行数/列数错误")
            return {
                name: values[:, index] for index, (name, _) in enumerate(properties)
            }
    if len(rows) != count:
        raise ValueError(f"Binary PLY vertex 截断: {len(rows)} != {count}")
    return {name: np.asarray(rows[name]) for name, _ in properties}


def _require(vertex: dict[str, np.ndarray], names: list[str]) -> None:
    missing = [name for name in names if name not in vertex]
    if missing:
        raise ValueError(f"Asset Harvester PLY 缺字段: {missing}")


def load_asset_harvester_ply(path: Path) -> dict[str, np.ndarray]:
    vertex = read_vertex_ply(path)
    fields = [
        "x",
        "y",
        "z",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
    ]
    _require(vertex, fields)
    means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    log_scales = np.stack(
        [vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1
    )
    logits = np.asarray(vertex["opacity"], dtype=np.float64)
    opacity = np.empty_like(logits)
    positive = logits >= 0
    opacity[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_values = np.exp(logits[~positive])
    opacity[~positive] = exp_values / (1.0 + exp_values)
    quats = normalize_quaternions(
        np.stack(
            [vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]],
            axis=1,
        )
    )
    f_dc = np.stack(
        [vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1
    )
    result = {
        "means": means.astype(np.float32),
        "scales": np.exp(log_scales).astype(np.float32),
        "quats": quats.astype(np.float32),
        "rgb": np.clip(f_dc * SH_C0 + 0.5, 0.0, 1.0).astype(np.float32),
        "opacity": np.clip(opacity, 1e-8, 1 - 1e-8).astype(np.float32),
    }
    validate_actor_asset(result)
    return result


def canonicalize_asset_harvester_ply(
    *,
    path: Path,
    target_lwh: np.ndarray,
    orientation_y_degrees: float = 90.0,
    support_sigma: float = 3.0,
) -> dict[str, np.ndarray]:
    native = load_asset_harvester_ply(path)
    fitted = fit_gaussians_to_actor_box(
        means=native["means"],
        scales=native["scales"],
        quats=native["quats"],
        target_lwh=target_lwh,
        orientation_y_degrees=orientation_y_degrees,
        support_sigma=support_sigma,
    )
    result = {
        "means": fitted["means"],
        "scales": fitted["scales"],
        "quats": fitted["quats"],
        "rgb": native["rgb"],
        "opacity": native["opacity"],
        "T_actor_asset": fitted["T_actor_asset"],
        "bounds_lower": fitted["bounds_lower"],
        "bounds_upper": fitted["bounds_upper"],
        "target_lwh": fitted["target_lwh"],
        "scale_xyz": fitted["scale_xyz"],
    }
    validate_actor_asset(result)
    return result


def inject_actor_asset(rigid: object, actor_index: int, asset: dict[str, np.ndarray]) -> dict[str, int]:
    """只替换一个 RigidNodes actor 的 Gaussian，保留冻结轨迹与其他 actor。"""
    import torch
    from torch.nn import Parameter

    validate_actor_asset(asset)
    actor_index = int(actor_index)
    old_ids = rigid.point_ids[..., 0]
    keep = old_ids != actor_index
    old_count = int((~keep).sum().item())
    if old_count == 0:
        raise ValueError(f"RigidNodes 不含 actor index {actor_index}")
    device = rigid._means.device
    dtype = rigid._means.dtype
    means = torch.as_tensor(asset["means"], device=device, dtype=dtype)
    scales = torch.as_tensor(asset["scales"], device=device, dtype=dtype)
    quats = torch.as_tensor(asset["quats"], device=device, dtype=dtype)
    rgb = torch.as_tensor(asset["rgb"], device=device, dtype=dtype).clamp(1e-6, 1 - 1e-6)
    opacity = torch.as_tensor(
        asset["opacity"], device=device, dtype=dtype
    ).clamp(1e-6, 1 - 1e-6)
    if int(rigid.sh_degree) > 0:
        features_dc = (rgb - 0.5) / SH_C0
    else:
        features_dc = torch.logit(rgb)
    features_rest = torch.zeros(
        (len(means),) + tuple(rigid._features_rest.shape[1:]),
        device=device,
        dtype=rigid._features_rest.dtype,
    )
    new_ids = torch.full(
        (len(means), 1),
        actor_index,
        device=rigid.point_ids.device,
        dtype=rigid.point_ids.dtype,
    )
    rigid._means = Parameter(torch.cat([rigid._means[keep], means], dim=0))
    rigid._scales = Parameter(
        torch.cat([rigid._scales[keep], torch.log(scales)], dim=0)
    )
    rigid._quats = Parameter(torch.cat([rigid._quats[keep], quats], dim=0))
    rigid._features_dc = Parameter(
        torch.cat([rigid._features_dc[keep], features_dc], dim=0)
    )
    rigid._features_rest = Parameter(
        torch.cat([rigid._features_rest[keep], features_rest], dim=0)
    )
    rigid._opacities = Parameter(
        torch.cat([rigid._opacities[keep], torch.logit(opacity)[:, None]], dim=0)
    )
    rigid.point_ids = torch.cat([rigid.point_ids[keep], new_ids], dim=0)
    if "target_lwh" in asset:
        rigid.instances_size[actor_index] = torch.as_tensor(
            asset["target_lwh"],
            device=rigid.instances_size.device,
            dtype=rigid.instances_size.dtype,
        )
    return {"removed_gaussians": old_count, "inserted_gaussians": int(len(means))}

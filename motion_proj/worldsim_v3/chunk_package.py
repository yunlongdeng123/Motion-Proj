"""A4-P3 exact chunk package 的物化、重组与确定性裁决工具。"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import torch

from motion_proj.worldsim_v3.contribution_prune import canonical_sha256, tensor_sha256
from motion_proj.worldsim_v3.mixed_precision import checkpoint_schema, recursive_tensor_rows


SENTINEL_TAG = "__worldsim_v3_chunk_row_tensor__"


def atomic_torch_save(path: Path, value: Any) -> None:
    """在同目录原子写入 torch 文件，并禁止覆盖既有证据。"""
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    """流式计算文件摘要。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_sha256(indices: torch.Tensor) -> str:
    """按协议对 little-endian int64 连续字节计算索引摘要。"""
    values = indices.detach().cpu().numpy().astype("<i8", copy=False)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def static_chunk_id(ix: int, iy: int) -> str:
    """按冻结格式生成静态块 ID。"""
    encode = lambda value: f"{'n' if value < 0 else 'p'}{abs(value):04d}"
    return f"static-x-{encode(ix)}-y-{encode(iy)}"


def row_paths_by_model(protocol: Mapping[str, Any]) -> dict[str, list[str]]:
    """展开 common 与 model-specific row tensor 路径。"""
    schema = protocol["row_tensor_schema"]
    common = [str(row["path"]) for row in schema["common_gaussian_row_tensors"]]
    return {
        model: common
        + [str(row["path"]) for row in spec["additional_row_tensors"]]
        for model, spec in schema["models"].items()
    }


def row_schema_by_model(protocol: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """返回可直接写入 manifest 的规范化 row schema。"""
    schema = protocol["row_tensor_schema"]
    common = [
        {
            "path": str(row["path"]),
            "dtype": str(row["dtype"]),
            "shape_tail": [int(value) for value in row["shape_tail"]],
        }
        for row in schema["common_gaussian_row_tensors"]
    ]
    return {
        model: common
        + [
            {
                "path": str(row["path"]),
                "dtype": str(row["dtype"]),
                "shape_tail": [int(value) for value in row["shape_tail"]],
            }
            for row in spec["additional_row_tensors"]
        ]
        for model, spec in schema["models"].items()
    }


def dotted_get(value: Mapping[str, Any], path: str) -> Any:
    """从仅含 mapping 的 checkpoint 子树读取点分路径。"""
    current: Any = value
    for name in path.split("."):
        current = current[name]
    return current


def _mapping_like(source: Mapping[str, Any], items: Sequence[tuple[str, Any]]) -> Any:
    """尽量保持 OrderedDict 等原容器类型。"""
    if isinstance(source, OrderedDict):
        return OrderedDict(items)
    try:
        return type(source)(items)
    except TypeError:
        return dict(items)


def build_skeleton(
    source: Mapping[str, Any], protocol: Mapping[str, Any]
) -> tuple[MutableMapping[str, Any], int]:
    """保留全部共享状态，并把 51 个 row tensor 替换为唯一 sentinel。"""
    full_paths = {
        f"models.{model}.{path}": (model, path)
        for model, paths in row_paths_by_model(protocol).items()
        for path in paths
    }
    seen: set[str] = set()

    def transform(value: Any, prefix: str = "") -> Any:
        if prefix in full_paths:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"P3 row path is not tensor: {prefix}")
            model, field_path = full_paths[prefix]
            seen.add(prefix)
            return {
                "tag": SENTINEL_TAG,
                "model": model,
                "field_path": field_path,
                "rows": int(value.shape[0]),
                "shape_tail": [int(item) for item in value.shape[1:]],
                "dtype": str(value.dtype).removeprefix("torch."),
            }
        if isinstance(value, Mapping):
            return _mapping_like(
                value,
                [
                    (
                        str(name),
                        transform(child, f"{prefix}.{name}" if prefix else str(name)),
                    )
                    for name, child in value.items()
                ],
            )
        if isinstance(value, list):
            return [transform(child, f"{prefix}[{index}]") for index, child in enumerate(value)]
        if isinstance(value, tuple):
            return tuple(
                transform(child, f"{prefix}[{index}]") for index, child in enumerate(value)
            )
        return value

    skeleton = transform(source)
    if seen != set(full_paths):
        missing = sorted(set(full_paths) - seen)
        raise RuntimeError(f"P3 skeleton row paths missing: {missing}")
    return skeleton, len(seen)


def _non_tensor_signature(value: Any) -> Any:
    """绑定容器类型、顺序与非 tensor 值，同时把 tensor 化为结构标记。"""
    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "dtype": str(value.dtype).removeprefix("torch."),
            "shape": [int(item) for item in value.shape],
        }
    if isinstance(value, Mapping):
        return {
            "kind": type(value).__name__,
            "items": [
                (str(name), _non_tensor_signature(child)) for name, child in value.items()
            ],
        }
    if isinstance(value, list):
        return {"kind": "list", "items": [_non_tensor_signature(child) for child in value]}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_non_tensor_signature(child) for child in value]}
    return {"kind": type(value).__name__, "repr": repr(value)}


def non_tensor_signature_sha256(value: Any) -> str:
    """生成共享 scalar/container 状态的稳定摘要。"""
    return canonical_sha256(_non_tensor_signature(value))


def row_tensor_digest(row_tensors: Mapping[str, torch.Tensor]) -> str:
    """绑定一个 asset 内全部 row tensor 的顺序、dtype、shape 与值。"""
    lines = []
    for path, tensor in row_tensors.items():
        lines.append(
            f"{path}\t{str(tensor.dtype).removeprefix('torch.')}\t"
            f"{json.dumps(list(tensor.shape), separators=(',', ':'))}\t"
            f"{tensor_sha256(tensor)}\n"
        )
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def _extract_rows(
    model: Mapping[str, Any], paths: Sequence[str], indices: torch.Tensor
) -> OrderedDict[str, torch.Tensor]:
    """按严格升序源索引抽取一个 asset 的全部字段。"""
    output: OrderedDict[str, torch.Tensor] = OrderedDict()
    for path in paths:
        source = dotted_get(model, path)
        output[path] = source.index_select(0, indices).contiguous()
    return output


def static_memberships(
    means: torch.Tensor, contract: Mapping[str, Any]
) -> list[tuple[int, int, torch.Tensor]]:
    """重算固定 50 m 网格的非空 cell 与升序源索引。"""
    origin = torch.tensor(contract["origin_xy_m"], dtype=torch.float64)
    cells = torch.floor(
        (means[:, :2].double() - origin) / float(contract["cell_size_m"])
    ).to(torch.int64)
    unique = sorted(tuple(map(int, row)) for row in torch.unique(cells, dim=0).tolist())
    return [
        (
            ix,
            iy,
            torch.nonzero(
                (cells[:, 0] == ix) & (cells[:, 1] == iy), as_tuple=False
            ).reshape(-1),
        )
        for ix, iy in unique
    ]


def actor_memberships(
    points_ids: torch.Tensor, contract: Mapping[str, Any]
) -> list[tuple[int, torch.Tensor]]:
    """按 points_ids 生成 24 个 actor 的显式升序源索引。"""
    values = points_ids.reshape(-1).to(torch.int64)
    first, last = map(int, contract["actor_index_domain_inclusive"])
    return [
        (
            actor_index,
            torch.nonzero(values == actor_index, as_tuple=False).reshape(-1),
        )
        for actor_index in range(first, last + 1)
    ]


def _bounds_xyz(means: torch.Tensor) -> dict[str, list[float]] | None:
    """返回 asset 的 XYZ 边界；空 actor 使用 null。"""
    if means.shape[0] == 0:
        return None
    return {
        "min": [float(value) for value in means.amin(dim=0)],
        "max": [float(value) for value in means.amax(dim=0)],
    }


def _asset_record(root: Path, path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """把已写 asset 转为 manifest 记录。"""
    row_tensors = payload["row_tensors"]
    record = {
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "asset_kind": payload["asset_kind"],
        "asset_id": payload["asset_id"],
        "model": payload["model"],
        "source_row_count": int(payload["source_row_count"]),
        "row_count": int(payload["row_count"]),
        "source_flat_indices_sha256": index_sha256(payload["source_flat_indices"]),
        "row_tensor_digest": row_tensor_digest(row_tensors),
        "bounds_xyz": _bounds_xyz(row_tensors["_means"]),
    }
    for name in ("cell", "actor_index", "availability"):
        if name in payload:
            record[name] = payload[name]
    return record


def materialize_chunk_package(
    source: Mapping[str, Any],
    *,
    package_root: Path,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    project_commit: str,
) -> dict[str, Any]:
    """一次性物化 skeleton、133 个 static 与 24 个 actor assets。"""
    if package_root.exists():
        raise FileExistsError(package_root)
    (package_root / "static").mkdir(parents=True)
    (package_root / "actors").mkdir()
    paths = row_paths_by_model(protocol)
    schemas = row_schema_by_model(protocol)
    models = source["models"]

    skeleton, sentinel_count = build_skeleton(source, protocol)
    skeleton_path = package_root / "skeleton.pth"
    atomic_torch_save(skeleton_path, skeleton)
    skeleton_record = {
        "path": "skeleton.pth",
        "sha256": sha256_file(skeleton_path),
        "bytes": skeleton_path.stat().st_size,
        "row_tensor_sentinel_count": sentinel_count,
        "non_tensor_signature_sha256": non_tensor_signature_sha256(skeleton),
    }

    static_records = []
    for ix, iy, indices in static_memberships(
        models["Background"]["_means"], protocol["static_chunk_contract"]
    ):
        asset_id = static_chunk_id(ix, iy)
        payload = OrderedDict(
            schema_version=1,
            package_format=protocol["package_contract"]["package_format"],
            asset_kind="static",
            asset_id=asset_id,
            model="Background",
            source_row_count=int(models["Background"]["_means"].shape[0]),
            row_count=int(indices.numel()),
            source_flat_indices=indices.to(torch.int64).contiguous(),
            row_tensors=_extract_rows(models["Background"], paths["Background"], indices),
            cell={"ix": ix, "iy": iy},
        )
        path = package_root / "static" / f"{asset_id}.pth"
        atomic_torch_save(path, payload)
        static_records.append(_asset_record(package_root, path, payload))

    actor_records = []
    for actor_index, indices in actor_memberships(
        models["RigidNodes"]["points_ids"], protocol["actor_chunk_contract"]
    ):
        asset_id = f"actor-{actor_index:04d}"
        availability = "available" if indices.numel() else "unavailable_empty"
        payload = OrderedDict(
            schema_version=1,
            package_format=protocol["package_contract"]["package_format"],
            asset_kind="actor",
            asset_id=asset_id,
            model="RigidNodes",
            source_row_count=int(models["RigidNodes"]["_means"].shape[0]),
            row_count=int(indices.numel()),
            source_flat_indices=indices.to(torch.int64).contiguous(),
            row_tensors=_extract_rows(models["RigidNodes"], paths["RigidNodes"], indices),
            actor_index=actor_index,
            availability=availability,
        )
        path = package_root / "actors" / f"{asset_id}.pth"
        atomic_torch_save(path, payload)
        actor_records.append(_asset_record(package_root, path, payload))

    payload_records = [skeleton_record, *static_records, *actor_records]
    payload_lines = "".join(
        f"{row['sha256']}  ./{row['path']}\n"
        for row in sorted(payload_records, key=lambda row: row["path"])
    )
    manifest = {
        "schema_version": 1,
        "status": "done",
        "package_format": protocol["package_contract"]["package_format"],
        "protocol_sha256": protocol_sha256,
        "project_commit": project_commit,
        "source_checkpoint": dict(protocol["selected_asset"]["checkpoint"]),
        "source_actor_registry": dict(protocol["selected_asset"]["actor_registry"]),
        "static_grid": dict(protocol["static_chunk_contract"]),
        "actor_contract": {
            key: value
            for key, value in protocol["actor_chunk_contract"].items()
            if key != "expected_actors"
        },
        "row_tensor_schema": schemas,
        "skeleton": skeleton_record,
        "static_assets": static_records,
        "actor_assets": actor_records,
        "counts": {
            "static_assets": len(static_records),
            "actor_assets": len(actor_records),
            "data_assets": len(static_records) + len(actor_records),
            "payload_files": len(payload_records),
        },
        "payload_bytes": sum(int(row["bytes"]) for row in payload_records),
        "payload_sha256": hashlib.sha256(payload_lines.encode("utf-8")).hexdigest(),
        "source_non_tensor_signature_sha256": non_tensor_signature_sha256(source),
    }
    return manifest


def _validate_file(root: Path, record: Mapping[str, Any]) -> Path:
    """核对 manifest 中一个 payload 文件的 bytes 与 SHA。"""
    path = root / str(record["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"P3 package payload drift: {path}")
    return path


def _allocate_from_skeleton(
    value: Any,
    tensors: MutableMapping[tuple[str, str], torch.Tensor],
) -> Any:
    """把 sentinel 还原成精确 shape/dtype 的 CPU tensor。"""
    if isinstance(value, Mapping) and value.get("tag") == SENTINEL_TAG:
        dtype = getattr(torch, str(value["dtype"]))
        tensor = torch.empty(
            [int(value["rows"]), *[int(item) for item in value["shape_tail"]]],
            dtype=dtype,
        )
        tensors[(str(value["model"]), str(value["field_path"]))] = tensor
        return tensor
    if isinstance(value, Mapping):
        return _mapping_like(
            value,
            [(str(name), _allocate_from_skeleton(child, tensors)) for name, child in value.items()],
        )
    if isinstance(value, list):
        return [_allocate_from_skeleton(child, tensors) for child in value]
    if isinstance(value, tuple):
        return tuple(_allocate_from_skeleton(child, tensors) for child in value)
    return value


def _strictly_ascending(indices: torch.Tensor) -> bool:
    """空/单元素索引也视为严格升序。"""
    return bool(indices.numel() < 2 or torch.all(indices[1:] > indices[:-1]).item())


def reassemble_chunk_package(
    *,
    package_root: Path,
    manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[MutableMapping[str, Any], dict[str, Any]]:
    """校验全部 payload 并按源索引在内存中重组完整 checkpoint。"""
    expected_format = protocol["package_contract"]["package_format"]
    if manifest["package_format"] != expected_format:
        raise RuntimeError("P3 package format drift")
    skeleton_path = _validate_file(package_root, manifest["skeleton"])
    skeleton = torch.load(skeleton_path, map_location="cpu")
    tensors: dict[tuple[str, str], torch.Tensor] = {}
    state = _allocate_from_skeleton(skeleton, tensors)
    expected_paths = row_paths_by_model(protocol)
    if set(tensors) != {
        (model, path) for model, paths in expected_paths.items() for path in paths
    }:
        raise RuntimeError("P3 skeleton sentinel schema drift")

    coverage = {
        model: torch.zeros(
            int(protocol["row_tensor_schema"]["models"][model]["row_count"]),
            dtype=torch.int8,
        )
        for model in expected_paths
    }
    asset_rows = [*manifest["static_assets"], *manifest["actor_assets"]]
    manifest_exact = True
    row_fields_exact = True
    cell_membership_exact = True
    actor_membership_exact = True
    for record in asset_rows:
        path = _validate_file(package_root, record)
        payload = torch.load(path, map_location="cpu")
        model = str(record["model"])
        indices = payload["source_flat_indices"].to(torch.int64)
        row_tensors = payload["row_tensors"]
        manifest_exact = manifest_exact and all(
            (
                payload.get(name) == record.get(name)
                if name not in {"source_flat_indices_sha256", "row_tensor_digest", "bounds_xyz"}
                else True
            )
            for name in ("asset_kind", "asset_id", "model", "source_row_count", "row_count")
        )
        manifest_exact = manifest_exact and (
            int(indices.numel()) == int(record["row_count"])
            and index_sha256(indices) == record["source_flat_indices_sha256"]
            and row_tensor_digest(row_tensors) == record["row_tensor_digest"]
            and _bounds_xyz(row_tensors["_means"]) == record["bounds_xyz"]
        )
        row_fields_exact = row_fields_exact and (
            list(row_tensors) == expected_paths[model]
            and _strictly_ascending(indices)
            and all(
                int(tensor.shape[0]) == int(indices.numel())
                and str(tensor.dtype).removeprefix("torch.")
                == next(
                    row["dtype"]
                    for row in manifest["row_tensor_schema"][model]
                    if row["path"] == field
                )
                and list(tensor.shape[1:])
                == next(
                    row["shape_tail"]
                    for row in manifest["row_tensor_schema"][model]
                    if row["path"] == field
                )
                for field, tensor in row_tensors.items()
            )
        )
        if indices.numel():
            if int(indices.min()) < 0 or int(indices.max()) >= coverage[model].numel():
                raise RuntimeError(f"P3 asset indices out of range: {record['asset_id']}")
            coverage[model].index_add_(
                0, indices, torch.ones(indices.numel(), dtype=torch.int8)
            )
        if record["asset_kind"] == "static":
            ix = int(record["cell"]["ix"])
            iy = int(record["cell"]["iy"])
            origin = torch.tensor(
                protocol["static_chunk_contract"]["origin_xy_m"], dtype=torch.float64
            )
            cells = torch.floor(
                (row_tensors["_means"][:, :2].double() - origin)
                / float(protocol["static_chunk_contract"]["cell_size_m"])
            ).to(torch.int64)
            cell_membership_exact = cell_membership_exact and bool(
                cells.shape[0] == 0
                or torch.all(
                    (cells[:, 0] == ix) & (cells[:, 1] == iy)
                ).item()
            )
        else:
            actor_index = int(record["actor_index"])
            actor_membership_exact = actor_membership_exact and bool(
                row_tensors["points_ids"].numel() == 0
                or torch.all(row_tensors["points_ids"].reshape(-1) == actor_index).item()
            )
        for field, tensor in row_tensors.items():
            if indices.numel():
                tensors[(model, field)].index_copy_(0, indices, tensor)

    coverage_exact = all(torch.all(values == 1).item() for values in coverage.values())
    audit = {
        "payload_files_verified": len(asset_rows) + 1,
        "manifest_records_exact": bool(manifest_exact),
        "row_fields_exact": bool(row_fields_exact),
        "static_cell_membership_exact": bool(cell_membership_exact),
        "actor_membership_exact": bool(actor_membership_exact),
        "indices_unique_disjoint_exhaustive": bool(coverage_exact),
        "model_coverage": {
            model: {
                "rows": int(values.numel()),
                "covered_once": int((values == 1).sum()),
                "missing": int((values == 0).sum()),
                "duplicated": int((values > 1).sum()),
            }
            for model, values in coverage.items()
        },
    }
    return state, audit


def compare_checkpoint_states(
    source: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """比较完整 checkpoint 的容器、tensor 与共享 scalar 状态。"""
    source_rows = recursive_tensor_rows(source)
    candidate_rows = recursive_tensor_rows(candidate)
    tensor_paths_exact = set(source_rows) == set(candidate_rows)
    tensor_rows_exact = tensor_paths_exact and all(
        source_rows[path] == candidate_rows[path] for path in source_rows
    )
    return {
        "recursive_container_schema_exact": checkpoint_schema(source)
        == checkpoint_schema(candidate),
        "tensor_path_count": len(source_rows),
        "tensor_paths_exact": tensor_paths_exact,
        "tensor_shape_dtype_value_sha256_exact": tensor_rows_exact,
        "non_tensor_signature_sha256_source": non_tensor_signature_sha256(source),
        "non_tensor_signature_sha256_candidate": non_tensor_signature_sha256(candidate),
        "non_tensor_values_exact": non_tensor_signature_sha256(source)
        == non_tensor_signature_sha256(candidate),
        "all_exact": checkpoint_schema(source) == checkpoint_schema(candidate)
        and tensor_rows_exact
        and non_tensor_signature_sha256(source) == non_tensor_signature_sha256(candidate),
    }


def select_chunk_arm(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """只在全部冻结 integrity/quality/resource 门通过时选择 chunk package。"""
    required = (
        "exact_static_and_actor_asset_inventory",
        "exact_row_fields_and_shared_skeleton_without_duplication",
        "exact_package_manifest_hashes_bytes_counts_bounds_and_indices",
        "bitwise_exact_full_checkpoint_reassembly_and_reload",
        "p2_mixed_precision_runtime_adapter_exact",
        "source_baseline_replay_matches_p2_exact",
        "all_57_rgb_hashes_and_all_31_quality_endpoints_exact",
        "source_inputs_unchanged",
        "resources_within_frozen_ceilings",
    )
    if all(bool(candidate.get(name)) for name in required):
        return {
            "selected_arm": "p3-chunk-package",
            "method_state": "selected_exact_chunk_package",
            "fallback_exact_alias": False,
        }
    return {
        "selected_arm": "p3-source",
        "method_state": "rejected_chunk_integrity_quality_or_resource_gate",
        "fallback_exact_alias": True,
    }

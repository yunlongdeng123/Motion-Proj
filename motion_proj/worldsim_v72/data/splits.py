"""日志级数据角色清单的读取与互斥检查。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


FINAL_ROLES = frozenset({"route_select", "source_test", "external_test"})


def load_data_roles(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_data_roles(payload)
    return payload


def validate_data_roles(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != "worldsim_v72.data_roles.v1":
        raise ValueError("data roles schema 版本不匹配")
    datasets = payload.get("datasets")
    if not isinstance(datasets, Mapping) or not datasets:
        raise ValueError("data roles 必须包含 datasets")
    for dataset_name, dataset in datasets.items():
        if not isinstance(dataset, Mapping):
            raise ValueError(f"dataset {dataset_name} 配置无效")
        roles = dataset.get("group_roles")
        if not isinstance(roles, Mapping):
            raise ValueError(f"dataset {dataset_name} 缺少 group_roles")
        owner: dict[str, str] = {}
        for role, group_ids in roles.items():
            if not isinstance(group_ids, list):
                raise ValueError(f"dataset {dataset_name} role {role} 必须为列表")
            for group_id in group_ids:
                group = str(group_id)
                if group in owner:
                    raise ValueError(
                        f"dataset {dataset_name} group {group} 同时属于 {owner[group]} 和 {role}"
                    )
                owner[group] = str(role)
        frozen = dataset.get("frozen_roles", {})
        if not isinstance(frozen, Mapping):
            raise ValueError(f"dataset {dataset_name} frozen_roles 配置无效")
        for role in FINAL_ROLES:
            if roles.get(role) and not bool(frozen.get(role, False)):
                raise ValueError(f"dataset {dataset_name} 的 {role} 有成员但尚未冻结")


def require_role_access(payload: Mapping[str, Any], dataset_name: str, role: str) -> list[str]:
    dataset = payload["datasets"][dataset_name]
    if role in FINAL_ROLES and not bool(dataset.get("frozen_roles", {}).get(role, False)):
        raise PermissionError(f"{dataset_name}/{role} 尚未冻结，禁止读取")
    return [str(value) for value in dataset["group_roles"].get(role, [])]

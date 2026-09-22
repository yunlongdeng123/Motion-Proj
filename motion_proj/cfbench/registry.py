"""Model registry and capability resolution for the V7.5 pilot."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .schema import ContractError, capability_key


SUPPORT_LEVELS = ("native", "adapted", "consumer_only", "unsupported")


def load_registry(path: str | Path) -> dict[str, Any]:
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_registry(registry)
    return registry


def validate_registry(registry: Mapping[str, Any]) -> None:
    if registry.get("schema_version") != "worldsim_v75_cfbench_model_registry_v1":
        raise ContractError("invalid model registry schema_version")
    models = registry.get("models")
    if not isinstance(models, list) or not models:
        raise ContractError("registry.models must be a non-empty list")
    ids: set[str] = set()
    for model in models:
        if not isinstance(model, Mapping):
            raise ContractError("registry.models[] must be a mapping")
        model_id = model.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            raise ContractError("model_id must be non-empty")
        if model_id in ids:
            raise ContractError(f"duplicate model_id: {model_id}")
        ids.add(model_id)
        capabilities = model.get("capabilities")
        if not isinstance(capabilities, Mapping):
            raise ContractError(f"{model_id}.capabilities must be a mapping")
        for key, level in capabilities.items():
            if not isinstance(key, str) or level not in SUPPORT_LEVELS:
                raise ContractError(f"invalid capability {model_id}/{key}: {level}")


def model_map(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["model_id"]): row for row in registry["models"]}


def support_for_case(model: Mapping[str, Any], case: Mapping[str, Any]) -> str:
    key = capability_key(case)
    return str(model.get("capabilities", {}).get(key, "unsupported"))

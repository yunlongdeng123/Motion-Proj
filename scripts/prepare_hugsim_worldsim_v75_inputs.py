#!/usr/bin/env python3
"""整理 HUGSIM 官方 scene-0383 的本地路径与已发布 scenario。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


LEGACY_SUFFIX = "/postprocess/shadow.pth"


def _normalize_scenario(payload: dict[str, Any], car_root: Path) -> tuple[dict[str, Any], list[str]]:
    referenced: list[str] = []
    for actor in payload.get("plan_list", []):
        relative = str(actor[5])
        if relative.endswith(LEGACY_SUFFIX):
            relative = relative[: -len(LEGACY_SUFFIX)]
            actor[5] = relative
        car_id = relative.split("/", 1)[0]
        car_dir = car_root / car_id
        for required in ("gs.pth", "wlh.json"):
            if not (car_dir / required).is_file():
                raise FileNotFoundError(f"HUGSIM vehicle asset is missing: {car_dir / required}")
        referenced.append(car_id)
    return payload, referenced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    runtime = args.asset_root.resolve() / "hugsim" / "runtime" / "official"
    scene_root = runtime / "scenes"
    car_root = runtime / "3DRealCar"
    scenario_root = runtime / "scenarios" / "nuscenes"
    map_root = runtime / "map_cache" / "nusc_map_cache"
    for required in (
        scene_root / "scene-0383" / "scene.pth",
        scene_root / "scene-0383" / "cfg.yaml",
        scene_root / "scene-0383" / "ground_param.pkl",
    ):
        if not required.is_file():
            raise FileNotFoundError(f"HUGSIM exported scene is missing: {required}")

    output_root = args.output_root.resolve()
    output_scenarios = output_root / "scenarios"
    output_scenarios.mkdir(parents=True, exist_ok=True)
    scenario_rows = []
    for source in sorted(scenario_root.glob("scene-0383-*.yaml")):
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        payload, referenced = _normalize_scenario(payload, car_root)
        target = output_scenarios / source.name
        target.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        scenario_rows.append(
            {
                "name": source.name,
                "path": str(target),
                "mode": payload["mode"],
                "referenced_car_ids": referenced,
            }
        )

    base_config = {
        "realcar_path": str(car_root),
        "model_base": str(scene_root),
        "uniad_path": "__SET_WHEN_UNIAD_CLIENT_IS_AVAILABLE__",
        "vad_path": "__SET_WHEN_VAD_CLIENT_IS_AVAILABLE__",
        "ltf_path": "__SET_WHEN_LTF_CLIENT_IS_AVAILABLE__",
        "output_dir": "/root/autodl-tmp/runs/worldsim_v75_downstream_bench/hugsim/",
        "HD_map": {"path": str(map_root), "version": "nusc_trainval"},
    }
    base_path = output_root / "nuscenes_base.local.yaml"
    base_path.write_text(
        yaml.safe_dump(base_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    index = {
        "schema_version": "worldsim_v75_hugsim_inputs_v1",
        "gpu_operations_run": False,
        "source_scene": "scene-0383",
        "benchmark_aligned": False,
        "reason": "官方 smoke 场景，不是 DriveStudio 179/191/204 对齐场景",
        "scene_root": str(scene_root),
        "car_root": str(car_root),
        "base_config": str(base_path),
        "camera_config": str(args.source_root.resolve() / "configs" / "sim" / "nuscenes_camera.yaml"),
        "kinematic_config": str(args.source_root.resolve() / "configs" / "sim" / "kinematic.yaml"),
        "scenarios": scenario_rows,
        "asset_smoke_ready": bool(scenario_rows),
        "closed_loop_ad_client_ready": False,
        "model_execution_run": False,
    }
    index_path = output_root / "index.json"
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"scenario_count": len(scenario_rows), "index": str(index_path)}, indent=2))


if __name__ == "__main__":
    main()

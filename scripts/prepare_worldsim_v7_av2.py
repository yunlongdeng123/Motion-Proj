"""串行将冻结 AV2 log cohort 转为 V7 SceneIR 元数据。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from uuid import UUID

from motion_proj.worldsim_v7.sceneir_adapter import AV2SceneIRAdapter, scene_ir_to_dict


def validate_cohort(cohort: dict[str, Any]) -> dict[str, Any]:
    """在读取 AV2 payload 前强制检查冻结的 zero-shot 合同。"""

    if cohort.get("schema_version") != "worldsim_v7.av2_zero_shot_cohort.v1":
        raise ValueError("unsupported AV2 cohort schema")
    if cohort.get("source_split") != "val":
        raise ValueError("V7 AV2 cohort must use the Sensor val split")
    forbidden_flags = (
        "quality_read_before_freeze",
        "fine_tuning_allowed",
        "post_hoc_calibration_allowed",
        "threshold_selection_allowed",
        "failed_scene_deletion_allowed",
    )
    if any(cohort.get(flag) is not False for flag in forbidden_flags):
        raise ValueError("V7 AV2 zero-shot prohibitions must be explicitly false")

    rows = list(cohort.get("logs", []))
    expected_indices = list(range(0, 150, 5))
    if [int(row["index"]) for row in rows] != expected_indices:
        raise ValueError("AV2 cohort must contain the frozen every-fifth indices")
    expected_roles = ["quantitative"] * 20 + ["qualitative"] * 10
    if [str(row["role"]) for row in rows] != expected_roles:
        raise ValueError("AV2 cohort must contain 20 quantitative then 10 qualitative logs")
    log_ids = [str(row["log_id"]) for row in rows]
    if len(set(log_ids)) != len(log_ids):
        raise ValueError("AV2 cohort log IDs must be unique")
    if any(str(UUID(log_id)) != log_id for log_id in log_ids):
        raise ValueError("AV2 cohort log IDs must be canonical UUIDs")
    return cohort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    cohort = validate_cohort(json.loads(args.cohort.read_text(encoding="utf-8")))
    args.output_root.mkdir(parents=True, exist_ok=True)
    adapter = AV2SceneIRAdapter()
    summary_rows = []
    for item in cohort["logs"]:
        log_id = str(item["log_id"])
        source = args.dataset_root / log_id
        destination = args.output_root / f"{log_id}.sceneir.json"
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite {destination}")
        scene = adapter.build_scene_ir(source)
        payload = scene_ir_to_dict(scene)
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        summary_rows.append(
            {
                "log_id": log_id,
                "role": str(item["role"]),
                "actor_count": len(scene.actors),
                "ego_pose_count": len(scene.ego_poses),
                "sensor_count": len(scene.sensor_calibrations),
                "sceneir": destination.name,
            }
        )

    manifest = {
        "schema_version": "worldsim_v7.av2_sceneir_preprocess_manifest.v1",
        "source_split": cohort["source_split"],
        "selection_policy": cohort["selection_policy"],
        "logs": summary_rows,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

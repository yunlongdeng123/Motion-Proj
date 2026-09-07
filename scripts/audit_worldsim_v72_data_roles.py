"""仅用 metadata 合并 V7.1 的 nuScenes/AV2 数据暴露记录。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _cohort_ids(path: Path) -> set[str]:
    payload = _read_json(path)
    return {str(row["log_id"]) for row in payload.get("logs", [])}


def build_inventory(
    repo_root: Path, metadata_root: Path, av2_root: Path, v71_corpus_root: Path
) -> dict[str, Any]:
    split_path = repo_root / "configs/worldsim_v71/source_split_v1.json"
    source_split = _read_json(split_path)
    scenes = _read_json(metadata_root / "scene.json")
    scene_to_log = {str(row["name"]): str(row["log_token"]) for row in scenes}
    role_scenes = {
        **{str(role): list(values) for role, values in source_split["roles"].items()},
        "train_reserve": list(source_split.get("train_reserve", [])),
    }
    role_scenes["v71_model_train"] = sorted(
        path.name for path in (v71_corpus_root / "train").iterdir() if path.is_dir()
    )
    fresh_final = _read_json(
        repo_root / "configs/worldsim_v7/nuscenes_fresh_final_cohort_v1.json"
    )
    role_scenes["v7_fresh_final"] = [
        str(row["scene_name"]) for row in fresh_final["scenes"]
    ]
    role_logs: dict[str, list[str]] = {}
    unknown_scenes: dict[str, list[str]] = {}
    ownership: dict[str, list[str]] = defaultdict(list)
    for role, names in role_scenes.items():
        unknown_scenes[role] = sorted(name for name in names if name not in scene_to_log)
        logs = sorted({scene_to_log[name] for name in names if name in scene_to_log})
        role_logs[role] = logs
        for log_id in logs:
            ownership[log_id].append(role)
    overlaps = {
        log_id: roles for log_id, roles in sorted(ownership.items()) if len(roles) > 1
    }
    declared_roles = {"train", "selection", "source_final", "train_reserve"}
    declared_overlaps = {
        log_id: [role for role in roles if role in declared_roles]
        for log_id, roles in sorted(ownership.items())
        if len([role for role in roles if role in declared_roles]) > 1
    }
    all_log_ids = sorted({str(row["log_token"]) for row in scenes})
    assigned_log_ids = set(ownership)

    cohort_paths = [
        repo_root / "configs/worldsim_v7/av2_zero_shot_cohort_v1.json",
        repo_root / "configs/worldsim_v7/av2_zero_shot_recovery_cohort_v1.json",
        repo_root / "configs/worldsim_v7/av2_evicomp_fresh_cohort_v1.json",
        repo_root / "configs/worldsim_v71/av2_zero_shot_cohort_v1.json",
    ]
    consumed_by_cohort = {
        str(path.relative_to(repo_root)): sorted(_cohort_ids(path)) for path in cohort_paths
    }
    consumed_av2 = set().union(*(set(values) for values in consumed_by_cohort.values()))
    available_av2 = sorted(path.name for path in av2_root.iterdir() if path.is_dir())

    return {
        "schema_version": "worldsim_v72.data_role_inventory.v1",
        "selection_basis": "metadata_and_existing_exposure_only",
        "quality_read": False,
        "source": {
            "metadata_scene_count": len(scenes),
            "metadata_log_count": len(all_log_ids),
            "v71_corpus_actor_file_count": sum(
                1 for _ in (v71_corpus_root / "train").glob("*/*.npz")
            ),
            "role_scene_counts": {role: len(values) for role, values in role_scenes.items()},
            "role_log_counts": {role: len(values) for role, values in role_logs.items()},
            "role_logs": role_logs,
            "declared_cross_role_log_overlap": declared_overlaps,
            "cross_role_log_overlap": overlaps,
            "unknown_scenes": {role: values for role, values in unknown_scenes.items() if values},
            "unassigned_log_ids": sorted(set(all_log_ids) - assigned_log_ids),
        },
        "external": {
            "consumed_by_cohort": consumed_by_cohort,
            "consumed_log_count": len(consumed_av2),
            "available_on_disk_count": len(available_av2),
            "available_on_disk_log_ids": available_av2,
            "available_unconsumed_on_disk_log_ids": sorted(set(available_av2) - consumed_av2),
        },
    }


def build_role_manifest(inventory: Mapping[str, Any]) -> dict[str, Any]:
    source = inventory["source"]
    role_logs = source["role_logs"]
    train = set(role_logs["v71_model_train"])
    exposed_evaluation = (
        set(role_logs["selection"])
        | set(role_logs["source_final"])
        | set(role_logs["v7_fresh_final"])
    )
    legacy_diagnostic = exposed_evaluation - train
    source_candidates = (
        set(source["unassigned_log_ids"]) | set(role_logs["train_reserve"])
    ) - train - legacy_diagnostic
    all_source_logs = set(source["unassigned_log_ids"]).union(
        *(set(values) for values in role_logs.values())
    )
    exposure_unknown = all_source_logs - train - legacy_diagnostic - source_candidates

    external = inventory["external"]
    consumed_external = set().union(
        *(set(values) for values in external["consumed_by_cohort"].values())
    )
    return {
        "schema_version": "worldsim_v72.data_roles.v1",
        "status": "p0_inventory_not_final_split",
        "split_unit": "driving_log",
        "quality_used_for_selection": False,
        "datasets": {
            "nuscenes": {
                "group_id_kind": "log_token",
                "group_roles": {
                    "train": sorted(train),
                    "legacy_diagnostic": sorted(legacy_diagnostic),
                    "exposure_unknown": sorted(exposure_unknown),
                    "source_candidate_pool": sorted(source_candidates),
                    "dev": [],
                    "route_select": [],
                    "source_test": [],
                },
                "frozen_roles": {"route_select": False, "source_test": False},
                "notes": [
                    "V7.1 scene-level roles share logs; overlap remains recorded in data_role_inventory.json.",
                    "source_candidate_pool is metadata-only and is not a frozen test split.",
                ],
            },
            "av2_sensor_val": {
                "group_id_kind": "log_id",
                "group_roles": {
                    "legacy_diagnostic": sorted(consumed_external),
                    "external_candidate_pool": list(
                        external["available_unconsumed_on_disk_log_ids"]
                    ),
                    "external_test": [],
                },
                "frozen_roles": {"external_test": False},
                "official_val_log_count": 150,
                "unconsumed_official_log_count": 150 - len(consumed_external),
                "unconsumed_on_disk_count": len(
                    external["available_unconsumed_on_disk_log_ids"]
                ),
                "notes": [
                    "Only on-disk IDs are listed in external_candidate_pool.",
                    "The remaining official IDs must be listed metadata-only before any download or quality read.",
                ],
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=Path("/root/autodl-tmp/data/worldsim_v4/drivestudio_raw_trainval/v1.0-trainval"),
    )
    parser.add_argument(
        "--av2-root", type=Path, default=Path("/root/autodl-tmp/data/av2/sensor/val")
    )
    parser.add_argument(
        "--v71-corpus-root",
        type=Path,
        default=Path("/root/autodl-tmp/data/worldsim_v71/corpus_v1"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--roles-output", type=Path)
    args = parser.parse_args()
    inventory = build_inventory(
        args.repo_root.resolve(), args.metadata_root, args.av2_root, args.v71_corpus_root
    )
    rendered = json.dumps(inventory, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    if args.roles_output is not None:
        roles_rendered = (
            json.dumps(build_role_manifest(inventory), ensure_ascii=False, indent=2) + "\n"
        )
        args.roles_output.parent.mkdir(parents=True, exist_ok=True)
        temporary_roles = args.roles_output.with_suffix(args.roles_output.suffix + ".tmp")
        temporary_roles.write_text(roles_rendered, encoding="utf-8")
        temporary_roles.replace(args.roles_output)


if __name__ == "__main__":
    main()

"""Select 45 audit scenes / 70 clips and quarantine 25 final scenes.

Only metadata-derived target size, visibility, scene description, box overlap
and fixed pseudorandom order are used. No image or model output is inspected.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path


SEED = 770128
AREA_SMALL = 0.00361
AREA_LARGE = 0.032106
SCENE_SPARSE = 11


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    assert not path.exists(), f"Refusing to overwrite frozen selection: {path}"
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def scene_stratum(rows):
    if rows[0]["night_proxy"]:
        return "night"
    return "day_sparse" if statistics.median(r["vehicle_count"] for r in rows) <= SCENE_SPARSE else "day_dense"


def area_group(row):
    return "small" if row["area_fraction"] <= AREA_SMALL else "large" if row["area_fraction"] >= AREA_LARGE else "medium"


def camera_group(row):
    return "front" if row["camera"] == "CAM_FRONT" else "rear" if row["camera"] == "CAM_BACK" else "side"


def risk(row):
    flags = []
    if area_group(row) == "small":
        flags.append("small_target")
    if row["occlusion_proxy"] == "partial":
        flags.append("partial_visibility_GT")
    if row["near_vehicle_proxy"]:
        flags.append("near_vehicle_projection")
    if row["behind_vehicle_proxy"]:
        flags.append("behind_vehicle_projection")
    if row["night_proxy"]:
        flags.append("night_metadata")
    if row["vehicle_count"] > SCENE_SPARSE:
        flags.append("dense_GT_scene")
    level = "high" if len(flags) >= 3 or ("small_target" in flags and "night_metadata" in flags) else "medium" if flags else "low"
    return level, flags


def choose_target(rows, used_tracks, counts, desired, rng):
    valid = [r for r in rows if r["instance_token"] not in used_tracks]
    assert valid
    # Avoid an otherwise meaningless preference for one of many adjacent starts
    # of the same actor/camera/size/occlusion family.
    groups = {}
    for row in valid:
        key = (row["instance_token"], row["camera"], area_group(row), row["occlusion_proxy"], row["behind_vehicle_proxy"])
        groups.setdefault(key, []).append(row)
    pool = []
    for rows2 in groups.values():
        rows2.sort(key=lambda r: r["start_keyframe"])
        pool.append(rows2[len(rows2) // 2])
    rng.shuffle(pool)

    def score(row):
        features = [
            ("area", area_group(row), 2.0),
            ("camera", camera_group(row), 2.0),
            ("visibility", row["occlusion_proxy"], 1.0),
            ("behind", str(row["behind_vehicle_proxy"]), 1.0),
        ]
        value = 0.0
        for family, key, weight in features:
            target = desired[family][key]
            value += weight * (target - counts[family][key]) / max(1, target)
        return value

    selected = max(pool, key=score)
    for family, key in [
        ("area", area_group(selected)), ("camera", camera_group(selected)),
        ("visibility", selected["occlusion_proxy"]),
        ("behind", str(selected["behind_vehicle_proxy"])),
    ]:
        counts[family][key] += 1
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pool = read(args.pool)
    rng = random.Random(SEED)
    strata = {name: scene_stratum(rows) for name, rows in pool.items()}
    by_stratum = {key: sorted(name for name in pool if strata[name] == key) for key in ["night", "day_sparse", "day_dense"]}
    for names in by_stratum.values():
        rng.shuffle(names)
    audit_quota = {"night": 6, "day_sparse": 12, "day_dense": 27}
    final_quota = {"night": 4, "day_sparse": 6, "day_dense": 15}
    for key in audit_quota:
        assert len(by_stratum[key]) >= audit_quota[key] + final_quota[key], (key, len(by_stratum[key]))
    audit = [name for key, count in audit_quota.items() for name in by_stratum[key][:count]]
    final = [name for key, count in final_quota.items() for name in by_stratum[key][audit_quota[key]:audit_quota[key] + count]]
    rng.shuffle(audit)
    rng.shuffle(final)
    assert len(audit) == 45 and len(final) == 25 and len(set(audit + final)) == 70

    # 25 scenes contribute a second, different actor. The duplicate-scene rule
    # is fixed before seeing any RGB or model results.
    dual_scenes = []
    for name in audit:
        if len({r["instance_token"] for r in pool[name]}) >= 2:
            dual_scenes.append(name)
        if len(dual_scenes) == 25:
            break
    assert len(dual_scenes) == 25
    desired = {
        "area": {"small": 22, "medium": 26, "large": 22},
        "camera": {"front": 24, "side": 24, "rear": 22},
        "visibility": {"clear": 35, "partial": 35},
        "behind": {"True": 26, "False": 44},
    }
    counts = {family: Counter() for family in desired}
    used = {name: set() for name in audit}
    selected = []
    for pass_number in [1, 2]:
        for scene in audit:
            if pass_number == 2 and scene not in dual_scenes:
                continue
            row = choose_target(pool[scene], used[scene], counts, desired, rng)
            used[scene].add(row["instance_token"])
            level, flags = risk(row)
            selected.append(dict(
                row, clip_id=f"A{len(selected) + 1:03d}", actor_ordinal_in_scene=pass_number,
                size_bucket=area_group(row), camera_bucket=camera_group(row),
                input_difficulty_proxy=level, difficulty_factors=flags,
                classification_source="official_GT_and_scene_description_only; one-frame visual review pending",
                task="single_actor_DELETE", human_verdict=None,
            ))
    assert len(selected) == 70 and len({(x["scene"], x["instance_token"]) for x in selected}) == 70
    record = dict(
        task_id="WS-V77-DELETE-AUDIT-20260928", run_id="r1",
        sample_role="nuScenes official val problem-audit; all 9 previous development scenes excluded",
        official_val_scene_count=150, audit_scene_count=45, audit_clip_count=70,
        final_quarantine_scene_count=25, scene_seed=SEED,
        audit_scene_quota=audit_quota, final_scene_quota=final_quota,
        target_quota=desired, area_fraction_thresholds=[AREA_SMALL, AREA_LARGE],
        sparse_scene_gt_vehicle_threshold=SCENE_SPARSE,
        sample_rule="Every third 2Hz keyframe start, 6-keyframe ~3s window; >=5 GT presences, >=4 visibility>=2, >=12x9px anchor projection. Fixed quotas and pseudorandom order; one actor per clip, at most two per scene.",
        original_nine_exposed_scene_names=sorted({"scene-0230", "scene-0255", "scene-0061", "scene-0436", "scene-0875", "scene-0242", "scene-0535", "scene-0471", "scene-0998"}),
        audit_scenes=audit, final_quarantined_scenes=final,
        clips=selected,
        achieved_target_buckets={k: dict(v) for k, v in counts.items()},
        human_verdict=None, failure_ledger_refs=["V77-F02"],
    )
    write(args.out, record)
    print(json.dumps({"audit_scenes": len(audit), "audit_clips": len(selected), "final_quarantine": len(final),
                      "achieved_buckets": record["achieved_target_buckets"], "night_clips": sum(x["night_proxy"] for x in selected)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

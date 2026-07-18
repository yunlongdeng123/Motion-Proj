from motion_proj.data.motion_feature_records import split_fingerprint, stable_scene_split


def test_probe_split_is_scene_disjoint_and_stable():
    records = []
    for scene in range(12):
        for start in (8, 0):
            records.append(
                {
                    "scene_name": f"scene-{scene:03d}",
                    "scene_token": f"token-{scene:03d}",
                    "start_index": start,
                    "sample_id": f"sample-{scene:03d}-{start}",
                }
            )
    first = stable_scene_split(records, train_count=6, dev_count=3, holdout_count=3)
    second = stable_scene_split(list(reversed(records)), train_count=6, dev_count=3, holdout_count=3)
    assert split_fingerprint(first) == split_fingerprint(second)
    scene_sets = [{row["scene_token"] for row in first[name]} for name in ("train", "dev", "holdout")]
    assert not (scene_sets[0] & scene_sets[1] or scene_sets[0] & scene_sets[2] or scene_sets[1] & scene_sets[2])
    assert all(row["start_index"] == 0 for rows in first.values() for row in rows)


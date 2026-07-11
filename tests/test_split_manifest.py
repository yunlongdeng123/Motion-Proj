from types import SimpleNamespace

from motion_proj.data.nuscenes_dataset import (
    NuScenesFutureVideoDataset,
    official_scene_names,
)
from motion_proj.data.split_manifest import build_split_manifest, summarize_manifest


class _FakeNuScenes:
    def __init__(self, scenes, samples):
        self.scene = scenes
        self._samples = samples

    def get(self, table, token):
        assert table == "sample"
        return self._samples[token]


def test_official_train_val_scenes_are_disjoint():
    train = official_scene_names("v1.0-trainval", "train")
    val = official_scene_names("v1.0-trainval", "val")

    assert train is not None and len(train) == 700
    assert val is not None and len(val) == 150
    assert train.isdisjoint(val)


def test_dataset_builds_only_selected_scenes():
    scene_names = sorted(official_scene_names("v1.0-mini", "mini_train") or [])
    scenes = []
    samples = {}
    for scene_index, name in enumerate(scene_names):
        first = f"sample-{scene_index}-0"
        second = f"sample-{scene_index}-1"
        scenes.append(
            {
                "name": name,
                "token": f"scene-token-{scene_index}",
                "first_sample_token": first,
            }
        )
        samples[first] = {"next": second}
        samples[second] = {"next": ""}

    dataset = NuScenesFutureVideoDataset.__new__(NuScenesFutureVideoDataset)
    dataset.nusc = _FakeNuScenes(scenes, samples)
    dataset.version = "v1.0-mini"
    dataset.split = "mini_train"
    dataset.camera = "CAM_FRONT"
    dataset.K = 2
    dataset.stride = 1

    clips = dataset._build_clips()

    assert len(clips) == 8
    assert dataset.scene_names == scene_names
    assert {row["scene_name"] for row in dataset.clip_records} == set(scene_names)


def test_split_manifest_has_stable_fingerprint_and_count_checks(tmp_path):
    metadata_root = tmp_path / "v1.0-mini"
    metadata_root.mkdir()
    (metadata_root / "scene.json").write_text("[]\n", encoding="utf-8")
    (metadata_root / "sample.json").write_text("[]\n", encoding="utf-8")
    scene = {
        "name": "scene-0001",
        "token": "scene-token",
        "nbr_samples": 2,
    }
    dataset = SimpleNamespace(
        clip_records=[
            {
                "scene_name": "scene-0001",
                "scene_token": "scene-token",
                "start_index": 0,
                "sample_tokens": ["sample-0", "sample-1"],
                "sample_id": "sample-0_CAM_FRONT",
            }
        ],
        nusc=SimpleNamespace(scene=[scene]),
        scene_names=["scene-0001"],
        dataroot=str(tmp_path),
        version="v1.0-mini",
        split="mini_train",
        camera="CAM_FRONT",
        K=2,
        stride=1,
    )

    first = build_split_manifest(dataset)
    second = build_split_manifest(dataset)
    summary = summarize_manifest(
        first,
        {"expected_scene_count": 1, "expected_clip_count": 1},
    )

    assert first["split_fingerprint"] == second["split_fingerprint"]
    assert first["scene_count"] == 1
    assert first["clip_count"] == 1
    assert summary["accepted"]

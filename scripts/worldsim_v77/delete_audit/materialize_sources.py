"""Resolve the frozen nuScenes audit clips to exact public RGB archive members."""
from __future__ import annotations

import argparse
import bisect
import json
from collections import defaultdict
from pathlib import Path

import ijson
from nuscenes.utils.splits import val


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    assert not path.exists(), f"Refusing to overwrite frozen source manifest: {path}"
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    meta = args.metadata
    selection = read(args.selection)
    scenes = {s["name"]: s for s in read(meta / "scene.json")}
    samples = {s["token"]: s for s in read(meta / "sample.json")}
    selected = set(selection["audit_scenes"])
    assert len(selected) == 45 and selected.issubset(val)
    channels = {s["token"]: s["channel"] for s in read(meta / "sensor.json")}
    cal = {s["token"]: s["sensor_token"] for s in read(meta / "calibrated_sensor.json")}
    scene_by_token = {scenes[name]["token"]: name for name in selected}
    sample_to_scene = {token: scene_by_token[row["scene_token"]]
                       for token, row in samples.items() if row["scene_token"] in scene_by_token}
    needed_cameras = defaultdict(set)
    for clip in selection["clips"]:
        needed_cameras[clip["scene"]].add(clip["camera"])
    camera_rows = defaultdict(list)
    with (meta / "sample_data.json").open("rb") as handle:
        for row in ijson.items(handle, "item", use_float=True):
            scene = sample_to_scene.get(row["sample_token"])
            if scene is None:
                continue
            channel = channels[cal[row["calibrated_sensor_token"]]]
            if channel in needed_cameras[scene]:
                camera_rows[(scene, channel)].append({
                    "filename": row["filename"], "timestamp": row["timestamp"],
                    "width": row["width"], "height": row["height"],
                    "sample_data_token": row["token"],
                })
    for rows in camera_rows.values():
        rows.sort(key=lambda x: x["timestamp"])
    print("CAMERA_STREAMS", len(camera_rows), flush=True)
    output = []
    missing = []
    all_files = set()
    for clip in selection["clips"]:
        scene = scenes[clip["scene"]]
        ordered = []
        token = scene["first_sample_token"]
        while token:
            ordered.append(samples[token])
            token = samples[token]["next"]
        start = clip["start_keyframe"]
        keyframes = ordered[start:start + 6]
        assert len(keyframes) == 6
        target_times = []
        for before, after in zip(keyframes[:-1], keyframes[1:]):
            t0, t1 = before["timestamp"], after["timestamp"]
            for j in range(5):
                target_times.append(int(round(t0 + j * (t1 - t0) / 5)))
        target_times.append(keyframes[-1]["timestamp"])
        assert len(target_times) == 26 and all(b > a for a, b in zip(target_times, target_times[1:]))
        rows = camera_rows[(clip["scene"], clip["camera"])]
        times = [r["timestamp"] for r in rows]
        frames = []
        for i, wanted in enumerate(target_times):
            ix = bisect.bisect_left(times, wanted)
            candidates = [rows[j] for j in (ix - 1, ix) if 0 <= j < len(rows)]
            if not candidates:
                missing.append(dict(clip_id=clip["clip_id"], frame=i, reason="no_camera_frame"))
                continue
            source = min(candidates, key=lambda x: (abs(x["timestamp"] - wanted), x["timestamp"]))
            delta_ms = abs(source["timestamp"] - wanted) / 1000
            if delta_ms > 65:
                missing.append(dict(clip_id=clip["clip_id"], frame=i, reason="nearest_camera_gap_gt65ms", delta_ms=delta_ms))
            frames.append(dict(frame=i, target_timestamp_us=wanted, source_timestamp_us=source["timestamp"],
                               delta_ms=round(delta_ms, 3), filename=source["filename"],
                               width=source["width"], height=source["height"],
                               sample_data_token=source["sample_data_token"]))
            all_files.add(source["filename"])
        output.append(dict(clip_id=clip["clip_id"], scene=clip["scene"], camera=clip["camera"],
                           actor_instance_token=clip["instance_token"], start_keyframe=start,
                           keyframe_sample_tokens=[r["token"] for r in keyframes],
                           frames=frames, unique_rgb_files=len({x["filename"] for x in frames})))
        if len(output) % 10 == 0:
            print("CLIPS_RESOLVED", len(output), flush=True)
    assert len(output) == 70
    result = dict(task_id=selection["task_id"], run_id=selection["run_id"],
                  source="official nuScenes trainval metadata; nearest actual camera frame for each 10Hz target timestamp",
                  clip_count=70, frame_count=sum(len(c["frames"]) for c in output),
                  unique_rgb_file_count=len(all_files), missing_or_large_gap=missing,
                  clips=output, human_verdict=None)
    write(args.out, result)
    print("SOURCES", result["clip_count"], result["frame_count"], result["unique_rgb_file_count"],
          "GAPS", len(missing), flush=True)


if __name__ == "__main__":
    main()

"""为全部case补充同窗口nuScenes事实RGB参考视频，不插帧、不做生成。"""
import argparse
import json
from pathlib import Path
import shutil
import time

import av
import numpy as np
from PIL import Image


def make_original(case, camera, out):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "original-nuscenes.mp4"
    source = Path(case["dataset"]["root"]) / case["dataset"]["scene_id"]
    start = case["anchor"]["event_frame"]-case["anchor"]["pre_frames"]
    end = case["anchor"]["event_frame"]+case["anchor"]["rollout_frames"]-1
    frames = list(range(start, end+1))
    assert len(frames) == 24
    record = {"role": "observed_factual_reference_not_generated", "dataset": "nuScenes DriveStudio-processed10Hz RGB",
              "case_id": case["case_id"], "scene_id": case["dataset"]["scene_id"], "camera_index": camera,
              "source_root": str(source), "source_frames": frames, "source_images": [f"images/{f:03d}_{camera}.jpg" for f in frames],
              "fps": 10, "frame_count": 24, "resolution": [1600, 900], "first_frame_time_s": 0,
              "last_frame_time_s": 2.3, "intervention_time_s": .5,
              "interpolation": "none; original processed dataset frames in temporal order", "encoding": "H264 CRF18 yuv420p",
              "note": "参考事实观测；反事实不存在配对真实GT。编码不是逐像素无损，原始JPEG仍在source_images。"}
    provenance = out / "original-nuscenes.json"
    if path.exists():
        assert provenance.exists() and json.loads(provenance.read_text()) == record, path
    else:
        with av.open(str(path), "w") as writer:
            stream = writer.add_stream("libx264", rate=10)
            stream.width, stream.height, stream.pix_fmt = 1600, 900, "yuv420p"
            stream.options = {"crf": "18"}
            for f in frames:
                image = np.array(Image.open(source / f"images/{f:03d}_{camera}.jpg").convert("RGB"))
                assert image.shape == (900, 1600, 3)
                for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                    writer.mux(packet)
            for packet in stream.encode():
                writer.mux(packet)
        provenance.write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n")
    with av.open(str(path)) as reader:
        assert sum(1 for _ in reader.decode(video=0)) == 24
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--wait-for-case-dir", action="store_true")
    args = p.parse_args()
    index = json.loads((args.inputs / "index.json").read_text())
    for row in index["cases"]:
        if "case" not in row:
            continue
        if args.wait_for_case_dir:
            while not (args.run / row["case_id"]).exists():
                terminal = args.run / "queue-result.json"
                if terminal.exists() and json.loads(terminal.read_text())["status"] == "failed_stopped":
                    raise RuntimeError("推理队列失败，停止等待case目录")
                time.sleep(10)
        record = make_original(row["case"], row.get("camera_index", 0), args.run / row["case_id"])
        print(json.dumps({"original_video_added": record["case_id"], "frames": 24, "camera": record["camera_index"]}), flush=True)


if __name__ == "__main__":
    main()

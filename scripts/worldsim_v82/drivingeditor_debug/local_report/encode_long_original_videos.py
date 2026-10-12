"""Encode canonical B1 true49 PNGs with CPU libx264; decode every MP4 frame.

New long-window assets only. Existing30300 short videos/receipts are untouched.
"""
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
RUN = HERE / "continuation_diagnostic/original30k_to32000"
OUT = HERE.parents[1] / "outputs/drivingeditor_vgpu_debug"
ASSETS = OUT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
IDS = ["image_0/0017" + str(i).zfill(3) for i in range(49)]
CHANNELS = ("gt", "composite", "static", "dynamic")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    summary = json.loads((HERE / "long_original_summary_cpu.json").read_bytes())
    assert summary["verified_PNG_count"] == 294 and summary["case"] == RUN.name
    canonical = {(r["frame_id"], r["channel"]): r["sha256"] for r in summary["verified_PNG_sha256"]}
    inputs = {}
    for channel in CHANNELS:
        for frame_id in IDS:
            path = RUN / "extra_eval/32000" / channel / (Path(frame_id).name + ".png")
            assert sha(path) == canonical[(frame_id, channel)]
            inputs[str(path.relative_to(HERE)).replace("\\", "/")] = sha(path)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
    records = []
    for channel in (*CHANNELS, "four_panel"):
        mosaic = channel == "four_panel"
        width, height = (1920, 352) if mosaic else (480, 320)
        destination = ASSETS / ("B1_original32000_" + channel + "_actual49.mp4")
        temp = destination.with_suffix(".part")
        relevant = inputs if mosaic else {p: h for p, h in inputs.items() if "/" + channel + "/" in p}
        with av.open(str(temp), "w", format="mp4", options={"movflags": "+faststart"}) as container:
            stream = container.add_stream("libx264", rate=10)
            stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
            stream.codec_context.thread_count = 4
            stream.options = {"crf": "18", "preset": "medium"}
            for index, frame_id in enumerate(IDS):
                if mosaic:
                    canvas = Image.new("RGB", (width, height), (8, 14, 23))
                    draw = ImageDraw.Draw(canvas)
                    for col, name in enumerate(CHANNELS):
                        path = RUN / "extra_eval/32000" / name / (Path(frame_id).name + ".png")
                        with Image.open(path) as source:
                            assert source.size == (480, 320)
                            canvas.paste(source.convert("RGB"), (480 * col, 32))
                        draw.text((480 * col + 8, 3), "B1 @32000 / " + name, font=font, fill=(225, 235, 246))
                    values = np.asarray(canvas)
                else:
                    path = RUN / "extra_eval/32000" / channel / (Path(frame_id).name + ".png")
                    with Image.open(path) as source:
                        values = np.asarray(source.convert("RGB"))
                        assert values.shape == (320, 480, 3)
                frame = av.VideoFrame.from_ndarray(values, format="rgb24")
                frame.pts, frame.time_base = index, Fraction(1, 10)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        temp.replace(destination)
        pts = []
        with av.open(str(destination)) as container:
            assert container.streams.video[0].codec_context.name == "h264"
            assert float(container.streams.video[0].average_rate) == 10
            for frame in container.decode(video=0):
                assert (frame.width, frame.height) == (width, height)
                decoded = frame.to_ndarray(format="rgb24")
                pts.append(float(frame.pts * frame.time_base))
                if mosaic and len(pts) == 11:
                    Image.fromarray(decoded).save(ASSETS / "B1_original32000_four_panel_decoded_raw10.png")
        assert len(pts) == 49 and all(abs(t - i / 10) < 1e-9 for i, t in enumerate(pts))
        records.append({
            "case": RUN.name, "arm": "B1", "iteration": 32000, "channel": channel,
            "file": destination.name, "sha256": sha(destination), "bytes": destination.stat().st_size,
            "frame_ids": IDS, "frame_count": 49, "fps": 10, "duration_seconds": 4.9,
            "dimensions": [width, height], "pts_seconds": pts, "all_frames_decoded": True,
            "same_camera_panels_encoded_in_one_frame": mosaic,
            "input_PNG_sha256": relevant, "CPU_encoder": "libx264, four threads, CRF18, no hardware encoder",
            "metrics_computed_from_float32_region_receipts_not_lossy_MP4": True,
            "browser_playback_checked": False,
        })
        print(json.dumps({key: records[-1][key] for key in ("file", "frame_count", "dimensions", "all_frames_decoded")}))
    receipt = {
        "status": "B1_LONG_TRUE49_MP4_ALL245_FRAMES_DECODED",
        "videos": records, "video_count": 5, "decoded_frame_count": 245,
        "summary_cpu_sha256": sha(HERE / "long_original_summary_cpu.json"),
        "short30300_media_modified": False, "GPU_used": False, "browser_playback_checked": False,
        "encoder_script_sha256": sha(__file__)
    }
    path = OUT / "long_B1_video_validation.json"
    path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "video_count": 5, "receipt_sha256": sha(path)}))


if __name__ == "__main__":
    main()

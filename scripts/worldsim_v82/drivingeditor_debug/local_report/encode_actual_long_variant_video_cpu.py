"""Encode one new true49 V four-panel MP4 and decode its49 frames, CPU only."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
RUN = HERE / 'continuation_diagnostic/opacity_outside30k_to32000'
OUT = HERE.parents[1] / 'outputs/drivingeditor_vgpu_debug'
CHANNELS = ('gt', 'composite', 'static', 'dynamic')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = RUN / 'receipts/full49_summary.json'
    assert sha(source) == '9e57f90a2e7e9a95d7087dfd37aef6e220b78260a9e5d304523f70d8a4d61e20'
    summary = json.loads(source.read_bytes())
    assert summary['harness_exit_code'] == summary['wrapper_exit_code'] == 0
    pins = {(r['frame_id'], r['channel']): r for r in summary['artifact_pins'] if 'channel' in r}
    frames = ['image_0/0017' + str(i).zfill(3) for i in range(49)]
    pngs = []
    for frame_id in frames:
        for channel in CHANNELS:
            path = RUN / 'extra_eval/32000' / channel / (Path(frame_id).name + '.png')
            assert sha(path) == pins[(frame_id, channel)]['sha256']
            pngs.append({'frame_id': frame_id, 'channel': channel, 'path': str(path), 'sha256': sha(path)})
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 20)
    destination = OUT / 'assets/V_opacity_outside32000_four_panel_actual49.mp4'
    temp = destination.with_suffix('.part')
    with av.open(str(temp), 'w', format='mp4', options={'movflags': '+faststart'}) as container:
        stream = container.add_stream('libx264', rate=10)
        stream.width, stream.height, stream.pix_fmt = 1920, 352, 'yuv420p'
        stream.codec_context.thread_count = 4
        stream.options = {'crf': '18', 'preset': 'medium'}
        for index, frame_id in enumerate(frames):
            canvas = Image.new('RGB', (1920, 352), (8, 14, 23))
            draw = ImageDraw.Draw(canvas)
            for col, channel in enumerate(CHANNELS):
                path = RUN / 'extra_eval/32000' / channel / (Path(frame_id).name + '.png')
                with Image.open(path) as image:
                    assert image.size == (480, 320)
                    canvas.paste(image.convert('RGB'), (480 * col, 32))
                draw.text((480 * col + 8, 3), 'V @32000 / ' + channel + ' / raw' + str(index), font=font, fill=(225, 235, 246))
            frame = av.VideoFrame.from_ndarray(np.asarray(canvas), format='rgb24')
            frame.pts, frame.time_base = index, Fraction(1, 10)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    temp.replace(destination)
    pts = []
    with av.open(str(destination)) as container:
        assert container.streams.video[0].codec_context.name == 'h264'
        assert float(container.streams.video[0].average_rate) == 10
        for frame in container.decode(video=0):
            assert (frame.width, frame.height) == (1920, 352)
            decoded = frame.to_ndarray(format='rgb24')
            pts.append(float(frame.pts * frame.time_base))
            if len(pts) == 11:
                Image.fromarray(decoded).save(OUT / 'assets/V_opacity_outside32000_four_panel_decoded_raw10.png')
    assert len(pts) == 49 and all(abs(t - i / 10) < 1e-9 for i, t in enumerate(pts))
    receipt = {'status': 'V_TRUE49_FOUR_PANEL_MP4_ALL49_DECODED', 'case': RUN.name, 'iteration': 32000,
               'file': destination.name, 'sha256': sha(destination), 'bytes': destination.stat().st_size,
               'frame_ids': frames, 'frame_count': 49, 'fps': 10, 'duration_seconds': 4.9,
               'dimensions': [1920, 352], 'pts_seconds': pts, 'all_frames_decoded': True,
               'input196_PNG_sha256': pngs, 'actual_summary_sha256': sha(source),
               'same_camera_panels_encoded_in_one_frame': True,
               'CPU_encoder': 'libx264 four threads CRF18; no hardware encoder',
               'original_B1_or_short30300_videos_reencoded': False,
               'GPU_used': False, 'browser_playback_checked': False,
               'metrics_come_from_float32_receipts_not_lossy_video': True,
               'script_sha256': sha(Path(__file__))}
    path = OUT / 'long_V_video_validation.json'
    path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps({'status': receipt['status'], 'sha256': sha(path), 'frame_count': 49, 'video_sha256': receipt['sha256']}))


if __name__ == '__main__':
    main()

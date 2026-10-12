"""Create a new actual B1/B2/S synchronous49 video, no existing video edits."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / 'outputs/drivingeditor_vgpu_debug'
CHANNELS = ('gt', 'composite', 'static', 'dynamic')
ARMS = [('B1', 'original30k_to32000', '36879999ab39e208b0cb031e7557c83a3dabc11f32e361687796b8e641eced31'),
        ('B2', 'original30k_to32000_noise_reference', '2ded7dda0952e0bf8605154a642576049ce45b41e8c1df1ddda3bec62085c92e'),
        ('S', 'observed_static_rgb30k_to32000', '79f1cf9be8ecb2fe43ea377dee11c799bb8581bca65e41241c614f09c090db55')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    frames = ['image_0/0017' + str(i).zfill(3) for i in range(49)]
    pngs = []
    for arm, run, expected in ARMS:
        root = HERE / 'continuation_diagnostic' / run
        source = root / 'receipts/full49_summary.json'
        assert sha(source) == expected
        summary = json.loads(source.read_bytes())
        assert summary['harness_exit_code'] == summary['wrapper_exit_code'] == 0
        pins = {(r['frame_id'], r['channel']): r for r in summary['artifact_pins'] if 'channel' in r}
        for frame_id in frames:
            for channel in CHANNELS:
                path = root / 'extra_eval/32000' / channel / (Path(frame_id).name + '.png')
                assert sha(path) == pins[(frame_id, channel)]['sha256']
                pngs.append({'arm': arm, 'frame_id': frame_id, 'channel': channel, 'path': str(path), 'sha256': sha(path)})
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 20)
    destination = OUT / 'assets/B1_B2_S32000_synchronous_actual49.mp4'
    temp = destination.with_suffix('.part')
    with av.open(str(temp), 'w', format='mp4', options={'movflags': '+faststart'}) as container:
        stream = container.add_stream('libx264', rate=10)
        stream.width, stream.height, stream.pix_fmt = 1920, 1056, 'yuv420p'
        stream.codec_context.thread_count = 4
        stream.options = {'crf': '18', 'preset': 'medium'}
        for index, frame_id in enumerate(frames):
            canvas = Image.new('RGB', (1920, 1056), (8, 14, 23))
            draw = ImageDraw.Draw(canvas)
            for row, (arm, run, _) in enumerate(ARMS):
                for col, channel in enumerate(CHANNELS):
                    path = HERE / 'continuation_diagnostic' / run / 'extra_eval/32000' / channel / (Path(frame_id).name + '.png')
                    with Image.open(path) as image:
                        assert image.size == (480, 320)
                        canvas.paste(image.convert('RGB'), (480 * col, 352 * row + 32))
                    draw.text((480 * col + 8, 352 * row + 3), arm + ' @32000 / ' + channel + ' / raw' + str(index), font=font, fill=(225, 235, 246))
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
            assert (frame.width, frame.height) == (1920, 1056)
            decoded = frame.to_ndarray(format='rgb24')
            pts.append(float(frame.pts * frame.time_base))
            if len(pts) == 11:
                Image.fromarray(decoded).save(OUT / 'assets/B1_B2_S32000_synchronous_decoded_raw10.png')
    assert len(pts) == 49 and all(abs(t - i / 10) < 1e-9 for i, t in enumerate(pts))
    result = {'status': 'ACTUAL_B1_B2_S_SYNCHRONOUS49_MP4_ALL49_DECODED',
              'file': destination.name, 'sha256': sha(destination), 'bytes': destination.stat().st_size,
              'frame_count': 49, 'frame_ids': frames, 'fps': 10, 'duration_seconds': 4.9,
              'dimensions': [1920, 1056], 'pts_seconds': pts, 'all_frames_decoded': True,
              'row_order': ['B1', 'B2', 'S'], 'column_order': list(CHANNELS),
              'actual588_source_PNG_sha256': pngs,
              'actual_arm_summary_sha256': {arm: expected for arm, run, expected in ARMS},
              'same_camera_and_endpoint_all_panels_one_video_frame': True,
              'short30300_not_mixed': True, 'old_B1_or_short_videos_reencoded': False,
              'CPU_encoder': 'libx264 four threads CRF18, no hardware encoder',
              'GPU_used': False, 'browser_playback_checked': False,
              'metrics_from_actual_float32_receipts_not_lossy_video': True,
              'encoder_script_sha256': sha(Path(__file__))}
    path = OUT / 'long_S_video_validation.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps({'status': result['status'], 'sha256': sha(path), 'video_sha256': result['sha256']}))


if __name__ == '__main__':
    main()

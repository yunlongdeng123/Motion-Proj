"""解码生成视频，保存固定时刻预览；检查文件完整性，不代替科学评价。"""
import argparse
import json
from pathlib import Path
import av
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('video', type=Path)
p.add_argument('--expected-frames', type=int, required=True)
a = p.parse_args()
indices = sorted(set([0, 4, a.expected_frames // 4, a.expected_frames // 2,
                      3 * a.expected_frames // 4, a.expected_frames - 1]))
rows = []
with av.open(str(a.video)) as container:
    stream = container.streams.video[0]
    rate = float(stream.average_rate)
    for i, frame in enumerate(container.decode(stream)):
        rgb = frame.to_ndarray(format='rgb24')
        rows.append({'frame': i, 'pts_s': float(frame.pts * frame.time_base),
                     'mean': float(rgb.mean()), 'std': float(rgb.std()),
                     'min': int(rgb.min()), 'max': int(rgb.max())})
        assert rgb.shape == (704, 1280, 3)
        if i in indices:
            frame.to_image().save(a.video.parent / f'{a.video.stem}-frame-{i:03d}.png')
assert len(rows) == a.expected_frames, (len(rows), a.expected_frames)
assert rate == 30
assert all(rows[i + 1]['pts_s'] > rows[i]['pts_s'] for i in range(len(rows) - 1))
result = {'video': str(a.video), 'status': 'decode_passed', 'frames': len(rows),
          'width': 1280, 'height': 704, 'fps': rate, 'duration_s': len(rows) / rate,
          'minimum_frame_std': min(r['std'] for r in rows), 'frames_summary': rows,
          'sample_frames': indices, 'human_verdict': None}
a.video.with_suffix('.validation.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'frames_summary'}))

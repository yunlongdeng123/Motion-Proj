"""逐像素比较解码结果；不使用哈希，也不将相同 seed 当作确定性证据。"""
import argparse
import json
from itertools import zip_longest
from pathlib import Path
import av
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('first', type=Path)
p.add_argument('second', type=Path)
p.add_argument('--frames', type=int, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
rows = []
with av.open(str(a.first)) as first, av.open(str(a.second)) as second:
    for i, pair in enumerate(zip_longest(first.decode(video=0), second.decode(video=0))):
        if i >= a.frames:
            break
        left, right = pair
        assert left is not None and right is not None
        x = left.to_ndarray(format='rgb24').astype(np.int16)
        y = right.to_ndarray(format='rgb24').astype(np.int16)
        delta = np.abs(x - y)
        rows.append({'frame': i, 'mae_8bit': float(delta.mean()),
                     'max_abs_8bit': int(delta.max()),
                     'different_channel_values': int(np.count_nonzero(delta))})
assert len(rows) == a.frames
result = {'first': str(a.first), 'second': str(a.second), 'compared_frames': len(rows),
          'decoded_frames_exactly_equal': all(r['max_abs_8bit'] == 0 for r in rows),
          'mean_mae_8bit': float(np.mean([r['mae_8bit'] for r in rows])),
          'max_abs_8bit': max(r['max_abs_8bit'] for r in rows), 'frames_summary': rows,
          'scope': 'decoded uint8 RGB only; raw floating-point tensors were not retained'}
a.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'frames_summary'}))

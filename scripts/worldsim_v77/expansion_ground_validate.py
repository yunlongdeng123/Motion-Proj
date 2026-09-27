"""核对r7只改变固定z、相机投影与mesh放置，不重新生成或渲染。"""
from pathlib import Path
import sys, copy, numpy as np
sys.path.insert(0, '/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from repair_common import read, dump

T = Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPAND-EDIT-20260928')
reg = read(T / 'r7/registration.json')
a, b = read(T / 'r5/camera_frames.json'), read(T / 'r7/camera_frames.json')
expected = copy.deepcopy(a)
for frame in expected:
    actor = next(x for x in frame['all_boxes'] if x['actor_id'] == '12')
    actor['pose'][2][3] += reg['delta_z_m']
assert b == expected
layer = read(T / 'r7/actor_layers/render_summary.json')
placement = read(T / 'r7/actor_layers/placement_checks.json')
assert len(layer['rows']) == len(placement) == 11
assert max(x['projection_max_error_px'] for x in layer['rows']) < .02
assert max(x['center_error_m'] for x in placement) < .002
assert max(x['size_error_m'] for x in placement) < .002
assert (T / 'r5/actor.glb').read_bytes() == (T / 'r7/actor.glb').read_bytes()
for row in layer['rows']:
    depth = np.load(T / 'r7/actor_layers' / f"f{row['frame']:03}_cam{row['camera']}_depth.npy")
    assert depth.shape == (384, 688) and np.isfinite(depth).any()
report = dict(fixed_z_only=True, delta_z_m=reg['delta_z_m'], camera_frames=len(b),
              layer_renders=11, same_asset_bytes=True, move_executed=False,
              projection_max_px=max(x['projection_max_error_px'] for x in layer['rows']),
              center_error_max_m=max(x['center_error_m'] for x in placement),
              size_error_max_m=max(x['size_error_m'] for x in placement), human_verdict=None)
dump(T / 'r7/validation.json', report)
base = read(T / 'validation.json')
base['ground_control'] = report
dump(T / 'validation.json', base)
print('GROUND_VALIDATED', report)

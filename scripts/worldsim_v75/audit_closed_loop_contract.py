"""真实AV2场景上的有限动作→相机→条件检查；不是自动策略闭环实验。"""
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw
import torch
from closed_loop_bridge import FeedbackBridge, DriverCommand

SOURCE = Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-VISIBLE-DEV-01/20260920-r1')
CASE = SOURCE/'cases/24642607-2a51-384a-90a7-228067956d05'
OUT = Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-CLOSEDLOOP-CONTRACT-01/20260920-r1')


def main():
    assert not OUT.exists()
    OUT.mkdir(parents=True)
    protocol = {'task_id': 'WS-V75-CLOSEDLOOP-CONTRACT-01', 'run_id': '20260920-r1',
                'frozen_utc': datetime.now(timezone.utc).isoformat(), 'base': str(CASE/'base'),
                'role': 'engineering_development', 'source_selection': 'reuse completed first visible case; no new scientific selection',
                'branches': ['coast', 'brake', 'left_steer'], 'frames_per_branch': 13,
                'feedback_policy': False, 'world_model_generation_calls': 0,
                'boundary': 'command/camera/render contract only; manual commands are not perception policies',
                'failure_ledger_delta': 'none', 'human_verdict': None}
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    result = {'status': 'started', 'human_verdict': None}; started = time.monotonic()
    try:
        true = json.loads((CASE/'base/scene.json').read_text())
        estimated = json.loads((CASE/'rollouts/dvgt_metric/scene.json').read_text())
        original = copy.deepcopy(true)
        assert true['lines'] == estimated['lines'] and true['crossings'] == estimated['crossings']
        assert sum(a != b for a, b in zip(true['tracks'], estimated['tracks'])) == 1
        data = {}
        for name, command in [('coast', DriverCommand()), ('brake', DriverCommand(brake=1)),
                              ('left_steer', DriverCommand(steer=.2, steer_is_direct=True))]:
            bridge = FeedbackBridge(CASE/'base', estimated)
            first = bridge.next_chunk(DriverCommand(), 5)
            nxt = bridge.next_chunk(command, 8)
            assert np.max(abs(first['camera_world'][0]-bridge.trajectory['camera_world'][0])) < 1e-5
            assert np.array_equal(np.r_[first['indices'], nxt['indices']], np.arange(13))
            expected = bridge.trajectory['timestamps_us'][:13]
            assert np.array_equal(np.r_[first['timestamps_us'], nxt['timestamps_us']], expected)
            poses = np.r_[first['camera_world'], nxt['camera_world']]
            conditions = np.r_[first['conditions'], nxt['conditions']]
            data[name] = {'poses': poses, 'conditions': conditions, 'speed': bridge.state.speed_mps,
                          'state': asdict(bridge.state), 'source': bridge.speed_source}
            np.savez(OUT/f'{name}.npz', camera_world=poses, timestamps_us=expected)
            Image.fromarray(conditions[-1]).save(OUT/f'{name}-condition-012.png')
            del bridge; gc.collect(); torch.cuda.empty_cache()
        coast = data['coast']
        changes = {}
        for name in ['brake', 'left_steer']:
            row = data[name]
            assert np.array_equal(row['conditions'][:5], coast['conditions'][:5])
            assert np.array_equal(row['poses'][:5], coast['poses'][:5])
            moved = float(np.linalg.norm(row['poses'][-1, :3, 3]-coast['poses'][-1, :3, 3]))
            pixels = int(np.any(row['conditions'][-1] != coast['conditions'][-1], axis=-1).sum())
            assert moved > 0 and pixels > 0
            changes[name] = {'end_camera_translation_delta_m': moved, 'changed_condition_pixels': pixels,
                             'end_speed_mps': row['speed'], 'shared_prefix_identical': True}
        assert data['brake']['speed'] < coast['speed']
        assert true == original == json.loads((CASE/'base/scene.json').read_text())
        sheet = Image.new('RGB', (1280, 2*380), '#122335'); draw = ImageDraw.Draw(sheet)
        cells = [('Real initial RGB', Image.open(CASE/'base/initial_rgb.png'))] + [
            (f'{name}: condition at frame12 (not generated RGB)', Image.open(OUT/f'{name}-condition-012.png')) for name in data]
        for i, (label, image) in enumerate(cells):
            x, y = (i % 2)*640, (i//2)*380
            draw.text((x+8, y+8), label, fill='white'); sheet.paste(image.resize((640, 352)), (x, y+27))
        sheet.save(OUT/'contract-review.jpg', quality=94)
        result.update(status='passed', checks=changes, initial_speed_source=coast['source'],
                      coast_end_speed_mps=coast['speed'], fixed_reference_unchanged=True,
                      raster_frames=39, world_model_generation_calls=0, policy_feedback_tested=False,
                      scope='flat-ground non-interactive engineering adapter; actual policy/OmniDreams feedback remains unexecuted')
    except BaseException as exc:
        result.update(status='oom_stopped' if isinstance(exc, torch.OutOfMemoryError) else 'failed_stopped',
                      error_type=type(exc).__name__, error=str(exc)); raise
    finally:
        result.update(wall_s=time.monotonic()-started, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        (OUT/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

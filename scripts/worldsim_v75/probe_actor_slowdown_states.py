"""验证每个重建状态的减速编辑在前缀完全一致、事件后可分。"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageChops, ImageDraw
import torch

from closed_loop_bridge import upload_scene
from render_argoverse import render


ROOT = Path('/root/autodl-tmp/runs/worldsim_v75')
OUT = ROOT/'WS-V75-ACTOR-SLOWDOWN-STATE-QUALIFY-01/20260921-r2'
FRAMES = [0, 4, 5, 45, 61, 85, 109, 116]
LATE = [45, 61, 85, 109, 116]
ARMS = ['reference', 'dvgt_metric', 'class_prior']


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    global OUT
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args(); OUT = args.output
    prior = json.loads((OUT/'result.json').read_text())
    assert prior['status'] == 'passed_pending_raster' and prior['raster_probe_required']
    assert not (OUT/'raster_result.json').exists()
    selected = prior['selected']; base = Path(selected['base']); tr = np.load(base/'trajectory.npz')
    began = time.monotonic(); rows = []; rendered = {}
    result = {'status': 'started', 'frames': FRAMES, 'arms': ARMS, 'world_model_generation_calls': 0,
              'human_verdict': None, 'failure_ledger_delta': 'none'}
    try:
        for arm in ARMS:
            info = selected['state_inputs'][arm]
            for variant, path in [('unedited', Path(info['source'])), ('edited', Path(info['edited']))]:
                scene = json.loads(path.read_text()); ctx, sid, fit = upload_scene(scene, tr['timestamps_us'], tr['K'])
                images = render(ctx, sid, tr['timestamps_us'][FRAMES], tr['camera_world'][FRAMES])
                rendered[arm, variant] = images
                del ctx; torch.cuda.empty_cache()
            for i, frame in enumerate(FRAMES):
                before, after = rendered[arm, 'unedited'][i], rendered[arm, 'edited'][i]
                mask = np.any(before != after, axis=-1); ys, xs = np.where(mask)
                bounds = None if len(xs) == 0 else [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]
                rows.append({'arm': arm, 'frame': frame, 'changed_pixels': int(mask.sum()),
                             'difference_bounds_xyxy': bounds, 'exactly_equal': bool(np.array_equal(before, after))})
        gate = {arm: (all(next(x for x in rows if x['arm'] == arm and x['frame'] == f)['exactly_equal'] for f in [0, 4]) and
                      all(next(x for x in rows if x['arm'] == arm and x['frame'] == f)['changed_pixels'] >= 50 for f in LATE))
                for arm in ARMS}
        sheet = Image.new('RGB', (5*320, 3*320), '#142130'); draw = ImageDraw.Draw(sheet)
        for ri, arm in enumerate(ARMS):
            for ci, frame in enumerate(LATE):
                idx = FRAMES.index(frame); before = Image.fromarray(rendered[arm, 'unedited'][idx]); after = Image.fromarray(rendered[arm, 'edited'][idx])
                diff = ImageChops.difference(before, after).point(lambda x: min(255, x*4))
                panel = Image.new('RGB', (320, 285), 'black'); panel.paste(before.resize((320, 176)), (0, 0))
                panel.paste(after.resize((160, 88)), (0, 176)); panel.paste(diff.resize((160, 88)), (160, 176))
                changed = next(x['changed_pixels'] for x in rows if x['arm'] == arm and x['frame'] == frame)
                ImageDraw.Draw(panel).text((4, 266), f'f{frame} changed={changed}', fill='white')
                sheet.paste(panel, (ci*320, ri*320+25))
            draw.text((5, ri*320+4), f'{arm}: top original; lower slowed / x4 diff', fill='white')
        sheet.save(OUT/'raster-probe-review.jpg', quality=94)
        passed = all(gate.values())
        result.update(status='passed' if passed else 'failed', rows=rows, arm_gates=gate,
                      exact_prefix_through_frame4=passed, post_event_input_changed=passed,
                      generation_admitted=passed, condition_render_calls=len(ARMS)*2,
                      rendered_condition_frames=len(ARMS)*2*len(FRAMES))
    except BaseException as exc:
        result.update(status='oom_stopped' if isinstance(exc, torch.OutOfMemoryError) else 'failed_stopped',
                      error_type=type(exc).__name__, error=str(exc), generation_admitted=False)
        raise
    finally:
        result.update(wall_s=time.monotonic()-began, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      world_model_generation_calls=0)
        save(OUT/'raster_result.json', result)
    prior.update(status='qualified' if result['generation_admitted'] else result['status'],
                 generation_admitted=result['generation_admitted'], raster_result=str(OUT/'raster_result.json'))
    save(OUT/'result.json', prior)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

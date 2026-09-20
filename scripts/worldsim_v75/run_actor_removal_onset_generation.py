"""单一预注册机制诊断：对象从第0帧状态缺失，区分图像锚定与历史锁定。"""
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import time

import av
import numpy as np
from PIL import Image
import torch

from closed_loop_bridge import upload_scene
from common import CAMERA, config
from render_argoverse import render


ROOT = Path('/root/autodl-tmp/runs/worldsim_v75')
QUAL = ROOT/'WS-V75-ACTOR-REMOVAL-CONFIRM-QUALIFY-01/20260921-r3'
SOURCE = ROOT/'WS-V75-ACTOR-REMOVAL-CONFIRM-GENERATION-01/20260921-r1'
OUT = ROOT/'WS-V75-ACTOR-REMOVAL-ONSET-01/20260921-r1'
FRAMES, BLOCKS = 117, 15


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('LOCAL_FILES_ONLY', '1')
    assert not OUT.exists(); OUT.mkdir(parents=True)
    q = json.loads((QUAL/'result.json').read_text())
    old = json.loads((SOURCE/'evaluation.json').read_text())
    assert q['status'] == 'qualified' and not old['reference_gate_passed']
    assert next(x for x in old['summaries'] if x['arm']=='reference')['removed']['actor_present'] == 10
    selected = q['selected']; actor = selected['actor']; behind = selected['behind']
    protocol = {
        'task_id': 'WS-V75-ACTOR-REMOVAL-ONSET-01', 'run_id': OUT.name,
        'frozen_utc': datetime.now(timezone.utc).isoformat(), 'source_revision': 'ff62e1b2',
        'qualification': str(QUAL), 'source_generation': str(SOURCE),
        'question': 'Does removal fail because the unchanged initial RGB anchors A, or because frames0-4 first establish A in autoregressive history?',
        'intervention': 'remove target A from the reference condition scene at every frame starting frame0; initial RGB/text embeddings remain unchanged and still visibly contain A',
        'comparison': 'reuse frozen seed42 reference-unedited and reference-removed-after-frame5 outputs; generate exactly one reference-removed-from-frame0 sequence',
        'fixed': ['same initial RGB/text embeddings', 'reference state for all other actors/map', 'seed42', '117 frames', 'recorded camera', 'generator weights/config'],
        'evaluation_frames': [4, 5, 13, 21, 29, 37, 45, 61, 85, 109, 116],
        'frozen_interpretation': {
            'initial_image_anchor': 'A detected at >=6 of the ten post-frame5 samples even when absent from all state conditions',
            'history_lock': 'A detected at <=2 samples for frame0 removal while existing frame5 removal has A at >=6 samples',
            'mixed_or_inconclusive': 'all other outcomes',
        },
        'boundary': 'one independent source and one post-failure mechanism diagnostic; detector identity remains a proxy and review image is required',
        'stop': 'one sequence only; OOM/error stops without retry, new seed/source/event time/resolution or threshold',
        'human_verdict': None, 'failure_ledger_delta': 'none',
    }
    save(OUT/'protocol.json', protocol)
    result = {'status': 'started', 'human_verdict': None, 'failure_ledger_delta': 'none'}
    began = time.monotonic()
    try:
        reference = json.loads(Path(selected['state_inputs']['reference']['source']).read_text())
        onset = json.loads(json.dumps(reference))
        before = len(onset['tracks'])
        onset['tracks'] = [t for t in onset['tracks'] if t['id'] != actor]
        assert len(onset['tracks']) < before and all(t['id'] != actor for t in onset['tracks'])
        scene_path = OUT/'condition-reference-removed-from-000.json'; save(scene_path, onset)
        base = Path(selected['base']); tr = np.load(base/'trajectory.npz')
        conditions = np.lib.format.open_memmap(OUT/'conditions.npy', mode='w+', dtype=np.uint8, shape=(FRAMES,704,1280,3))
        ctx, sid, fit = upload_scene(onset, tr['timestamps_us'], tr['K'])
        for start in range(0, FRAMES, 8):
            stop = min(FRAMES, start+8)
            conditions[start:stop] = render(ctx, sid, tr['timestamps_us'][start:stop], tr['camera_world'][start:stop])
        conditions.flush(); del ctx; torch.cuda.empty_cache()
        old_conditions = np.load(SOURCE/'reference-unedited/conditions.npy', mmap_mode='r')
        raster = []
        for frame in [0,5,30,90]:
            mask = np.any(old_conditions[frame] != conditions[frame], axis=-1)
            raster.append({'frame': frame, 'changed_pixels': int(mask.sum()), 'exactly_equal': bool(not mask.any())})
        assert all(x['changed_pixels'] >= 50 for x in raster)
        save(OUT/'raster_result.json', {'status':'passed','rows':raster,'condition_render_frames':FRAMES,
                                        'world_model_generation_calls':0,'human_verdict':None,'failure_ledger_delta':'none'})
        cfg = config(); cfg.text_encoder = None; cfg.image_encoder = None; cfg.diffusion_model.seed = 42
        pipeline = cfg.setup().to('cuda').eval()
        embeddings = torch.load(Path(selected['conditioning'])/'embeddings.pt', weights_only=True, map_location='cpu')
        cache = pipeline.initialize_cache_from_embeddings(**embeddings, view_names=[CAMERA])
        torch.cuda.reset_peak_memory_stats()
        generated = np.lib.format.open_memmap(OUT/'generated.npy', mode='w+', dtype=np.uint8, shape=(FRAMES,704,1280,3))
        cursor = 0; timings = []
        with av.open(str(OUT/'generated.mp4'), 'w') as writer:
            stream = writer.add_stream('libx264', rate=30); stream.width=1280; stream.height=704; stream.pix_fmt='yuv420p'; stream.options={'crf':'18'}
            for block in range(BLOCKS):
                count = 5 if block == 0 else 8
                source = torch.from_numpy(np.array(conditions[cursor:cursor+count], copy=True)).permute(0,3,1,2)[None,None]
                source = source.to('cuda', dtype=torch.bfloat16)/127.5-1; start = time.monotonic()
                with torch.inference_mode():
                    output = pipeline.generate(autoregressive_index=block, cache=cache, input=source)
                    pipeline.finalize(autoregressive_index=block, cache=cache)
                torch.cuda.synchronize()
                frames = ((output[0,0].float().clamp(-1,1)+1)*127.5).round().byte().permute(0,2,3,1).cpu().numpy()
                generated[cursor:cursor+count] = frames
                for frame in frames:
                    for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format='rgb24')): writer.mux(packet)
                cursor += count; timings.append({'block':block,'last_frame':cursor-1,'elapsed_s':time.monotonic()-start})
                print(json.dumps(timings[-1]), flush=True)
            for packet in stream.encode(): writer.mux(packet)
        generated.flush(); assert cursor == FRAMES
        for frame in protocol['evaluation_frames']:
            Image.fromarray(np.asarray(generated[frame])).save(OUT/f'generated-{frame:03d}.jpg', quality=94)
        result.update(status='complete', frames=FRAMES, blocks=BLOCKS, seed=42, generation_forwards=BLOCKS,
                      condition_render_frames=FRAMES, scene=str(scene_path), timings=timings)
    except BaseException as exc:
        result.update(status='oom_stopped' if isinstance(exc, torch.OutOfMemoryError) else 'failed_stopped',
                      error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result.update(wall_s=time.monotonic()-began, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        save(OUT/'result.json', result)
        print(json.dumps({k:v for k,v in result.items() if k!='timings'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

"""Two matched 21-frame ProPainter controls; no model or Gaussian training."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, '/root/autodl-tmp/_wsjob')
from vace_run_clip import CLIPS, OUT, R5, SRC, VramSampler
from scene065_propainter_20261126_r5 import feather_ramp, tone_medirq, region_stats

PY = '/root/autodl-tmp/envs/driveeditor/bin/python'
REPO = '/root/autodl-tmp/third_party/worldsim_v77/ProPainter-hfspace'
TAG = 'w8_swept_cd18'

for clip, (lo, hi) in CLIPS.items():
    dest = Path(OUT) / clip / 'matched_propainter'
    dest.mkdir(parents=True, exist_ok=True)
    raw_dir = dest / 'raw'
    cmd = [PY, 'inference_propainter.py', '-i', f'{OUT}/{clip}/src_video',
           '-m', f'{OUT}/{clip}/src_mask', '-o', str(raw_dir), '--fp16', '--save_frames',
           '--mask_dilation', '4', '--neighbor_length', '10', '--ref_stride', '10',
           '--subvideo_length', '50', '--save_fps', '10']
    cached = list(raw_dir.rglob('frames/*.png')) if raw_dir.exists() else []
    if len(cached) != hi-lo+1:
        sampler = VramSampler(); sampler.start(); start = time.perf_counter()
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        wall = time.perf_counter()-start
        sampler.stop(); sampler.join(timeout=6)
        (dest/'stdout.log').write_text((proc.stdout or '')+'\n'+(proc.stderr or ''))
        info = dict(returncode=proc.returncode, wall_s=wall, peak_vram_mb=sampler.peak,
                    command=cmd, model_training_updates=0, gaussian_updates=0)
        (dest/'run_info.json').write_text(json.dumps(info, indent=2))
        if proc.returncode:
            raise RuntimeError((proc.stderr or '')[-2500:])
        cached = list(raw_dir.rglob('frames/*.png'))
    preds = sorted(cached)
    assert len(preds) == hi-lo+1, (clip, len(preds))
    for tone in ('none', 'medirq'):
        folder = dest/tone/'final'; folder.mkdir(parents=True, exist_ok=True)
        rows = []
        for k, raw in enumerate(range(lo, hi+1)):
            orig = cv2.imread(f'{R5}/input_frames/{raw:05d}.png')
            pred = cv2.imread(str(preds[k]))
            mask = cv2.imread(f'{OUT}/{clip}/src_mask/{k:05d}.png', 0)>0
            assert pred.shape == orig.shape == (320,480,3)
            if tone == 'medirq': pred, _ = tone_medirq(orig, pred, mask)
            ramp = feather_ramp(mask)[...,None]
            final = np.clip(orig.astype(np.float32)*(1-ramp)+pred.astype(np.float32)*ramp,0,255).astype(np.uint8)
            final[~mask] = orig[~mask]
            K = np.load(f'{SRC}/domains/{raw:03d}.npz')['K'].astype(bool)
            delta = np.abs(final.astype(np.int16)-orig.astype(np.int16)).max(2)
            assert not delta[~mask].any() and not delta[K].any()
            cv2.imwrite(str(folder/f'{raw:05d}.png'),final)
            rows.append(dict(raw=raw, mask_pixels=int(mask.sum()), outside_changed=0, retained_changed=0,
                             metrics=region_stats(final,mask)))
        (dest/tone/'summary.json').write_text(json.dumps(dict(clip=clip,tone=tone,raw_range=[lo,hi],
            frames=len(rows),model_training_updates=0,gaussian_updates=0,per_frame=rows),indent=2))
    print(f'ProPainter matched clip {clip}: {len(preds)} frames complete', flush=True)

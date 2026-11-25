"""VACE smoke test: 5 frames, 2 sampling steps, clip 087 input.

The point is not the picture, it is to prove on this machine that
  - the official `generate.py --task vace-1.3B` entry loads the 19 GB checkpoint,
  - the 480*320 size registration works,
  - the src_video / src_mask / src_ref_images plumbing reads our directories,
  - and to measure peak VRAM and wall time per step, before burning a 25-step clip.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, "/root/autodl-tmp/_wsjob")
from vace_run_clip import CKPT, OUT, PYBIN, REPO, SITE, VramSampler, log  # noqa: E402

CLIP = "087"
SMOKE = f"{OUT}/smoke_{CLIP}"
PROMPT = "a clean empty asphalt road at a traffic light intersection, lane markings, daylight"


def main() -> int:
    os.makedirs(SMOKE, exist_ok=True)
    for kind in ('src_video', 'src_mask'):
        os.makedirs(f'{SMOKE}/{kind}', exist_ok=True)
        for i in range(5):
            shutil.copyfile(f'{OUT}/{CLIP}/{kind}/{i:05d}.png', f'{SMOKE}/{kind}/{i:05d}.png')
    cmd = [
        PYBIN, "generate.py", "--task", "vace-1.3B", "--size", "480*320",
        "--ckpt_dir", CKPT,
        "--src_video", f"{SMOKE}/src_video",
        "--src_mask", f"{SMOKE}/src_mask",
        "--prompt", PROMPT,
        "--frame_num", "5", "--sample_steps", "2", "--base_seed", "2026",
        "--save_file", f"{SMOKE}/smoke.mp4",
    ]
    low = sys.argv[1:] if len(sys.argv) > 1 else []
    if "--low-vram" in low:
        cmd += ["--offload_model", "True", "--t5_cpu"]
    log("smoke cmd: " + " ".join(cmd))

    sampler = VramSampler()
    sampler.start()
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO, env=dict(os.environ, PYTHONPATH=SITE),
                          capture_output=True, text=True)
    wall = time.time() - t0
    sampler.stop()
    sampler.join(timeout=6)

    with open(f"{SMOKE}/stdout.log", "w", encoding="utf-8") as fh:
        fh.write(proc.stdout or "")
        fh.write("\n--- STDERR ---\n")
        fh.write(proc.stderr or "")

    info = {
        "returncode": proc.returncode,
        "wall_s": round(wall, 1),
        "peak_vram_mb": round(sampler.peak, 1),
        "low_vram": bool(low),
        "frame_num": 5, "sample_steps": 2, "size": "480*320",
        "mp4_exists": os.path.exists(f"{SMOKE}/smoke.mp4"),
    }
    with open(f"{SMOKE}/smoke_info.json", "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    log(json.dumps(info, ensure_ascii=False))
    if proc.returncode != 0:
        log("\n".join((proc.stderr or "").splitlines()[-40:]))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""只为官方 DGGT mode 2 生成隔离兼容副本，保留网络、相机与高斯渲染公式。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--external', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    source = args.external/'inference.py'
    original = source.read_text(encoding='utf-8')
    text = original
    for line in (
        'from third_party.TAPIP3D.utils.inference_utils import load_model, read_video, inference, get_grid_queries, resize_depth_bilinear',
        'from utils.interplation import interp_all',
    ):
        if text.count(line) != 1:
            raise ValueError('官方 import 已变化，先核对源码')
        text = text.replace(line, '# mode 2: unused tracking/interpolation import omitted')
    if text.count('if args.difix:') != 1:
        raise ValueError('官方diffusion拼写已变化')
    text = text.replace('if args.difix:', 'if args.diffusion:')
    text = text.replace('    args = parser.parse_args()',
        "    args = parser.parse_args()\n    if args.mode != 2 or args.diffusion:\n"
        "        raise ValueError('此副本只跑无Difix的官方mode2基线；Difix另行离线运行')")
    for line in ("            gt_dy_map = batch['dynamic_mask'].to(device)\n",
                 "            gt_depth = batch['gt_depth'].to(device)\n"):
        if text.count(line) != 1:
            raise ValueError('GT展示字段读取已变化')
        text = text.replace(line, '')
    start = text.index('            gt_frames = target_image.detach().cpu()')
    end = text.index('            if args.depth:', start)
    text = (text[:start] + "            # Ordinary Waymo has no scene-flow GT depth; omit GT-only display.\n"
            "            depth_frames = predictions['depth'][0].detach().cpu()\n" + text[end:])
    marker = '            psnr, ssim, lpip = compute_metrics(rendered_image, target_image, loss_fn)'
    if text.count(marker) != 1:
        raise ValueError('官方render输出位置已变化')
    keys = ('extrinsic', 'intrinsic', 'point_map', 'gs_map', 'gs_conf', 'dy_map', 'bg_mask', 'timestamps', 'bg_render')
    values = ["'rgb': rendered_image.detach().cpu()", "'input_rgb': images.detach().cpu()"]
    values += [f"'{key}': {key}.detach().cpu()" for key in keys]
    trace = "            torch.save({" + ', '.join(values) + "}, os.path.join(args.output_path, 'official_trace.pt'))\n"
    text = text.replace(marker, trace + marker)
    compile(text, str(source), 'exec')
    destination = args.run/'source/official_mode2_runtime.py'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_text(encoding='utf-8') != text:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        shutil.copy2(destination, destination.with_name(f'official_mode2_runtime_{stamp}.bak.py'))
    shutil.copy2(source, destination.with_name('official_mode2_original.py'))
    destination.write_text(text, encoding='utf-8')
    evidence = args.run/'evidence/official_runtime_compatibility.json'
    evidence.parent.mkdir(exist_ok=True)
    evidence.write_text(json.dumps({
        'official_source': str(source),
        'official_revision': subprocess.check_output(['git', '-C', str(args.external), 'rev-parse', 'HEAD'], text=True).strip(),
        'runtime_source': str(destination), 'only_mode': 2, 'model_renderer_changes': False,
        'pretrained_weights_unchanged': True,
        'changes': ['Omit unused mode3 tracking imports', 'Fix args.difix spelling to args.diffusion',
                    'Disallow mode3 and in-script diffusion', 'Save actual renderer trace',
                    'Omit unavailable GT dynamic/depth display; retain predicted depth export'],
        'gt_depth_role': 'unavailable; not model input',
        'gt_dynamic_role': 'not required; omitted GT-only panel',
        'official_edit_cli_released': False, 'paper_editing_supported': True},
        ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'runtime_prepared', 'path': str(destination)}))


if __name__ == '__main__':
    main()

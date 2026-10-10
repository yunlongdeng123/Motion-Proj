"""六个固定短窗的全25帧PNG误差诊断；不是论文正式指标，不计算FVD。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


SEQUENCES = ('00f88c4f0a', '7e625db8c4', 'ff6eb95840')
SIDES = (.125, .33)


def load(folder):
    expected = [f'{i:05d}.png' for i in range(25)]
    if sorted(p.name for p in folder.glob('*.png')) != expected:
        raise ValueError(f'需要完整25张原PNG: {folder}')
    images = []
    for name in expected:
        with Image.open(folder / name) as image:
            if image.mode != 'RGB' or image.size != (256, 256):
                raise ValueError(f'PNG尺寸/通道错误: {folder/name}')
            images.append(np.asarray(image).astype(np.float32) / 255)
    return np.stack(images)


def metrics(target, output, mask):
    delta = output - target
    per_frame = []
    for index in range(25):
        error = delta[index][mask]
        mse = float(np.square(error).mean())
        per_frame.append({'frame': index, 'l1_0_1': float(np.abs(error).mean()),
                          'psnr_db': float(-10 * np.log10(mse)) if mse > 0 else None,
                          'exact_match': mse == 0})
    finite_psnr = [row['psnr_db'] for row in per_frame if row['psnr_db'] is not None]
    motion_error = np.diff(output, axis=0) - np.diff(target, axis=0)
    return {'mean_frame_l1_0_1': float(np.mean([r['l1_0_1'] for r in per_frame])),
            'mean_frame_psnr_db': float(np.mean(finite_psnr)) if finite_psnr else None,
            'finite_psnr_frames': len(finite_psnr),
            'temporal_rgb_delta_l1_0_1': float(np.abs(motion_error[:, mask]).mean()),
            'per_frame': per_frame}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--media', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps({'status': 'static_ready', 'gpu_started': False, 'frames_per_window': 25}))
        return
    phase = args.media / 'paper_bidirectional_m4/validation'
    cases = []
    for sequence in SEQUENCES:
        for side in SIDES:
            folders = {step: phase / f'step{step:06d}' / sequence / f'side_{side:g}'
                       for step in (5000, 7500)}
            metadata = {step: json.loads((folder/'run.json').read_text(encoding='utf-8'))
                        for step, folder in folders.items()}
            for step, record in metadata.items():
                if (record.get('checkpoint_step') != step or record.get('status') != 'complete'
                        or record.get('seed') != 2026 or record.get('steps') != 25
                        or record.get('mode') != 'paper-feedforward'
                        or record.get('propagation_protocol') != 'paper-bidirectional-m4'):
                    raise ValueError(f'正式短窗协议不符: {folders[step]}')
            for key in ('source_frames', 'mask_pixels_left', 'mask_pixels_right', 'reference_indices', 'flow_pairs'):
                if metadata[5000].get(key) != metadata[7500].get(key):
                    raise ValueError(f'5000/7500未匹配: {sequence}/{side}/{key}')
            target = load(folders[7500]/'frames_gt')
            visible = load(folders[7500]/'frames_visible')
            if not np.array_equal(target, load(folders[5000]/'frames_gt')):
                raise ValueError('两个学习点GT不完全一致')
            hole = np.zeros((256, 256), dtype=bool)
            hole[:, :metadata[7500]['mask_pixels_left']] = True
            hole[:, 256-metadata[7500]['mask_pixels_right']:] = True
            regions = {'hole': hole, 'visible': ~hole, 'full': np.ones_like(hole)}
            result = {}
            for step, folder in folders.items():
                pred = load(folder/'frames_pred')
                comp = load(folder/'frames_comp')
                if not np.array_equal(comp[:, ~hole], visible[:, ~hole]):
                    raise ValueError('硬写回可见区不是真实输入')
                if not np.array_equal(comp[:, hole], pred[:, hole]):
                    raise ValueError('硬写回洞区不是模型生成')
                result[str(step)] = {role: {name: metrics(target, values, mask)
                                           for name, mask in regions.items()}
                                     for role, values in (('native', pred), ('composite', comp))}
            cases.append({'sequence_id': sequence, 'side_ratio_each': side,
                          'num_frames': 25, 'regions': result,
                          'native_hole_l1_gain_positive_is_better':
                              result['5000']['native']['hole']['mean_frame_l1_0_1'] -
                              result['7500']['native']['hole']['mean_frame_l1_0_1']})
    report = {'status': 'complete', 'kind': 'six_fixed_validation_windows_all25_png_diagnostic',
              'formal_metrics_computed': False, 'optimizer_updates': 0, 'human_verdict': None,
              'cases': cases, 'source': '原始PNG，不从压缩MP4抽帧',
              'limits': ['不是DAVIS90/YouTube60正式指标，也不是论文PSNR汇总协议。',
                         '真实中心硬写回误差为0不计模型能力。',
                         'RGB时间差误差不是光流对齐的闪烁指标；必须结合全帧结构/动作审核。',
                         '生成多解使逐像素GT误差不能独立代表视觉合理性。',
                         'PSNR为逐帧区域均值；精确匹配帧PSNR为无穷，JSON用null并记录计数。']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'status': 'complete', 'windows': len(cases), 'output': str(args.output)}))


if __name__ == '__main__':
    main()

"""评价口径回归；小张量，不运行场景训练或49帧渲染。"""

from pathlib import Path
import copy
import json
import math
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'engine' if (ROOT/'engine').exists() else ROOT/'engine_overlay'))
sys.path.insert(0, str(ROOT/'tools'))
from v4_eval_metrics import region_metrics, full_perceptual_metrics, aggregate49
from utils.loss_utils import psnr, ssim

FRAMES = [f'0008{raw:03d}' for raw in range(89, 138)]


class MetricTests(unittest.TestCase):
    def test_above_one_matches_V1_without_modifying_raw_float(self):
        image = torch.full((3, 16, 16), 1.2)
        target = torch.ones_like(image)
        before = image.clone()
        mask = torch.ones(1, 16, 16, dtype=torch.bool)
        compatible = region_metrics(image, target, mask)
        raw = region_metrics(image, target, mask, clamp=False)
        self.assertEqual(compatible['mae'], 0.)
        self.assertEqual(compatible['mse'], 0.)
        self.assertIsNone(compatible['psnr'])
        self.assertTrue(compatible['psnr_is_positive_infinity'])
        self.assertAlmostEqual(raw['psnr'], 13.9794, places=4)
        self.assertTrue(torch.equal(image, before))
        json.dumps(compatible, allow_nan=False)

    def test_prediction_and_GT_both_clamped_as_V1(self):
        image = torch.linspace(-.5, 1.5, 3*16*16).reshape(3, 16, 16)
        target = image.flip(-1)+.1
        result = region_metrics(image, target, torch.ones(1, 16, 16, dtype=torch.bool))
        self.assertEqual(result['psnr'], float(psnr(image.clamp(0, 1), target.clamp(0, 1))))
        self.assertEqual(result['mae'], float((image.clamp(0, 1)-target.clamp(0, 1)).abs().mean()))

    def test_perceptual_uses_native_SSIM_and_V1_LPIPS_arguments(self):
        image = torch.linspace(-.3, 1.4, 3*16*16).reshape(3, 16, 16)
        target = image.flip(-1)
        calls = []

        def lpips_check(prediction, truth, **kwargs):
            self.assertTrue(torch.equal(prediction, image.clamp(0, 1)))
            self.assertTrue(torch.equal(truth, target.clamp(0, 1)))
            self.assertEqual(kwargs, dict(net_type='vgg', version='0.1'))
            calls.append(True)
            return torch.tensor(.125)

        result = full_perceptual_metrics(image, target, lpips_fn=lpips_check)
        self.assertEqual(result['ssim'], float(ssim(image.clamp(0, 1), target.clamp(0, 1))))
        self.assertEqual(result['lpips'], .125)
        self.assertEqual(len(calls), 1)

    def test_empty_domain_explicit_and_nonfinite_raw_rejected(self):
        image = torch.zeros(3, 8, 8)
        empty = region_metrics(image, image, torch.zeros(1, 8, 8, dtype=torch.bool))
        self.assertEqual(empty['pixels'], 0)
        self.assertIsNone(empty['psnr'])
        self.assertFalse(empty['psnr_is_positive_infinity'])
        image[0, 0, 0] = math.inf
        with self.assertRaises(ValueError):
            region_metrics(image, torch.zeros_like(image), torch.ones(1, 8, 8, dtype=torch.bool))

    def records(self):
        records = []
        for index, frame in enumerate(FRAMES):
            value = dict(pixels=16, mae=.1, mse=.01 if index % 2 else .1,
                         psnr=20. if index % 2 else 10., psnr_is_positive_infinity=False)
            raw = {name: dict(full=dict(value)) for name in ('v1', 'composite')}
            regular = copy.deepcopy(raw)
            for name in regular:
                regular[name]['full'].update(ssim=.75+index/1000, lpips=.25-index/1000)
            records.append(dict(frame=frame, regions=regular, raw_regions=raw))
        return records

    def test_49_summary_is_per_frame_mean_not_PSNR_of_mean_MSE(self):
        records = self.records()
        summary = aggregate49(records, FRAMES)
        full = summary['regions']['composite']['full']
        expected = sum(record['regions']['composite']['full']['psnr'] for record in records)/49
        self.assertEqual(full['psnr'], expected)
        self.assertNotAlmostEqual(full['psnr'], -10*math.log10(full['mse']))
        self.assertAlmostEqual(full['ssim'], .774)
        self.assertAlmostEqual(full['lpips'], .226)
        self.assertEqual(full['nonempty_frame_count'], 49)
        self.assertEqual(full['pixels'], 49*16)
        json.dumps(summary, allow_nan=False)

    def test_perfect_frame_not_dropped_or_capped_in_mean(self):
        records = self.records()
        for domain in ('regions', 'raw_regions'):
            for name in ('v1', 'composite'):
                records[0][domain][name]['full'].update(psnr=None, psnr_is_positive_infinity=True)
        full = aggregate49(records, FRAMES)['regions']['v1']['full']
        self.assertIsNone(full['psnr'])
        self.assertTrue(full['psnr_is_positive_infinity'])
        self.assertEqual(full['psnr_positive_infinity_frames'], 1)
        self.assertEqual(full['nonempty_frame_count'], 49)

    def test_missing_duplicate_frames_or_perceptual_metric_rejected(self):
        records = self.records()
        with self.assertRaises(ValueError):
            aggregate49(records[:-1], FRAMES)
        repeated = copy.deepcopy(records); repeated[-1]['frame'] = FRAMES[0]
        with self.assertRaises(ValueError):
            aggregate49(repeated, FRAMES)
        records[0]['regions']['v1']['full']['lpips'] = None
        with self.assertRaises(ValueError):
            aggregate49(records, FRAMES)


if __name__ == '__main__':
    unittest.main()

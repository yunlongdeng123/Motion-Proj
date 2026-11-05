import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
import torch

spec = importlib.util.spec_from_file_location("paper_metrics", Path(__file__).with_name("paper_metrics.py"))
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


class RegionMetricsTests(unittest.TestCase):
    def test_known_roi_vs_zero_padding_and_removal(self):
        target = torch.zeros(3, 2, 2)
        prediction = torch.full_like(target, 0.5)
        mask = torch.tensor([[[1, 0], [0, 0]]])
        row = metrics.measure_frame(prediction, target, target, mask, "frame")
        self.assertAlmostEqual(row["dynamic_psnr_star"]["psnr_roi"], 10 * math.log10(4))
        self.assertAlmostEqual(row["dynamic_psnr_star"]["psnr_zero_padded"], 10 * math.log10(16))
        self.assertEqual(row["static_psnr_s"]["pixels"], 3)
        self.assertTrue(row["static_psnr_sr"]["perfect"])

    def test_empty_mask_and_json_have_no_nan(self):
        target = torch.zeros(3, 2, 2)
        row = metrics.measure_frame(target, target, target, torch.zeros(1, 2, 2), 0)
        self.assertIsNone(row["dynamic_psnr_star"]["psnr_roi"])
        with tempfile.TemporaryDirectory() as output:
            report = metrics.save_metrics([row], output, "train", 100)
            self.assertEqual(report["summary"]["static_psnr_s"]["perfect_frames"], 1)
            self.assertNotIn("NaN", (Path(output) / "paper_region_metrics.json").read_text())

    def test_only_selected_pixels_contribute(self):
        target = torch.zeros(3, 2, 2)
        prediction = torch.ones_like(target)
        prediction[:, 0, 0] = 0.25
        row = metrics.measure_frame(prediction, None, target, torch.tensor([[[1, 0], [0, 0]]]), 1)
        self.assertAlmostEqual(row["dynamic_psnr_star"]["psnr_roi"], 10 * math.log10(16))


if __name__ == "__main__":
    unittest.main()

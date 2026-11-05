import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch

spec = importlib.util.spec_from_file_location("paper_masks", Path(__file__).with_name("paper_masks.py"))
masks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(masks)


class ExplicitMaskTests(unittest.TestCase):
    def test_black_actor_pixels_and_shadow_union_survive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, values in (("dynamic_mask_0", [[255, 0], [0, 0]]),
                                 ("shadow_mask_0", [[0, 0], [255, 0]])):
                folder = root / "dynamic_masks" / name
                folder.mkdir(parents=True)
                Image.fromarray(np.asarray(values, dtype=np.uint8)).save(folder / "0017000.png")
            camera = SimpleNamespace(image_path=str(root / "image/image_0/0017000.png"),
                                     original_image=torch.zeros(3, 2, 2), image_width=2,
                                     image_height=2, world_view_transform=torch.eye(4))
            actual = masks.load_dynamic_mask(root, camera)
            self.assertTrue(torch.equal(actual, torch.tensor([[[1., 0.], [1., 0.]]])))

    def test_missing_binary_mask_fails(self):
        with tempfile.TemporaryDirectory() as root:
            camera = SimpleNamespace(image_path=f"{root}/image/image_2/missing.png",
                                     image_width=2, image_height=2, world_view_transform=torch.eye(4))
            with self.assertRaises(FileNotFoundError):
                masks.load_dynamic_mask(root, camera)


if __name__ == "__main__":
    unittest.main()

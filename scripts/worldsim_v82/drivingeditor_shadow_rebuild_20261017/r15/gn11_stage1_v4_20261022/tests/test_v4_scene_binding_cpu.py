"""在不创建 CUDA Scene 或更新模型时，回归 V1 评价前必需的场景绑定。"""

from pathlib import Path
from types import SimpleNamespace
import json
import random
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from run_v4_arm import bind_scene_contract


class SceneBindingTests(unittest.TestCase):
    def fixture(self, directory):
        cameras = [SimpleNamespace(image_path=f'/input/image_0/{index:07d}.png',
            image_name=f'{index:07d}', timestamp=(index-24)*.02,
            image_width=480, image_height=320) for index in range(49)]
        spec = dict(frame_ids_with_camera=['image_0/'+camera.image_name for camera in cameras],
            model_timestamps=[camera.timestamp for camera in cameras], anchor_frame=cameras[10].image_name)
        # 检查可排序的视图，但不能重排真实 Scene 的采样列表。
        scene = SimpleNamespace(getTrainCameras=lambda: list(reversed(cameras)),
            getTestCameras=lambda scale: [])
        training = SimpleNamespace(Scene=lambda *args, **kwargs: scene)
        config = dict(followup_scene_spec=spec, model_path=str(directory))
        bind_scene_contract(training, config)
        return training, config, scene, cameras

    def test_binds_original_scene_without_reordering_or_rng_consumption(self):
        with tempfile.TemporaryDirectory() as directory:
            training, config, scene, cameras = self.fixture(directory)
            rng = random.getstate()
            result = training.Scene('unchanged argument')
            self.assertIs(result, scene)
            self.assertIs(result.followup_scene_spec, config['followup_scene_spec'])
            self.assertIs(type(result.followup_scene_spec), dict)
            self.assertEqual(result.getTrainCameras(), list(reversed(cameras)))
            self.assertEqual(random.getstate(), rng)
            record = json.loads((Path(directory)/'actual_scene_binding.json').read_text(encoding='utf-8'))
            self.assertEqual(record['frame_ids'], config['followup_scene_spec']['frame_ids_with_camera'])
            self.assertEqual(record['anchor_pose_frame'], cameras[10].image_name)

    def test_rejects_wrong_camera_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            training, _, scene, cameras = self.fixture(directory)
            cameras[0].image_path = '/input/image_1/0000000.png'
            with self.assertRaisesRegex(ValueError, 'Scene'):
                training.Scene()
            self.assertFalse(hasattr(scene, 'followup_scene_spec'))

    def test_rejects_wrong_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            training, _, _, cameras = self.fixture(directory)
            cameras[0].timestamp += .001
            with self.assertRaisesRegex(ValueError, 'timestamps'):
                training.Scene()

    def test_rejects_wrong_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            training, _, _, cameras = self.fixture(directory)
            cameras[0].image_width = 481
            with self.assertRaisesRegex(ValueError, 'dimensions'):
                training.Scene()

    def test_rejects_added_heldout(self):
        with tempfile.TemporaryDirectory() as directory:
            training, _, scene, cameras = self.fixture(directory)
            scene.getTestCameras = lambda scale: cameras[:1]
            with self.assertRaisesRegex(ValueError, 'heldout'):
                training.Scene()


if __name__ == '__main__':
    unittest.main()

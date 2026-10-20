"""真实发布/严格恢复函数的49帧集成回归。CPU夹具不是训练检查点。

默认仅mock CUDA接口，不mock schema校验；单独设置
V3_CHECKPOINT_TEST_REAL_CUDA=1 可复用实际CUDA RNG验证，不执行训练。
"""
import copy
import ast
from contextlib import ExitStack
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT/'engine_overlay' if (ROOT/'engine_overlay').is_dir() else ROOT/'engine'
sys.path.insert(0, str(ENGINE))
from utils import paper_checkpoint as checkpoint, v3_runtime_context
from utils.hybrid30k_runtime import CameraSampler
from test_v3_multiview_road_cpu import toy, toy_render, WEIGHTS


class Component:
    no_time_split = True
    def __init__(self):
        self.value = torch.nn.Parameter(torch.ones(2))
        self.optimizer = torch.optim.Adam([self.value], lr=.01)
        self.value.sum().backward(); self.optimizer.step(); self.optimizer.zero_grad()
    def capture(self):
        return self.value.detach().clone(), self.optimizer.state_dict()


class CheckpointTests(unittest.TestCase):
    def test_zero_update_eval_bypasses_uninitialized_training_audit(self):
        tools = ROOT/'tools' if (ROOT/'tools').is_dir() else ROOT
        tree = ast.parse((tools/'run_v3_arm.py').read_text())
        function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'observed_eval')
        calls = []
        namespace = dict(arguments=SimpleNamespace(mode='gate'),
                         original_eval=lambda *a, **k: calls.append((a, k)) or 'actual_eval')
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'observed_eval_cpu', 'exec'), namespace)
        self.assertEqual(namespace['observed_eval'](None, 0, {0}, marker=True), 'actual_eval')
        self.assertEqual(len(calls), 1)

    def fixture(self, root, v3=True):
        cameras = [SimpleNamespace(image_path=f'/input/FRONT/{i}.png', image_name=str(i),
                   colmap_id=i, timestamp=float(i), image_width=2, image_height=2) for i in range(49)]
        scene = SimpleNamespace(train_cameras={1.: cameras}, scale_index=0, resolution_scales=[1.],
                                getTrainCameras=lambda: scene.train_cameras[1.])
        scene.hybrid_sampler = CameraSampler(0, cameras)
        scene.hybrid_sampler.select(cameras, .5, .02)
        args = OmegaConf.create(dict(iterations=30000, position_lr_max_steps=30000,
            position_lr_init=.00016, position_lr_final=1.6e-6, position_lr_delay_mult=.01,
            t_lr_init=.0008, feature_lr=.0025, opacity_lr=.005, scaling_lr=.005,
            scaling_t_lr=.002, velocity_lr=.001, rotation_lr=.001, envmap_lr=.01,
            seed=0, hybrid_method=True, including_dynamic=True, env_map_res=2,
            paper_eval_train_each_checkpoint=True, source_path='/input', model_path=str(root),
            start_checkpoint=None, v3_bundle='/world-road/bundle.json' if v3 else None))
        model, road_args, road = toy()
        road.path, road.exclusions = Path('/world-road/bundle.json'), []
        road.objective_with_renderer(model, road_args, 1, WEIGHTS, toy_render)
        models = {key: Component() for key in ['static', 'dynamic', 'environment']}
        return scene, args, road, models

    def cuda_boundary(self, stack):
        if os.environ.get('V3_CHECKPOINT_TEST_REAL_CUDA') == '1':
            self.assertTrue(torch.cuda.is_available())
            torch.cuda.init()
            return None
        # 只替换硬件接口，strict validator、capture、restore、publisher均原样执行。
        state = torch.get_rng_state().clone()
        for name, value in [('is_initialized', True), ('is_available', True), ('device_count', 1),
                            ('get_rng_state_all', [state])]:
            stack.enter_context(patch.object(torch.cuda, name, return_value=value))
        return stack.enter_context(patch.object(torch.cuda, 'set_rng_state_all'))

    def evaluation(self, root, scene, v3):
        from PIL import Image
        image = root/'frame.png'; Image.new('RGB', (2, 2)).save(image)
        metrics = root/'metrics.json'; metrics.write_text('{}')
        road = root/'road.npz'
        np.savez_compressed(road, rgb_sum=np.zeros((3, 2, 2), np.float32),
                            alpha=np.ones((1, 2, 2), np.float32), depth_sum=np.ones((1, 2, 2), np.float32))
        channels = ['gt', 'composite', 'static', 'dynamic']
        if v3:
            channels += ['road_raw_pre_environment', 'road_surface_color', 'road_alpha']
        frames = []
        for camera in scene.getTrainCameras():
            record = checkpoint.camera_record(camera)
            record['channels'] = {key: str(image) for key in channels}
            if v3:
                record['road_float'] = str(road)
            frames.append(record)
        return [dict(split='train', frames=frames, eval_dir=str(root),
                     metrics=str(metrics), paper_metrics=str(metrics))]

    def test_v3_seven_channels_publishes_four_components_and_strict_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp); scene, args, road, models = self.fixture(root)
            self.cuda_boundary(stack)
            stack.enter_context(patch.object(v3_runtime_context, 'ACTIVE_CONTEXT', SimpleNamespace(road=road)))
            state = checkpoint.capture_training_state(1, scene, [2, 4], models, args, ema={'loss': .2})
            parts = checkpoint.save_components(root, 1, models)
            evaluations = self.evaluation(root, scene, True)
            event = checkpoint.publish_checkpoint_event(root, 1, parts, state, evaluations, args, scene)
            self.assertEqual(len(event['frames']), 49)
            self.assertEqual(len(event['checkpoints']), 4)
            self.assertTrue((root/'checkpoint_events/iteration_000001.json').is_file())
            self.assertFalse((root/'training_state1.unpublished.pth').exists())
            expected = (random.random(), np.random.random(), torch.rand(3),
                        scene.hybrid_sampler.select(scene.getTrainCameras(), .5, .02)[0].image_name)
            road.visits.fill(0); road.reliability.fill_(.1)
            saved_visits = state['v3_world_road_runtime']['visits']
            scene.train_cameras[1.] = list(reversed(scene.train_cameras[1.]))
            restored, ema = checkpoint.restore_training_state(root/'training_state1.pth', 1, scene, models, args)
            self.assertEqual(restored, [2, 4]); self.assertEqual(ema, {'loss': .2})
            self.assertEqual(random.random(), expected[0]); self.assertEqual(np.random.random(), expected[1])
            torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)
            self.assertEqual(scene.hybrid_sampler.select(scene.getTrainCameras(), .5, .02)[0].image_name, expected[3])
            np.testing.assert_array_equal(road.visits, saved_visits)
            torch.testing.assert_close(road.reliability, state['v3_world_road_runtime']['reliability'])

    def test_legacy_four_channel_publication_still_works(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp); scene, args, road, models = self.fixture(root, False)
            self.cuda_boundary(stack)
            state = checkpoint.capture_training_state(1, scene, [], models, args, ema={'loss': .2})
            event = checkpoint.publish_checkpoint_event(root, 1, checkpoint.save_components(root, 1, models),
                                                        state, self.evaluation(root, scene, False), args, scene)
            self.assertEqual(len(event['checkpoints']), 4)

    def test_zero_update_gate_uses_the_real_publisher_and_strict_restore(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp); scene, args, road, models = self.fixture(root)
            self.cuda_boundary(stack)
            args.v3_zero_update_gate = True
            scene.hybrid_sampler = CameraSampler(0, scene.getTrainCameras())
            for model in models.values():
                model.optimizer.state.clear()
            context = SimpleNamespace(road=road, models=list(models.values()),
                                      optimizer_steps=dict.fromkeys(models, 0))
            stack.enter_context(patch.object(v3_runtime_context, 'ACTIVE_CONTEXT', context))
            def evaluate(*unused, **kwargs):
                return self.evaluation(root/'publication_smoke', scene, True)
            receipt = checkpoint.zero_update_publication_smoke(scene, None, args, None, None, None, evaluate)
            self.assertEqual(receipt['sampler_steps'], 0)
            self.assertEqual(receipt['optimizer_updates'], 0)
            event = json.loads(Path(receipt['event']).read_text())
            self.assertTrue(event['diagnostic_only_zero_update_snapshot'])
            self.assertEqual(event['iteration'], 0)

    def test_invalid_assets_preserve_unpublished_state_never_complete_event(self):
        cases = ['missing_base_channel', 'missing_road_channel', 'unknown_channel', 'duplicate_camera',
                 'missing_camera', 'missing_road_float', 'truncated_component', 'empty_image', 'nonfinite_float']
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                root = Path(temp); scene, args, road, models = self.fixture(root)
                self.cuda_boundary(stack)
                stack.enter_context(patch.object(v3_runtime_context, 'ACTIVE_CONTEXT', SimpleNamespace(road=road)))
                state = checkpoint.capture_training_state(1, scene, [], models, args, ema={'loss': .2})
                parts = checkpoint.save_components(root, 1, models); evaluations = self.evaluation(root, scene, True)
                frames = evaluations[0]['frames']
                if case == 'missing_base_channel': frames[0]['channels'].pop('dynamic')
                if case == 'missing_road_channel': frames[0]['channels'].pop('road_alpha')
                if case == 'unknown_channel': frames[0]['channels']['unknown'] = str(root/'frame.png')
                if case == 'duplicate_camera': frames[1]['frame_id'] = frames[0]['frame_id']
                if case == 'missing_camera': frames.pop()
                if case == 'missing_road_float': frames[0]['road_float'] = str(root/'absent.npz')
                if case == 'truncated_component': parts['static']['bytes'] += 1
                if case == 'empty_image': (root/'frame.png').write_bytes(b'')
                if case == 'nonfinite_float':
                    np.savez(root/'road.npz', rgb_sum=np.full((3, 2, 2), np.nan),
                             alpha=np.ones((1, 2, 2)), depth_sum=np.ones((1, 2, 2)))
                with self.assertRaises(ValueError):
                    checkpoint.publish_checkpoint_event(root, 1, parts, state, evaluations, args, scene)
                self.assertTrue((root/'training_state1.unpublished.pth').is_file())
                self.assertFalse((root/'checkpoint_events/iteration_000001.json').exists())
                self.assertFalse((root/'training_state1.pth').exists())

    def test_strict_v3_schema_missing_runtime_is_rejected_without_mocking_validator(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp); scene, args, road, models = self.fixture(root)
            self.cuda_boundary(stack)
            stack.enter_context(patch.object(v3_runtime_context, 'ACTIVE_CONTEXT', SimpleNamespace(road=road)))
            state = checkpoint.capture_training_state(1, scene, [], models, args, ema={'loss': .2})
            state.pop('v3_world_road_runtime')
            with self.assertRaisesRegex(ValueError, 'complete continuation schema'):
                checkpoint._validate_continuation(state, 1, scene, models, args, None)

    def test_published_event_and_existing_state_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/'training_state1.unpublished.pth').write_bytes(b'preserved')
            with self.assertRaises(FileExistsError):
                checkpoint.publish_checkpoint_event(root, 1, {}, {'iteration': 1}, [], {}, None)
            self.assertEqual((root/'training_state1.unpublished.pth').read_bytes(), b'preserved')


if __name__ == '__main__':
    unittest.main()

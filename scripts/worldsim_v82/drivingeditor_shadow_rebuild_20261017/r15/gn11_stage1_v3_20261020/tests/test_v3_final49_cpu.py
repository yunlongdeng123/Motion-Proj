"""CPU admission only; actual native49 replay remains required after training."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('final49', ROOT/'tools/evaluate_v3_final49.py')
final49 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(final49)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def campaign_fixture(root):
    save(root/'four_arm_registry.json', dict(status='FOUR_ARMS_TRAINING_COMPLETE_EVALUATION_PENDING',
        max_concurrent_training=1, completed_arms=list(final49.ARMS)))
    for arm in final49.ARMS:
        run = root/'arms'/arm/'run30k'
        save(run/'status.json', dict(status='complete', iteration=30000, arm=arm,
            second_stage_updates=0,
            optimizer_step_counts=dict.fromkeys(('static', 'dynamic', 'environment'), 30000)))
        dummy = run/'fixture.bin'
        dummy.write_bytes(b'x')
        frames = [dict(image_name=f, frame_id='image_0/'+f, width=480, height=320,
            channels=dict.fromkeys(final49.CHANNELS, str(dummy)), road_float=str(dummy),
            hybrid_metrics=str(dummy)) for f in final49.FRAMES]
        for step in final49.STEPS:
            save(run/'checkpoint_events'/f'iteration_{step:06d}.json', dict(status='complete',
                iteration=step, checkpoints={k: dict(path=str(dummy), bytes=1) for k in (
                    'static', 'dynamic', 'environment', 'training_state')}, frames=frames))


class Final49CPU(unittest.TestCase):
    def test_complete_contract_is_cpu_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_fixture(root)
            with patch.object(torch.cuda, 'is_available', side_effect=AssertionError('no CUDA')):
                self.assertEqual(set(final49.require_complete(root)), set(final49.ARMS))

    def test_partial_and_zero_update_gates_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_fixture(root)
            path = root/'arms/ownership_only/run30k/status.json'
            base = final49.read(path)
            for changes in [dict(status='zero_update_gate_passed', iteration=0),
                            dict(optimizer_step_counts=dict.fromkeys(('static', 'dynamic', 'environment'), 29900)),
                            dict(second_stage_updates=1)]:
                save(path, dict(base, **changes))
                with self.assertRaises(RuntimeError):final49.require_complete(root)

    def test_missing_node_and_frame_identity_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_fixture(root)
            path = root/'arms/full/run30k/checkpoint_events/iteration_005000.json'
            value = final49.read(path)
            value['frames'][0]['image_name'] = '0008198'
            save(path, value)
            with self.assertRaises(RuntimeError):final49.require_complete(root)
            campaign_fixture(root)
            path.unlink()
            with self.assertRaises(FileNotFoundError):final49.require_complete(root)

    def test_checkpoint_escape_or_wrong_size_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_fixture(root)
            path = root/'arms/full/run30k/checkpoint_events/iteration_005000.json'
            value = final49.read(path)
            value['checkpoints']['static']['bytes'] = 2
            save(path, value)
            with self.assertRaises(RuntimeError):final49.require_complete(root)
            value['checkpoints']['static'] = dict(path=str(root/'four_arm_registry.json'),
                                                  bytes=(root/'four_arm_registry.json').stat().st_size)
            save(path, value)
            with self.assertRaises(RuntimeError):final49.require_complete(root)

    def test_loader_transform_is_verified_without_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'transform_matrix.txt'
            np.savetxt(path, np.eye(4))
            before = final49.file_state([path])
            original = np.savetxt
            with final49.preserve_transform_file(path):
                np.savetxt(path, np.eye(4))
                with self.assertRaises(RuntimeError):np.savetxt(path, np.eye(4)*2)
                with self.assertRaises(RuntimeError):np.savetxt(path.parent/'other', np.eye(4))
            self.assertIs(np.savetxt, original)
            self.assertEqual(before, final49.file_state([path]))

    def test_replay_checks_raw_not_display_clamped(self):
        actual = torch.tensor([[[1.2, 0.]]])
        self.assertEqual(final49.assert_close(actual, actual.numpy(), 'raw'), 0.)
        with self.assertRaises(RuntimeError):final49.assert_close(actual, actual.clamp(0, 1), 'clipped')
        with self.assertRaises(RuntimeError):final49.assert_close(actual, actual+float('nan'), 'nonfinite')

    def test_projection_uses_native_half_pixel_convention(self):
        from types import SimpleNamespace
        camera = SimpleNamespace(full_proj_transform=torch.eye(4), image_width=4, image_height=4)
        uv, valid = final49.raster_uv(torch.tensor([[0., 0., 0.], [2., 0., 0.]]), camera)
        self.assertTrue(torch.equal(uv[0], torch.tensor([1.5, 1.5])))
        self.assertEqual(valid.tolist(), [True, False])

    def test_cli_dry_run_creates_no_receipt_or_gpu_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_fixture(root)
            output = root/'final49/full'
            with patch.object(sys, 'argv', ['evaluate_v3_final49.py', '--campaign', str(root),
                    '--arm', 'full', '--output', str(output)]), \
                    patch.object(final49.subprocess, 'check_output', side_effect=AssertionError('no GPU')), \
                    contextlib.redirect_stdout(io.StringIO()) as stream:
                final49.main()
            self.assertIn('CPU_PREFLIGHT_ONLY_NOT_RENDERED', stream.getvalue())
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()

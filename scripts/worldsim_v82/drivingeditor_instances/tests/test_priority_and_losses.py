import json
import subprocess
import tempfile
from pathlib import Path
import unittest
import torch
from scripts.worldsim_v82.drivingeditor_instances.priority import inspect_priority
from scripts.worldsim_v82.drivingeditor_instances.losses import visible_instance_loss, supervision_contract


class PriorityLossChecks(unittest.TestCase):
    def test_street_preparation_blocks_even_without_cuda(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid = Path(tmp) / '123'
            pid.mkdir()
            (pid / 'cmdline').write_bytes(b'/env/python\0/root/autodl-tmp/streetcrafter_work/download.py\0')
            def query(*args, **kwargs):
                return subprocess.CompletedProcess(args, 0, '', '')
            result = inspect_priority(tmp, query, own_pid=456)
            self.assertFalse(result['allowed'])
            self.assertEqual(result['streetcrafter_processes'][0]['pid'], 123)

    def test_query_failure_is_not_empty_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            def query(*args, **kwargs):
                raise FileNotFoundError('missing nvidia-smi')
            self.assertFalse(inspect_priority(tmp, query)['allowed'])

    def test_unknown_mask_has_no_false_zero_gradient(self):
        contribution = torch.tensor([[[.5]], [[.8]]], requires_grad=True)
        loss = visible_instance_loss(contribution, torch.tensor([[[1]], [[0]]]), [True, False])
        loss.backward()
        self.assertEqual(contribution.grad[1].item(), 0)
        self.assertLess(contribution.grad[0].item(), 0)

    def test_known_outside_has_gradient_and_joint_needs_masks(self):
        value = torch.tensor([[[.5, .6]]], requires_grad=True)
        visible_instance_loss(value, torch.tensor([[[1, 0]]]), [True]).backward()
        self.assertGreater(value.grad[0, 0, 1].item(), 0)
        with self.assertRaises(ValueError):
            supervision_contract([True, False], freeze_background=False)
        self.assertTrue(supervision_contract([True, False], freeze_background=True)['background_frozen'])


if __name__ == '__main__':
    unittest.main()

import unittest
from unittest.mock import patch
import runtime

POLICY = {'schema': 'preserve_joint_background_v2',
          'initialization': 'preserve_all_GN11_static_Gaussians',
          'freeze_environment': True, 'learning_rate_scale': .1,
          'expected_initial_points': 646337}

class Static:
    get_xyz = range(646337)
    def prune_points(self, mask):
        raise AssertionError('repair must never invoke center-box prune')

class RepairTests(unittest.TestCase):
    def test_preserves_every_pretrained_point(self):
        with patch.object(runtime, 'remove_vehicle_background', side_effect=AssertionError('legacy pruning called')):
            result = runtime.initialize_background(Static(), object(), {'background_repair': POLICY})
        self.assertEqual(result['removed_points'], 0)
        self.assertEqual(result['remaining_points'], 646337)

    def test_rejects_wrong_prior(self):
        with self.assertRaises(ValueError):
            runtime.initialize_background(type('S', (), {'get_xyz': range(613405)})(), None, {'background_repair': POLICY})

    def test_invalid_scale_fails(self):
        with self.assertRaises(ValueError):
            runtime.repair_policy({'background_repair': dict(POLICY, learning_rate_scale=0)})

    def test_missing_policy_never_falls_back_to_legacy_pruning(self):
        with patch.object(runtime, 'remove_vehicle_background', side_effect=AssertionError('legacy pruning called')):
            with self.assertRaisesRegex(ValueError, 'explicit preserve_joint_background_v2'):
                runtime.initialize_background(Static(), None, {})

if __name__ == '__main__':
    unittest.main()

"""真实显存OOM后的单槽准入和完成/失败账目回归；不使用CUDA。"""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT/'tools' if (ROOT/'tools').exists() else ROOT
sys.path.insert(0, str(TOOLS))
from v3_resource_recovery import RESOURCE_RECOVERY_POLICY, validate_resource_recovery, exact_30k, failed_updates
import check_v3_progress


class ResourceRecoveryTests(unittest.TestCase):
    def valid(self):
        return dict(resource_recovery_policy=copy.deepcopy(RESOURCE_RECOVERY_POLICY),
                    max_concurrent_training=1, arm_order=['full','real_only','prior_only','ownership_only'])

    def test_single_slot_policy_needs_exact_plan_and_approval_without_stale_pairs(self):
        plan = self.valid(); approval = self.valid()
        validate_resource_recovery(plan, approval)
        for change in [dict(max_concurrent_training=2), dict(pair_policy={}), dict(resource_recovery_policy={})]:
            bad = dict(plan, **change)
            with self.assertRaises(RuntimeError): validate_resource_recovery(bad, approval)
            with self.assertRaises(RuntimeError): validate_resource_recovery(plan, dict(approval, **change))

    def test_terminal_and_failure_use_actual_three_Adam_counts_not_stale_status_step(self):
        state = dict(status='complete', iteration=30000,
                     optimizer_step_counts=dict(static=30000,dynamic=30000,environment=30000))
        self.assertTrue(exact_30k(state))
        state['optimizer_step_counts']['environment'] = 29999
        self.assertFalse(exact_30k(state))
        failure = dict(iteration=7700, optimizer_step_counts=dict(static=7752,dynamic=7752,environment=7752))
        self.assertEqual(failed_updates(failure), 7752)
        failure['optimizer_step_counts']['static'] = 7751
        with self.assertRaises(ValueError): failed_updates(failure)

    def test_single_slot_snapshot_does_not_require_pending_peer_to_be_running(self):
        with tempfile.TemporaryDirectory() as temp:
            camp = Path(temp)
            plan = dict(self.valid(),method_version='test')
            (camp/'plan.json').write_text(json.dumps(plan))
            registry = dict(status='RUNNING_SEQUENTIALLY',current_arm='full',
                            logical_batches=RESOURCE_RECOVERY_POLICY['logical_batches'])
            (camp/'four_arm_registry.json').write_text(json.dumps(registry))
            run = camp/'arms/full/run30k'; run.mkdir(parents=True)
            (run/'status.json').write_text(json.dumps(dict(status='training',pid=10,iteration=5000,
                loss=.1,psnr=20.,optimizer_step_counts=dict(static=5000,dynamic=5000,environment=5000))))
            (run/'v3_capacity.jsonl').write_text(json.dumps(dict(fixed_road_ids_preserved=True,fixed_road=14034))+'\n')
            argv=['python',str(camp/'run_v3_arm.py'),'--plan',str(camp/'plan.json'),
                  '--arm','full','--mode','train','--output',str(run)]
            with patch.object(check_v3_progress.subprocess,'check_output',return_value='1000,32760,99'), \
                 patch.object(check_v3_progress,'process',return_value=dict(pid=10,argv=argv)), \
                 patch.object(check_v3_progress.shutil,'disk_usage',return_value=SimpleNamespace(free=50*1024**3)):
                state = check_v3_progress.snapshot(camp)
            self.assertTrue(state['healthy_now'])
            self.assertEqual(state['active_training_arms'], ['full'])
            self.assertEqual(state['current_logical_batch'], ['full','real_only'])
            self.assertFalse(state['both_current_arms_have_5k_node'])
            self.assertIsNone(state['arms']['real_only']['state'])


if __name__ == '__main__':
    unittest.main()

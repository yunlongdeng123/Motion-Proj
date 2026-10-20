"""2+2进程归属/审批/锁接管回归，纯CPU，不启动训练。"""

import copy
import fcntl
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT/'tools' if (ROOT/'tools').exists() else ROOT
sys.path.insert(0, str(TOOLS))
from v3_pair_runtime import PAIR_POLICY, validate_pair_plan, validate_peer_admission, matches_worker
from run_v3_pairs import completed, require_fresh_campaign


class PairTests(unittest.TestCase):
    def test_fresh_restart_requires_authority_and_new_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp).resolve()
            approval = dict(restart_policy='user_authorized_fresh_after_unresumable_publication_failure',
                            preserved_failed_campaign='/preserved-old-campaign')
            require_fresh_campaign(campaign, approval)
            with self.assertRaises(RuntimeError):
                require_fresh_campaign(campaign, {})
            bad = dict(approval, preserved_failed_campaign=str(campaign))
            with self.assertRaises(RuntimeError):
                require_fresh_campaign(campaign, bad)
            output = campaign/'arms/full/run30k'; output.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                require_fresh_campaign(campaign, approval)

    def test_2plus2_requires_exact_policy_in_both_plan_and_human_approval(self):
        valid = dict(pair_policy=copy.deepcopy(PAIR_POLICY), max_concurrent_training=2)
        validate_pair_plan(valid, valid)
        for field in ['pair_policy', 'max_concurrent_training']:
            bad = copy.deepcopy(valid); bad.pop(field)
            with self.assertRaises(RuntimeError):
                validate_pair_plan(valid, bad)

    def fixture(self):
        camp = Path('/campaign')
        owner = dict(pid=99, start_ticks=9, argv=['python', str(camp/'run_v3_pairs.py')])
        full = dict(pid=10, start_ticks=1, argv=['python', str(camp/'run_v3_arm.py'),
            '--plan', str(camp/'plan.json'), '--arm', 'full', '--mode', 'train', '--output', str(camp/'arms/full/run30k')])
        lease = dict(coordinator_pid=99, coordinator_start_ticks=9, batch_index=0,
            global_guard=dict(pid=10, start_ticks=1), slots={'full':dict(pid=10, start_ticks=1, phase='train'),
                                                          'real_only':dict(pid=None, phase='gate')})
        return camp, lease, {99:owner, 10:full}

    def test_join_only_with_real_parent_and_exact_adopted_full_identity(self):
        camp, lease, table = self.fixture()
        self.assertEqual(validate_peer_admission(lease, camp, 'real_only', 'gate', 11, 99, table.get), {10})
        for mutate in [lambda x:x.update(coordinator_start_ticks=8),
                       lambda x:x['slots'].update(prior_only=dict(pid=None,phase='gate')),
                       lambda x:x['global_guard'].update(start_ticks=8)]:
            bad = copy.deepcopy(lease); mutate(bad)
            with self.assertRaises(RuntimeError):
                validate_peer_admission(bad, camp, 'real_only', 'gate', 11, 99, table.get)

    def test_pid_reuse_wrong_phase_and_out_of_batch_peer_fail_closed(self):
        camp, lease, table = self.fixture()
        self.assertTrue(matches_worker(table[10], camp, 'full', 'train'))
        self.assertFalse(matches_worker(table[10], camp, 'full', 'gate'))
        with self.assertRaises(RuntimeError):
            validate_peer_admission(lease, camp, 'real_only', 'train', 11, 99, table.get)
        table[10]['start_ticks'] = 2
        with self.assertRaises(RuntimeError):
            validate_peer_admission(lease, camp, 'real_only', 'gate', 11, 99, table.get)

    def test_only_exact_finished30k_is_batch_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = dict(status='complete', iteration=30000, optimizer_step_counts=dict(static=30000,dynamic=30000,environment=30000))
            (root/'status.json').write_text(json.dumps(state)); self.assertTrue(completed(root))
            state['optimizer_step_counts']['environment'] = 29999
            (root/'status.json').write_text(json.dumps(state)); self.assertFalse(completed(root))

    def test_kernel_waiter_inherits_global_lock_without_recreating_lockfile(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'gpu.lock'
            parent, child = multiprocessing.Pipe()
            def holder():
                with path.open('a+') as handle:
                    fcntl.flock(handle, fcntl.LOCK_EX)
                    child.send('locked'); child.recv()
            worker = multiprocessing.get_context('fork').Process(target=holder); worker.start()
            self.assertEqual(parent.recv(), 'locked')
            ready = threading.Event()
            with path.open('a+') as handle:
                def waiter():
                    fcntl.flock(handle, fcntl.LOCK_EX); ready.set()
                thread = threading.Thread(target=waiter); thread.start()
                self.assertFalse(ready.wait(.05))
                parent.send('release'); worker.join(3)
                self.assertTrue(ready.wait(3)); thread.join()
            self.assertEqual(worker.exitcode, 0)


if __name__ == '__main__':
    unittest.main()

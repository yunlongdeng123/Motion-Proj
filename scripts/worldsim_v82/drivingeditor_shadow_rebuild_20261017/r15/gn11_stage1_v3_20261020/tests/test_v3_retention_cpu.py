"""Temporary fixtures only; no production data or CUDA touched by tests."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import evaluate_v3_final49 as final49
import prune_v3_completed_checkpoints as prune
sys.path.pop(0)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture(root):
    save(root/'four_arm_registry.json', dict(status='FOUR_ARMS_TRAINING_COMPLETE_EVALUATION_PENDING',
        max_concurrent_training=1, completed_arms=list(final49.ARMS)))
    for arm in final49.ARMS:
        run = root/'arms'/arm/'run30k'
        save(run/'status.json', dict(status='complete', iteration=30000, arm=arm,
            second_stage_updates=0,
            optimizer_step_counts=dict.fromkeys(('static', 'dynamic', 'environment'), 30000)))
        dummy = run/'evaluation.bin'
        dummy.write_bytes(b'eval')
        frames = [dict(image_name=f, frame_id='image_0/'+f, width=480, height=320,
            channels=dict.fromkeys(final49.CHANNELS, str(dummy)), road_float=str(dummy),
            hybrid_metrics=str(dummy)) for f in final49.FRAMES]
        for step in final49.STEPS:
            components = {}
            for name, prefix in final49.COMPONENT_PREFIXES.items():
                path = run/f'{prefix}{step}.pth'
                path.write_bytes(b'x')
                components[name] = dict(path=str(path), bytes=1)
            save(run/'checkpoint_events'/f'iteration_{step:06d}.json',
                 dict(status='complete', iteration=step, checkpoints=components, frames=frames))


class RetentionCPU(unittest.TestCase):
    def execute_fixture(self, root):
        plan = prune.make_plan(root)
        with patch.object(prune, 'EXPECTED_CAMPAIGN', root), patch.object(prune, 'process_guard'):
            return prune.execute(plan)

    def test_plan_is_readonly_exact40_keep8(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve();fixture(root)
            plan = prune.make_plan(root)
            self.assertEqual(len(plan['removed']), 40)
            self.assertEqual(len(plan['kept']), 8)
            self.assertEqual(plan['reclaim_bytes'], 40)
            self.assertTrue(all(Path(x['path']).is_file() for x in plan['removed']))
            self.assertFalse((root/'checkpoint_retention.json').exists())

    def test_execute_preserves_latest_eval_and_both_other_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve();fixture(root)
            other = {str(p):p.read_bytes() for arm in ('prior_only','ownership_only')
                     for p in (root/'arms'/arm).rglob('*') if p.is_file()}
            result = self.execute_fixture(root)
            self.assertEqual(result['status'], 'complete')
            self.assertTrue(all(not Path(x['path']).exists() for x in result['removed']))
            self.assertTrue(all(Path(x['path']).is_file() for x in result['kept']))
            self.assertTrue(all(Path(p).read_bytes()==v for p,v in other.items()))
            self.assertEqual(len(final49.retention_removals(root)), 40)
            self.assertEqual(set(final49.require_complete(root)), set(final49.ARMS))

    def test_no_receipt_missing_file_is_still_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve();fixture(root)
            (root/'arms/full/run30k/chkpnt5000.pth').unlink()
            with self.assertRaises(RuntimeError):final49.require_complete(root)

    def test_latest_and_other_arm_never_exempted(self):
        for arm, step in [('full',30000),('prior_only',5000)]:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve();fixture(root);self.execute_fixture(root)
                (root/f'arms/{arm}/run30k/chkpnt{step}.pth').unlink()
                with self.assertRaises(RuntimeError):final49.require_complete(root)

    def test_tampered_event_receipt_and_missing_eval_rejected(self):
        for case in ['event','scope','incomplete','protection','eval']:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve();fixture(root);self.execute_fixture(root)
                receipt = root/'checkpoint_retention.json'
                data = final49.read(receipt)
                if case=='event':
                    event = root/'arms/full/run30k/checkpoint_events/iteration_005000.json'
                    event.write_text(event.read_text()+' ')
                elif case=='scope':
                    data['removed'][0]['arm']='prior_only';save(receipt,data)
                elif case=='incomplete':
                    data['status']='planned';save(receipt,data)
                elif case=='protection':
                    data['latest30k_sha256_unchanged']=False;save(receipt,data)
                else:
                    (root/'arms/full/run30k/evaluation.bin').unlink()
                with self.assertRaises(RuntimeError):final49.require_complete(root)

    def test_partial_training_symlink_and_changed_target_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve();fixture(root)
            state = root/'arms/full/run30k/status.json'
            value = final49.read(state);value['iteration']=29900;save(state,value)
            with self.assertRaises(RuntimeError):prune.make_plan(root)
            fixture(root)
            path = root/'arms/full/run30k/chkpnt5000.pth';path.unlink()
            path.symlink_to(root/'arms/prior_only/run30k/chkpnt5000.pth')
            with self.assertRaises(RuntimeError):prune.make_plan(root)
            path.unlink();fixture(root);plan=prune.make_plan(root);path.write_bytes(b'changed')
            with patch.object(prune,'EXPECTED_CAMPAIGN',root), patch.object(prune,'process_guard'):
                with self.assertRaises(RuntimeError):prune.execute(plan)
            self.assertFalse((root/'checkpoint_retention.intent.json').exists())

    def test_live_process_and_wrong_campaign_block_deletion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();fixture(root);plan=prune.make_plan(root)
            with self.assertRaises(RuntimeError):prune.execute(plan)
            with patch.object(prune,'EXPECTED_CAMPAIGN',root), \
                    patch.object(prune,'process_guard',side_effect=RuntimeError('live worker')):
                with self.assertRaises(RuntimeError):prune.execute(plan)
            self.assertTrue(all(Path(x['path']).exists() for x in plan['removed']))


if __name__ == '__main__':
    unittest.main()

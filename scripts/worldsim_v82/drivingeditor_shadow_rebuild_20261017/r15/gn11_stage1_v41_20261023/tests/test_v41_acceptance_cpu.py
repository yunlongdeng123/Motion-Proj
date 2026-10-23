"""Deletion gains must be paired and retain the unknown full-target denominator."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from v41_acceptance import paired_deleted_metrics, deletion_summary, CONTRACT
from audit_v41_coverage import evidence_funnel


class PairedDeletionTests(unittest.TestCase):
    def metric(self, before=.1, after=.3, labelled=2):
        target = np.ones((2, 5), bool)
        domain = np.zeros((2, 5), bool); domain.ravel()[:labelled] = True
        return paired_deleted_metrics(np.full((3,2,5),before), np.full((3,2,5),after),
            np.zeros((3,2,5)), target, domain, np.ones((2,5)))

    def test_high_contribution_can_still_worsen_static(self):
        r = self.metric()
        self.assertAlmostEqual(r['delta']['mae'], .2)
        self.assertEqual(r['pixel_change']['degraded_fraction_labelled'], 1)
        self.assertEqual(r['pixel_change']['degraded_fraction_target'], .2)
        self.assertEqual(r['unknown_pixels'], 8)
        self.assertEqual(r['road_contribution_full_target'], 1.)
        self.assertEqual(r['thresholds']['0.25']['effective_full_target']['coverage_fraction'], 1)
        self.assertEqual(r['thresholds']['0.25']['verified_color_full_target']['coverage_fraction'], 0)

    def test_correct_subset_does_not_become_full_coverage(self):
        r = self.metric(.1, 0)
        self.assertEqual(r['thresholds']['0.25']['verified_color_full_target']['coverage_fraction'], .2)
        self.assertEqual(r['thresholds']['0.25']['verified_color_labelled']['coverage_fraction'], 1)

    def test_no_label_is_unknown_not_zero_error(self):
        r = self.metric(labelled=0)
        self.assertIsNone(r['after']['mae'])
        self.assertIsNone(r['delta']['mse'])
        self.assertIsNone(r['pixel_change']['improved_fraction_labelled'])
        self.assertEqual(r['target_pixels'], 10)

    def test_clamp_compatibility_and_raw_retained(self):
        r = self.metric(1,1.2)
        self.assertEqual(r['delta']['mae'], 0)
        self.assertAlmostEqual(r['raw']['delta']['mae'], .2)

    def test_pixel_change_and_worst_frame_are_actor_specific(self):
        records = []
        for name,b,a,n in [('good',.4,.1,8), ('bad',.1,.3,2), ('unknown',.1,.2,0)]:
            actors = {str(actor):self.metric(b,a,n) for actor in CONTRACT['vehicle_ids']}
            actors['6'] = self.metric(labelled=0)
            records.append(dict(frame=name,crossview_deleted=dict(applicable=True,actors=actors)))
        result = deletion_summary(records)['actors']
        self.assertEqual(result['37']['worst_mae_delta']['frame'],'bad')
        self.assertAlmostEqual(result['37']['pixel_weighted']['delta']['mae'], -.2)
        self.assertEqual(result['37']['target_pixel_frames'], 30)
        self.assertEqual(result['37']['zero_label_frames'], ['unknown'])
        self.assertEqual(result['6']['labelled_frames'], 0)
        self.assertIsNone(result['6']['worst_mae_delta'])

    def test_labels_outside_target_rejected(self):
        with self.assertRaises(ValueError):
            paired_deleted_metrics(*[np.zeros((3,2,5))]*3, np.zeros((2,5),bool),
                np.ones((2,5),bool),np.zeros((2,5)))

    def test_evidence_funnel_partitions_every_target_pixel(self):
        r = evidence_funnel(np.ones(5,bool), np.array([0,1,1,1,1],bool),
            np.array([0,0,1,1,1],bool), np.array([0,0,0,1,1],bool),np.array([0,0,0,0,1],bool))
        self.assertEqual(r['target_pixels'],5)
        self.assertEqual(list(r.values())[1:],[1]*5)


if __name__ == '__main__': unittest.main(verbosity=2)

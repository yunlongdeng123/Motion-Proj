"""No color leakage or convenient pair selection in the incremental source rule."""
from pathlib import Path
import sys
import unittest
import numpy as np
p=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(p/('engine' if (p/'engine').exists() else 'engine_overlay')))
from utils.v41_sources import split_sources,source_consensus,validate_label_sources


def colors():
    x=np.full((49,1,3),np.nan,np.float32)
    x[[0,4,8,14,18]]=.3
    return x


class SourceTests(unittest.TestCase):
    def test_disjoint_blocks_and_boundary_guard(self):
        f,v=split_sources(45)
        self.assertFalse((f&v).any() or f[45] or v[45])
        self.assertFalse(f[[13,21,34,42]].any())
        self.assertTrue(v[[14,20,35,41]].all())

    def test_holdout_colors_never_change_training_color(self):
        a=colors();b=a.copy();b[[14,18]]=.8
        good,bad=source_consensus(a,45),source_consensus(b,45)
        np.testing.assert_array_equal(good['target'],bad['target'])
        self.assertTrue(good['accepted'][0]);self.assertFalse(bad['accepted'][0])

    def test_two_matching_sources_cannot_bypass_majority(self):
        x=colors();x[[0,4]]=.2;x[8]=.6;x[[9,10]]=.9
        self.assertFalse(source_consensus(x,45)['fit_ok'][0])

    def test_three_adjacent_sources_do_not_count_as_spread(self):
        x=np.full((49,1,3),np.nan,np.float32);x[[0,1,2,14,18]]=.3
        self.assertFalse(source_consensus(x,45)['fit_ok'][0])

    def test_missing_validation_and_target_rgb_are_not_accepted(self):
        x=colors();x[[14,18]]=np.nan
        self.assertFalse(source_consensus(x,45)['accepted'][0])
        x[45]=.3
        with self.assertRaises(ValueError):source_consensus(x,45)

    def test_zero_candidates_and_unknown_remain_empty(self):
        result=source_consensus(np.empty((49,0,3),np.float32),45)
        self.assertEqual(result['accepted'].size,0)
        result=source_consensus(np.full((49,2,3),np.nan,np.float32),45)
        self.assertFalse(result['accepted'].any())

    def test_loader_rejects_fit_validation_leakage(self):
        row=dict(deletion_domains=np.ones((5,1,1),bool),deletion_label_kind=np.full((1,1),2),
            deletion_legacy_valid=np.zeros((1,1),bool),deletion_source_ids=np.array([[[0,4]]]),
            deletion_validation_source_ids=np.array([[[14,18]]]),deletion_target_rgb=np.ones((3,1,1))*.3)
        for k,v in dict(fit_inliers=3,fit_count=3,fit_span=8,validation_inliers=2,validation_count=2).items():
            row['deletion_'+k]=np.full((1,1),v)
        validate_label_sources(row,45)
        row['deletion_validation_source_ids'][0,0]=[0,4]
        with self.assertRaises(ValueError):validate_label_sources(row,45)


if __name__=='__main__':unittest.main(verbosity=2)

import unittest
import numpy as np
from admission import decision
from donor_gate import check_masks,load_policy

class AdmissionTests(unittest.TestCase):
    def test_ai_two_does_not_promote(self):
        self.assertEqual(decision(2,'new'),'ai_2_candidate_not_qualified')
        self.assertEqual(decision(1,'new'),'ai_low_score_reject')
    def test_human_failure_overrides_ai_two(self):
        self.assertEqual(decision(2,'old','reject_visible_contamination'),'human_reject')
        self.assertEqual(decision(2,'old',blocked_instances=['old']),'donor_blocked')
    def test_bottom_mask_is_not_fixed_by_connectedness(self):
        m=np.zeros((576,1024),bool);m[330:532,420:650]=True
        r=check_masks('new',[m]);self.assertFalse(r['eligible_for_pairing']);self.assertIn('source_bottom_or_ego_ambiguity',r['reasons'])
    def test_disconnected_pixels_hold_without_silent_cleanup(self):
        m=np.zeros((576,1024),bool);m[230:400,420:600]=True;m[420:440,420:440]=True;copy=m.copy()
        r=check_masks('new',[m]);self.assertFalse(r['eligible_for_pairing']);self.assertTrue(np.array_equal(copy,m))
    def test_clean_control_and_instance_block(self):
        m=np.zeros((576,1024),bool);m[230:400,420:600]=True
        self.assertTrue(check_masks('new',[m])['eligible_for_pairing'])
        self.assertFalse(check_masks(load_policy()['blocked_instances'][0],[m])['eligible_for_pairing'])

if __name__=='__main__':unittest.main()

"""防止未独立检查/错来源的保护mask进入精确遮挡评估。"""
import unittest
from admit_exact import review_blockers
class AdmissionTests(unittest.TestCase):
    def setUp(self):self.job={'job_id':'C010_token','source_id':'C010','instance_token':'token'}
    def test_absent_review_blocks(self):self.assertTrue(review_blockers(self.job,None))
    def test_uncertain_not_pass(self):self.assertTrue(review_blockers(self.job,self.job|{'mask_status':'uncertain'}))
    def test_other_track_review_not_reusable(self):self.assertTrue(review_blockers(self.job,self.job|{'instance_token':'other','mask_status':'pass'}))
    def test_matching_pass_allows_next_gate(self):self.assertEqual(review_blockers(self.job,self.job|{'mask_status':'pass'}),[])
if __name__=='__main__':unittest.main()

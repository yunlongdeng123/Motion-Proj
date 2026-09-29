import unittest
import numpy as np
from data_contract import masked_condition_from_x,assert_pair_pixels,admitted_to_next_stage

class DataContractTests(unittest.TestCase):
    def setUp(self):
        self.y=np.random.default_rng(770129).integers(0,256,(3,16,24,3),dtype=np.uint8)
        self.h=np.zeros((3,16,24),bool);self.h[:,4:12,5:17]=True
        self.x=self.y.copy();self.x[self.h]=[100,40,10]
    def test_hidden_gt_does_not_change_model_condition(self):
        before=masked_condition_from_x(self.x,self.h)
        altered_gt=self.y.copy();altered_gt[self.h]=255
        self.assertFalse(np.array_equal(altered_gt,self.y))
        # 条件接口只接受X/H；监督更改不能影响任何条件。
        np.testing.assert_array_equal(before,masked_condition_from_x(self.x,self.h))
    def test_wholly_masked_donor_is_not_visible_condition(self):
        self.assertTrue(assert_pair_pixels(self.y,self.x,self.h,self.h))
        np.testing.assert_array_equal(masked_condition_from_x(self.y,self.h),masked_condition_from_x(self.x,self.h))
    def test_edge_escape_is_rejected(self):
        x=self.x.copy();x[:,0,0]=255
        with self.assertRaisesRegex(ValueError,'undeclared'):assert_pair_pixels(self.y,x,self.h,self.h)
        smaller=self.h.copy();smaller[:,4,5]=False
        with self.assertRaisesRegex(ValueError,'outside_deletion'):assert_pair_pixels(self.y,self.x,self.h,smaller)
    def test_all_frames_and_both_prior_gates_required(self):
        self.assertFalse(admitted_to_next_stage(True,True,{0:1,1:1},3))
        self.assertFalse(admitted_to_next_stage(True,True,{0:1,1:0,2:1},3))
        self.assertFalse(admitted_to_next_stage(False,True,{0:1,1:1,2:1},3))
        self.assertFalse(admitted_to_next_stage(True,False,{0:1,1:1,2:1},3))
        self.assertTrue(admitted_to_next_stage(True,True,{0:1,1:1,2:1},3))
if __name__=='__main__':unittest.main()

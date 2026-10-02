"""首因顺序不应改变实例补标的候选集；质量标签不可混入方法目录。"""
import unittest
import numpy as np
from recover_instance_audit import unknown_hits,candidate_key


class SelectionContract(unittest.TestCase):
    def test_order_and_later_frames(self):
        h=np.zeros((576,1024),bool);h[100:130,100:140]=True
        def ob(tok,cat):return {'instance_token':tok,'category':cat,'_projection':{'box':np.array([100,100,140,130])}}
        barrier=ob('barrier','movable_object.barrier');known=ob('known','vehicle.car');unknown=ob('unknown','vehicle.car')
        a=[[barrier,known],[barrier,unknown,known]]
        b=[list(reversed(x)) for x in a]
        expected={'unknown':{'category':'vehicle.car','frames':[1]}}
        self.assertEqual(unknown_hits([h,h],a,{'known':None}),expected)
        self.assertEqual(unknown_hits([h,h],b,{'known':None}),expected)

    def test_blank_or_nonoverlap_is_not_candidate(self):
        h=np.zeros((576,1024),bool);h[100:130,100:140]=True
        obs=[{'instance_token':'ego','category':'ego','_projection':None},
             {'instance_token':'far','category':'vehicle.car','_projection':{'box':np.array([500,100,540,130])}}]
        self.assertEqual(unknown_hits([h],[obs],{}),{})

    def test_identity_not_first_rejection(self):
        tr={'lane_token':'lane','lane_midpoint_distance_m':12.5,'speed_mps':3.0}
        self.assertEqual(candidate_key('s',tr),('s','lane',12.5,3.0))
        self.assertNotEqual(candidate_key('s',tr),candidate_key('t',tr))


if __name__=='__main__':unittest.main()

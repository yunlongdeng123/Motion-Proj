import copy,unittest
import numpy as np
from delete_full_common import apply_delete
from delete_full_query import compose_actor

class DeleteContract(unittest.TestCase):
    def test_only_target_visibility_changes(self):
        s={'background':[{'asset':'B_t.npz'}],'actors':[{'actor_id':'22','visible':True},{'actor_id':'2','visible':True}]};before=copy.deepcopy(s)
        e=apply_delete(s,22);self.assertEqual(s,before);self.assertEqual(e['background'],s['background']);self.assertTrue(e['actors'][1]['visible']);self.assertFalse(e['actors'][0]['visible'])
    def test_unknown_duplicate_and_repeated_delete_rejected(self):
        for actors in [[],[{'actor_id':'22','visible':False}],[{'actor_id':'22','visible':True}]*2]:
            with self.assertRaises(ValueError):apply_delete({'background':[],'actors':actors},22)
    def test_depth_occlusion_and_hole(self):
        bg=np.zeros((1,4,3),dtype='uint8');rgba=np.full((1,4,4),255,dtype='uint8');rgba[0,3,3]=0
        az=np.full((1,4),5.);bz=np.array([[3.,6.,np.inf,6.]])
        rgb,m=compose_actor(bg,rgba,az,bz)
        self.assertEqual(rgb[0,:,0].tolist(),[0,255,255,0]);self.assertEqual(m['occluded_asset_pixels'],1)
    def test_invalid_actor_depth_not_rendered(self):
        bg=np.zeros((1,2,3),dtype='uint8');rgba=np.full((1,2,4),255,dtype='uint8')
        rgb,_=compose_actor(bg,rgba,np.array([[np.inf,-1.]]),np.full((1,2),np.inf));self.assertFalse(rgb.any())

if __name__=='__main__':unittest.main()

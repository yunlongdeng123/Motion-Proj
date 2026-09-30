"""所有类别先拦截实际洞与自车禁入带交叠；不是只在dense提案生效。"""
import sys,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from build_pairs import evaluate

class EgoGuard(unittest.TestCase):
    def run_case(self,y):
        box=[100,y,200,y+60]
        df={'actors':[{'projection':{'box_xyxy':box}}]}
        g=SimpleNamespace(sources={'a':{'frames':[{}]},'d':{'frames':[df]}},obstacles={'a':[[]]})
        traj={'source_id':'a','donor_source_id':'d','frames':[{'box':box}]}
        m=np.zeros((576,1024),bool);m[y:y+60,100:200]=True
        return evaluate(g,traj,[m],np.array([1.]),{},ego_bottom_guard_px=64)
    def test_dilated_hole_hits_ego_band(self):
        # 车身仍在512上方，但预先扩洞后已侵入：必须在渲染前拒绝。
        self.assertEqual(self.run_case(447)[1],'conservative_ego_image_band')
    def test_clear_road_control(self):
        q,why=self.run_case(300);self.assertIsNone(why);self.assertEqual(q['type'],'background')

if __name__=='__main__':unittest.main()

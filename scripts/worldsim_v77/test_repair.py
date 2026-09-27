"""关键语义测试：不能把邻车、同时间重复证据、框矩形当精确删除。"""
import unittest
import tempfile,pathlib,json
import numpy as np
from repair_common import hull_mask,largest
from repair_guard import classify
from repair_admission import approved_paths
class MaskTests(unittest.TestCase):
 def test_silhouette_not_bbox(self):
  a=.7;p=np.eye(4);p[:3,:3]=[[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]];p[2,3]=8
  m=hull_mask({'pose':p,'size_lwh':[4,2,2]},np.eye(4),np.array([[100,0,64],[0,100,64],[0,0,1]]),(128,128))
  y,x=np.where(m);self.assertLess(m.sum(),(y.max()-y.min()+1)*(x.max()-x.min()+1))
 def test_component_cleanup(self):
  m=np.zeros((30,30),bool);m[2:12,2:12]=True;m[20:22,20:22]=True;r=largest(m);self.assertEqual(r.sum(),100);self.assertTrue(r[5,5])
 def test_grounded_projection(self):
  p=np.eye(4);p[2,3]=10;b={'pose':p,'size_lwh':[4,2,2]};k=np.array([[100,0,64],[0,100,64],[0,0,1]])
  m=hull_mask(b,np.eye(4),k,(128,128));self.assertTrue(m[64,64]);self.assertFalse(m[0,0])
 def test_new_car_is_blocked(self):
  core=np.zeros((40,40),bool);core[10:30,10:30]=True
  self.assertTrue(classify(core,core,core,[])['suspect_new_vehicle'])
 def test_existing_neighbor_not_new_car(self):
  core=np.zeros((40,40),bool);core[10:30,10:30]=True;n=np.zeros_like(core);n[5:20,5:15]=True
  self.assertFalse(classify(n,core,core,[n])['suspect_new_vehicle'])
 def test_vehicle_outside_edit_not_blocked(self):
  core=np.zeros((40,40),bool);core[10:20,10:20]=True;n=np.zeros_like(core);n[25:35,25:35]=True
  self.assertFalse(classify(n,core,core,[])['suspect_new_vehicle'])
 def test_rejected_background_cannot_reach_omega(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=pathlib.Path(tmp);(p/'guard_summary.json').write_text(json.dumps({'scenes':[{'scene':'x','arms':{'evidence_first':{'blocked_frames':[25],'source_miss_frames':[]}}}]}))
   with self.assertRaisesRegex(ValueError,'blocked_vehicle'):approved_paths(p,'x')
 def test_guard_clear_requires_visual_check(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=pathlib.Path(tmp);(p/'guard_summary.json').write_text(json.dumps({'scenes':[{'scene':'x','arms':{'evidence_first':{'blocked_frames':[],'source_miss_frames':[]}}}]}));(p/'observations.json').write_text('{}')
   with self.assertRaisesRegex(ValueError,'visual_not_admitted'):approved_paths(p,'x')
if __name__=='__main__':unittest.main()

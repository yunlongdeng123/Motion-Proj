import os,sys,tempfile,unittest
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).parent/'src'))
import numpy as np
from export_inspection import prepare_depth
from export_v1_ground import ground_mask

class V1GroundExport(unittest.TestCase):
 def test_tagged_projection_preserves_native_order_and_culling(self):
  class A:
   config={'neg_fov':True}
   def camera(self,raw):return dict(timestamp=-.22,W=np.eye(4,dtype=np.float32),K=np.array([[500,0,240],[0,500,160],[0,0,1]],np.float32))
  b=dict(xyz=np.array([[0,0,2],[.2,.3,-4],[.3,.1,3],[0,0,4]],np.float32),scale=np.array([[.03,.02,.01]]*4,np.float32),q=np.array([[1,0,0,0]]*4,np.float32),opacity=np.array([.5,.1,.8,.5],np.float32),t=np.array([0]*4,np.float32),sigma_t=np.array([1,1,1,.001],np.float32),vel=np.zeros((4,3),np.float32),road=np.full(4,-1,np.int64),cycle=.2)
  with tempfile.TemporaryDirectory() as folder:
   plain=prepare_depth(A(),b,102,Path(folder));tagged=prepare_depth(A(),b,102,Path(folder),True)
   np.testing.assert_array_equal(tagged['source_row'],[0,2])
   for k in plain:np.testing.assert_array_equal(plain[k],tagged[k])
 def test_geometric_candidate_is_not_semantic_or_covariance_overlap(self):
  world=np.array([[4,0,.1],[4,0,.2],[4,0,.4],[2,0,0],[4,9,0]])
  np.testing.assert_array_equal(ground_mask(world,np.eye(4),np.array([0,0,1,0]),.15),[1,0,0,0,0])
  np.testing.assert_array_equal(ground_mask(world,np.eye(4),np.array([0,0,1,0]),.3),[1,1,0,0,0])

if __name__=='__main__':unittest.main()

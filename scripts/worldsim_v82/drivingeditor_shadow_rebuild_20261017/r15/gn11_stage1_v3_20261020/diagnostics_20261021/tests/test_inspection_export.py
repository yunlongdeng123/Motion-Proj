"""针对最近回波覆盖与新增CPU深度导出的回归，不修改研究输入。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,tempfile,unittest,sys
from pathlib import Path
import numpy as np
candidate=Path(__file__).resolve().parents[1]/'src'
if candidate.exists():sys.path.insert(0,str(candidate))
from export_inspection import measured_frame,native_raster,prepare_depth
from ray_reference import project,covariance,composite,candidates

class InspectionExport(unittest.TestCase):
 def test_nearest_return_and_unified_pixel_center(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);(root/'pose').mkdir();(root/'velodyne').mkdir();np.savetxt(root/'pose/0008102.txt',np.eye(4))
   # Same projected pixel, intentionally near then far: old assignment retained the far return.
   np.array([[20.4,40.4,2,0,0,0],[81.6,161.6,8,0,0,0]],np.float32).tofile(root/'velodyne/0008102.bin')
   class A:
    config={'source_path':folder};scale=.1
    def camera(self,raw):return dict(meta={'world_to_camera':np.eye(4)},K=np.eye(3),W=np.eye(4))
   out=measured_frame(A(),102);self.assertAlmostEqual(out['lidar_nearest_m'][20,10],2);self.assertEqual(out['current_total'][0],1)
   np.testing.assert_allclose(out['current_xyz'][0],[2.1,4.1,.2],atol=1e-6)
 def test_dense_depth_sort_and_no_truncated_early_stop(self):
  with tempfile.TemporaryDirectory() as folder:
   p=dict(uv=np.array([[32,32]]*3,np.float32),conic=np.array([[1,0,1]]*3,np.float32),opacity=np.array([.5]*3,np.float32),depth=np.array([3,1,2],np.float32),lo=np.array([[2,2]]*3,np.int32),hi=np.array([[3,3]]*3,np.int32),road=np.zeros(3,bool))
   a,d=native_raster(Path(folder),p);self.assertAlmostEqual(a[32,32],.875);self.assertAlmostEqual(d[32,32],1.375)
   p['opacity'][:]=.99;a,d=native_raster(Path(folder),p);self.assertAlmostEqual(a[32,32],.99,places=6);self.assertAlmostEqual(d[32,32],.99,places=6)
 def test_cpp_projection_against_independent_numpy(self):
  with tempfile.TemporaryDirectory() as folder:
   class A:
    config={'neg_fov':True}
    def camera(self,raw):return dict(timestamp=-.22,W=np.eye(4,dtype=np.float32),K=np.array([[500,0,240],[0,500,160],[0,0,1]],np.float32))
   b=dict(xyz=np.array([[0,0,2],[.2,.3,4],[.3,.1,-1]],np.float32),scale=np.array([[.03,.02,.01]]*3,np.float32),q=np.array([[1,0,0,0]]*3,np.float32),opacity=np.array([.5,.1,.8],np.float32),t=np.array([0]*3,np.float32),sigma_t=np.ones(3,np.float32),vel=np.zeros((3,3),np.float32),road=np.array([-1,4,-1],np.int64),cycle=.2)
   a=A();p=prepare_depth(a,b,102,Path(folder));m=np.exp(-.5*((-.22-b['t'])/b['sigma_t'])**2);m[1]=1
   q=project(b['xyz'],covariance(b['scale'],b['q']),b['opacity']*m,m>.05,np.eye(4,dtype=np.float32),a.camera(102)['K'],True)
   for key in ['uv','conic','opacity','depth','lo','hi']:np.testing.assert_allclose(p[key],q[key],rtol=1e-5,atol=1e-5)

if __name__=='__main__':unittest.main()

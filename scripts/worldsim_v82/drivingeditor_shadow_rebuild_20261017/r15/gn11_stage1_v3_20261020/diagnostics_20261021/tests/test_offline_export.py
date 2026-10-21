"""离线车辆索引必须保留车后背景，区分三维框，并遵守本帧时间范围。"""
import sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from export_offline import memberships
from cloud_sampling import ACTORS

class OfflineRegionTests(unittest.TestCase):
    def test_projection_all_depths_and_box_and_time(self):
        with tempfile.TemporaryDirectory() as folder:
            label=Path(folder)/'label.png';image=np.zeros((320,480),np.uint8);image[160,240]=1;Image.fromarray(image).save(label)
            camera=dict(W=np.eye(4),K=np.array([[100.,0,240],[0,100,160],[0,0,1]]),timestamp=0.)
            boxes=[dict(actor=actor,center=[0.,0,1.] if i==0 else [100.*i,0,1],half=[.2]*3,axes=np.eye(3)) for i,actor in enumerate(ACTORS)]
            adapter=SimpleNamespace(camera=lambda raw:camera,scale=1.,meta={'frames':{'0008102':{'label_path':str(label)}}})
            sampler=SimpleNamespace(adapter=adapter,metadata={'vehicles':{'102':boxes}})
            branch=dict(xyz=np.array([[0.,0,1],[0,0,2],[10,0,1],[0,0,-1],[0,0,1]]),
                vel=np.zeros((5,3)),t=np.array([0.,0,0,0,10]),sigma_t=np.ones(5)*.1,cycle=1.)
            order=np.array([2,0,3,1,4])
            projection,box=memberships(sampler,branch,102,order)
            np.testing.assert_array_equal(projection,[0,1,0,1,0])
            np.testing.assert_array_equal(box,[0,1,0,0,0])

if __name__=='__main__':unittest.main()

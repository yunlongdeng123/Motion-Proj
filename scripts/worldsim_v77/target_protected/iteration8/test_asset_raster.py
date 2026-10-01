"""优化后的整数三角形并集必须与逐面栅格逐像素相同。"""
import cv2,numpy as np
def fast(polys):
    m=np.zeros((64,64),'uint8');p=polys.reshape(-1,2);m[p[:,1],p[:,0]]=1
    delta=polys[:,1:]-polys[:,:1];area=delta[:,0,0]*delta[:,1,1]-delta[:,0,1]*delta[:,1,0]
    keep=(area!=0)|(np.ptp(polys,axis=1).max(1)>1)
    for p in polys[keep]:cv2.fillConvexPoly(m,p,1)
    return m
def test_raster_union_subpixel_and_solid_triangles():
    rng=np.random.default_rng(7)
    for extent in [1,2,8,25]:
        anchors=rng.integers(2,35,(300,1,2));polys=(anchors+rng.integers(0,extent+1,(300,3,2))).astype('int32')
        expected=np.zeros((64,64),'uint8')
        for p in polys:cv2.fillConvexPoly(expected,p,1)
        assert np.array_equal(expected,fast(polys))
def test_overlap_does_not_cancel_interior():
    polys=np.int32([[[5,5],[50,5],[50,50]],[[5,5],[50,50],[5,50]]]*20)
    m=fast(polys);assert m[5:51,5:51].all()

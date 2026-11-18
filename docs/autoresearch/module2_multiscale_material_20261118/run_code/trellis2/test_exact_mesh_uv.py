import numpy as np
from exact_mesh_uv import TriangleLocatorXY,WorldMeshUVMap,sample_uv_rgba

def test_curved_nonaffine_world_points_exact_and_outside_rejected():
    v=np.array([[100.,200.,0],[102.,200.1,.2],[102.4,203.,.4],[99.9,202.8,-.1]])
    f=np.array([[0,1,2],[0,2,3]])
    weights=np.array([[.2,.3,.5],[.3,.5,.2]])
    q=np.einsum('ni,nij->nj',weights,v[f])
    r=TriangleLocatorXY(v,f).locate(np.r_[q[:,:2],[[90,190]]])
    assert r['valid'].tolist()==[True,True,False]
    np.testing.assert_allclose(r['world_xyz'][:2],q,atol=1e-12)
    np.testing.assert_allclose(r['barycentric'][:2],weights,atol=1e-12)

def test_texel_center_and_v_orientation():
    t=np.array([[[1,0,0,1],[0,1,0,1]],[[0,0,1,1],[1,1,1,1]]],float)
    np.testing.assert_allclose(sample_uv_rgba(t,np.array([[.25,.75],[.25,.25]])),[t[0,0],t[1,0]])

def test_per_face_seam_coordinates_are_not_averaged():
    v=np.array([[0.,0,0],[1,0,0],[1,1,0],[0,1,0]])
    f=np.array([[0,1,2],[0,2,3]])
    uv=np.array([[[.1,.2],[.2,.2],[.2,.3]],[[.8,.7],[.9,.8],[.8,.8]]])
    t=np.ones((2,2,4))
    r=WorldMeshUVMap(v,f,uv,t).query(np.array([[.75,.25],[.25,.75]]))
    assert (r['generated_uv'][0]<.4).all()
    assert (r['generated_uv'][1]>.6).all()
    np.testing.assert_array_equal(r['fixed_face_id'],[0,1])

def test_barycentric_reconstructs_all_boundary_vertices():
    v=np.array([[0.,0,0],[1,0,.1],[1,1,.2],[0,1,.3]])
    f=np.array([[0,1,2],[0,2,3]])
    r=TriangleLocatorXY(v,f).locate(v[:,:2])
    assert r['valid'].all()
    np.testing.assert_allclose(r['world_xyz'],v,atol=1e-14)

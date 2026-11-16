import numpy as np
from .road_asset import build_observation_priority_atlas
from .road_surface_projection import project_surface_mesh


def test_perspective_correct_triangle_hits_and_near_permission():
    keys=np.array([[0,0],[1,0],[0,1],[1,1]])
    world=np.c_[keys.astype(float),np.full(4,2.)]
    atlas,_=build_observation_priority_atlas(world,keys,np.full((4,3),.3,np.float32),np.ones(4,bool),np.ones(4,bool))
    intrinsic=np.array([[10.,0.,.5],[0.,10.,.5],[0.,0.,1.]])
    result=project_surface_mesh(atlas,np.ones(4,bool),intrinsic,np.eye(4),[6,6])
    assert result['valid'].all()
    assert np.allclose(result['world_xyz'][2,3],[.6,.4,2.],atol=1e-12)
    assert np.nanmax(result['reprojection_error_px'])<1e-12
    assert np.allclose(result['triangle_weights'].sum(axis=2),1)
    bounded=project_surface_mesh(atlas,np.ones(4,bool),intrinsic,np.eye(4),[6,6],np.full(4,3.))
    assert not bounded['valid'].any()

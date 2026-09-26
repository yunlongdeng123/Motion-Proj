"""针对坐标约定、非目标保持及空对象边界的独立小控制。"""
import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('v77_geometry',Path(__file__).parents[1]/'scripts/worldsim_v77/geometry.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)


def test_world_axis_translation_and_yaw_with_rotated_source():
    source=np.array([[0,-1,0,10],[1,0,0,20],[0,0,1,3],[0,0,0,1.]])
    points=np.array([[10,21,3],[40,50,6.]])
    target=g.target_pose(source,[3,0,0],90)
    result,_,_=g.edit_actor(points,np.zeros((2,3),np.uint8),[True,False],source,target,'MOVE')
    np.testing.assert_allclose(result[0],[12,20,3],atol=1e-12)
    np.testing.assert_array_equal(result[1],points[1]);np.testing.assert_array_equal(points[0],[10,21,3])


def test_delete_clone_and_empty_preserve_other_objects():
    points=np.array([[1.,2,3],[8,9,10]]);colors=np.array([[1,2,3],[4,5,6]],np.uint8)
    source=np.eye(4);target=g.target_pose(source,[3,0,0])
    deleted,_,ids=g.edit_actor(points,colors,[True,False],source,target,'DELETE')
    np.testing.assert_array_equal(deleted,points[1:]);np.testing.assert_array_equal(ids,[1])
    inserted,c,ids=g.edit_actor(points,colors,[True,False],source,target,'INSERT')
    np.testing.assert_array_equal(inserted,[[1,2,3],[8,9,10],[4,2,3]])
    np.testing.assert_array_equal(c[-1],colors[0]);np.testing.assert_array_equal(ids,[0,1,0])
    empty,_,_=g.edit_actor(points,colors,[False,False],source,target,'INSERT')
    np.testing.assert_array_equal(empty,points)


def test_sim3_recovers_known_metric_frame():
    x=np.array([[0.,0,0],[1,0,0],[0,2,0],[0,0,1],[-1,0,1],[0,-2,-1]])
    expected_r=np.array([[0,-1.,0],[1,0,0],[0,0,1]])
    y=x@expected_r.T*4+[11,22,33]
    s,r,t,errors=g.fit_sim3(x,y)
    np.testing.assert_allclose(s,4,atol=1e-12);np.testing.assert_allclose(r,expected_r,atol=1e-12)
    np.testing.assert_allclose(t,[11,22,33],atol=1e-12);assert errors.max()<1e-11

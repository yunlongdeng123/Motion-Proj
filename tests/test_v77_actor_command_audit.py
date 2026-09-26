"""覆盖只查终点会漏检的穿越碰撞，并验证旋转框与高度过滤。"""
import importlib.util,pathlib
import numpy as np
path=pathlib.Path(__file__).with_name('v77_actor_command_audit.py')
if not path.exists():path=pathlib.Path(__file__).resolve().parents[1]/'scripts/worldsim_v77/actor_command_audit.py'
spec=importlib.util.spec_from_file_location('v77_audit',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def instance(x,y,z=0,angle=0,size=(2,1,1)):
    c,s=np.cos(angle),np.sin(angle);pose=np.array([[c,-s,0,x],[s,c,0,y],[0,0,1,z],[0,0,0,1]])
    return {'class_name':'car','frame_annotations':{'frame_idx':[0],'obj_to_world':[pose.tolist()],'box_size':[list(size)]}}
def test_crossing_collision_without_endpoint_overlap():
    r=m.audit({'a':instance(0,0),'b':instance(3,0)},'a',1,np.array([6.,0,0]))
    assert r['endpoint_collision_frames']==0
    assert r['swept_collision_frames']==1 and r['swept_hit_ids']==['b']
    assert abs(r['max_new_swept_overlap_m2']-2)<1e-9
def test_parallel_clearance_is_metric():
    r=m.audit({'a':instance(0,0),'b':instance(3,2)},'a',1,np.array([6.,0,0]))
    assert r['swept_collision_frames']==0 and abs(r['minimum_swept_clearance_m']-1)<1e-9
def test_rotation_changes_footprint():
    p,s=m.box(instance(0,0,angle=np.pi/2,size=(4,2,1)),0);q=m.footprint(p,s)
    assert np.allclose(q.bounds,[-1,-2,1,2]) and abs(q.area-8)<1e-9
def test_nonoverlapping_height_is_excluded():
    r=m.audit({'a':instance(0,0),'b':instance(3,0,z=3)},'a',1,np.array([6.,0,0]))
    assert r['swept_collision_frames']==0 and r['minimum_swept_clearance_m'] is None

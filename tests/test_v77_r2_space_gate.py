import pathlib,sys
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts/worldsim_v77'))
from r2_space_gate import supported_voxels
def test_dense_single_scan_is_not_repeated_static_evidence():
    p=np.repeat([[1.02,.02,.02]],100,axis=0)
    v,s=supported_voxels(p,np.zeros(100,int),np.zeros(3));assert len(v)==1 and s.tolist()==[1]
    v,s=supported_voxels(p,np.arange(100)%3,np.zeros(3));assert s.tolist()==[3]
def test_empty_points_do_not_create_free_space_or_obstacles():
    v,s=supported_voxels([],[],np.zeros(3));assert v.shape==(0,3) and s.size==0

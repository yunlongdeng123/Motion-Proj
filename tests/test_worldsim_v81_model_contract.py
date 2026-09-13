import importlib.util
from pathlib import Path
import numpy as np

def test_dvgt_official_rdf_scale_and_camera_timestamp():
    spec=importlib.util.spec_from_file_location('v81_infer',Path(__file__).resolve().parents[1]/'scripts/run_worldsim_v81_inference.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    # RDF [right,down,forward]=[1,2,3] checkpoint units -> FLU [30,-10,-20] metres.
    first=np.eye(4);first[:3,3]=[100,200,3]
    target=np.eye(4);target[:3,3]=[102,203,4]
    bogus_lidar=np.eye(4);bogus_lidar[:3,3]=[-90,-90,-90]
    v={'world_from_ego_camera':first.tolist(),'world_from_ego':bogus_lidar.tolist()}
    out=mod.dvgt_camera_points(np.array([[1.,2.,3.]]),v,{'world_from_camera':target.tolist()})
    np.testing.assert_allclose(out,[[28,-13,-21]])

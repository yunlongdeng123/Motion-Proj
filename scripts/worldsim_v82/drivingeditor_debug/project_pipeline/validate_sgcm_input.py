"""Read-only SGCM ground readiness; preserve absent loader PCA as pending."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
import yaml

p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--repo',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
sys.path.insert(0,str(Path(a.repo)/'scene'))
from paper_sgcm import SGCMConfig,fit_ground_plane,transform_points,validate_similarity,create_sgcm_controller
source=Path(a.source);args=SimpleNamespace(**yaml.safe_load(Path(a.config).read_text()))
def read_xyz(path):
    vertices=PlyData.read(str(path))['vertex']
    return np.column_stack([vertices[key] for key in ('x','y','z')]).astype(np.float64)
actors=[read_xyz(path) for path in sorted(source.glob('points3d_dynamic_*.ply'))]
assert actors,'No actor priors'
static=read_xyz(source/'points3d.ply');dynamic=np.concatenate(actors)
distance,_=cKDTree(dynamic).query(static,k=1,distance_upper_bound=.1,workers=4)
static=static[distance>=.1]
pose_path=sorted((source/'pose').glob('*.txt'))[0];pose=np.loadtxt(pose_path)
assert np.isclose(validate_similarity(pose),1,rtol=1e-4)
affine=np.linalg.inv(pose)
static_anchor=transform_points(static,affine)
actors_anchor=[transform_points(actor,affine) for actor in actors]
bounds=[np.stack([actor.min(axis=0),actor.max(axis=0)]) for actor in actors_anchor]
plane,fit=fit_ground_plane(static_anchor,bounds,SGCMConfig.from_args(args))
result={'status':'ground_support_passed','gpu_used':False,'source':str(source),'config':a.config,
        'actor_points':[len(actor) for actor in actors],'static_after_dynamic_exclusion':len(static),
        'first_pose':str(pose_path),'ground_plane_first_vehicle_frame':plane.tolist(),'ground_fit':fit,
        'coordinate_normalization':'pending_current_loader_transform','dr_m':.5,
        'note':'Ground fitting only is validated in world meters. Runtime initialization must validate the actual Scene PCA transform and derive model dr; no assumed scale.'}
if (source/'transform_matrix.txt').is_file() or (source/'sgcm_transform_matrix.txt').is_file():
    controller=create_sgcm_controller(source,args)
    result.update(status='controller_initialization_passed',coordinate_normalization='validated',controller_metadata=controller.metadata)
Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

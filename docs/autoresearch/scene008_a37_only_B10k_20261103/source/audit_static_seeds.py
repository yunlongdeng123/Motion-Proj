"""只读核验：原 B 0.1m 初始化规则后，保留车辆是否仍有真实静态种子。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parent.parent
OLD=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/baseline_engineering_20261018/full5_pipeline/inputs/scene0008_089_137')
p=o3d.io.read_point_cloud(str(OLD/'points3d.ply'));xyz=np.asarray(p.points);ts=np.asarray(p.normals)[:,0]
t=o3d.io.read_point_cloud(str(OLD/'points3d_dynamic_4.ply'));target=np.asarray(t.points)
nearest,_=cKDTree(target).query(xyz,distance_upper_bound=.1);keep=nearest>=.1
result=dict(raw_scene_seed_count=len(xyz),after_original_B_single_actor_exclusion=int(keep.sum()),dynamic_aliases=[37],retained={})
for index,actor in enumerate([5,26,18,6]):
    p=o3d.io.read_point_cloud(str(OLD/f'points3d_dynamic_{index}.ply'));a=np.asarray(p.points);at=np.asarray(p.normals)[:,0]
    before=after=0
    for stamp in np.unique(at):
        ids=np.flatnonzero(np.abs(ts-stamp)<1e-7);query=a[np.abs(at-stamp)<1e-7]
        if not len(ids):continue
        d,nn=cKDTree(xyz[ids]).query(query,distance_upper_bound=.002);hit=d<.002
        before+=int(hit.sum());after+=int(keep[ids[nn[hit]]].sum())
    result['retained'][str(actor)]=dict(source_real_points=len(a),same_time_sampled_scene_matches=before,retained_after_target_exclusion=after)
assert all(v['retained_after_target_exclusion']>0 for v in result['retained'].values())
(ROOT/'static_seed_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

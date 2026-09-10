"""从 BUILD 点估计公共局部几何，不读取 QUERY 或模型预测。"""
import argparse
import json
import os
from pathlib import Path
import resource
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
from scipy.spatial import cKDTree

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    args=parser.parse_args()
    index=json.loads((args.data/'index.json').read_text())
    started=time.monotonic(); points_count=valid_count=0
    for row in index['cases']:
        folder=Path(row['build_file']).parent
        with np.load(folder/'build.npz',allow_pickle=False) as data:
            points=data['support_points_actor_m'].astype(np.float64)
            origins=data['origins_actor_m'][data['positive_actor']].astype(np.float64)
            observed=data['points_actor_m'][data['positive_actor']].astype(np.float64)
        count=len(points); k=min(16,count)
        normals=np.zeros((count,3),np.float32)
        eigenvalues=np.zeros((count,3),np.float32)
        basis=np.zeros((count,3,3),np.float32)
        valid=np.zeros(count,bool)
        neighbors=np.empty((count,k),np.int32)
        if count>=3:
            tree=cKDTree(points)
            sensor_indices=cKDTree(observed).query(points,k=1,workers=1)[1]
            sensor_vectors=origins[sensor_indices]-points
            # 分块局部协方差，避免点数较大对象一次产生 N*k*3 大临时数组。
            for left in range(0,count,2048):
                right=min(left+2048,count)
                indices=tree.query(points[left:right],k=k,workers=1)[1]
                local=points[indices]
                centered=local-local.mean(axis=1,keepdims=True)
                covariance=np.einsum('nki,nkj->nij',centered,centered)/k
                values,vectors=np.linalg.eigh(covariance)
                normal=vectors[:,:,0]
                flip=(normal*sensor_vectors[left:right]).sum(1)<0
                vectors[flip,:,0]*=-1
                vectors[:,:,2]=np.cross(vectors[:,:,0],vectors[:,:,1])
                # 仅定向朝测量原点，不将法向估计当完整表面真值或可靠硬见证。
                neighbors[left:right]=indices
                normals[left:right]=vectors[:,:,0]
                basis[left:right]=vectors
                eigenvalues[left:right]=values
                valid[left:right]=values[:,1]>1e-10
        elif count:
            neighbors[:]=np.arange(count)[None,:]
        np.savez(folder/'build_geometry.npz',normal_actor=normals,local_basis_actor=basis,
                 covariance_eigenvalues_m2=eigenvalues,normal_valid=valid,knn_indices=neighbors)
        points_count+=count; valid_count+=int(valid.sum())
    summary={'task':'WS-V74-P0-DATA-01','status':'done','cases':len(index['cases']),
        'build_points':points_count,'valid_normal_points':valid_count,'invalid_normal_points':points_count-valid_count,
        'neighbors':'min(16, point count)','normal_orientation':'toward nearest matching BUILD return sensor origin',
        'degeneracy':'fewer than 3 points or second covariance eigenvalue <= 1e-10 m^2 -> normal_valid=false',
        'boundary':'BUILD-only optional initialization/normal baseline input; normals are estimates, never heldout GT; no method scores',
        'wall_s':time.monotonic()-started,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20}
    for path in [args.data/'geometry_summary.json',Path(__file__).resolve().parents[1]/'docs/autoresearch/worldsim_v74/p0/geometry_summary.json']:
        path.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary),flush=True)

if __name__=='__main__': main()

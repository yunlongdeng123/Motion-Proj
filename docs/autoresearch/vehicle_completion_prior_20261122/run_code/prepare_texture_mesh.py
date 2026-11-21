"""将单个车辆网格归一化到 TRELLIS 的固定[-.5,.5]域，保存可逆米制映射。"""
from pathlib import Path
import argparse, json, os
os.environ['OMP_NUM_THREADS']='2'
import numpy as np
import trimesh

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mesh',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve existing prepared mesh')
    mesh=trimesh.load(a.mesh,force='mesh',process=False)
    xyz=np.asarray(mesh.vertices,dtype=np.float64)
    faces=np.asarray(mesh.faces)
    if not np.isfinite(xyz).all() or len(faces)==0:raise ValueError('invalid mesh')
    # Native actor: x forward, y left, z up. TRELLIS mesh: y up.
    center=(xyz.min(0)+xyz.max(0))/2
    span=float(np.ptp(xyz,axis=0).max())/.98
    normalized=(xyz-center)[:,[0,2,1]]/span
    result=trimesh.Trimesh(vertices=normalized,faces=faces[:,[0,2,1]],process=False)
    a.output.mkdir(parents=True)
    result.export(a.output/'mesh_yup_normalized.obj')
    receipt=dict(schema='vehicle_mesh_texture_transform.v1',source=str(a.mesh),
        actor_axes='x forward, y left, z up; metres',texture_axes='x forward, y up, z left',
        actor_center_m=center.tolist(),metres_per_texture_unit=span,
        axis_permutation=[0,2,1],reflection_winding_reversed=True,
        input_bounds_m=mesh.bounds.tolist(),normalized_bounds=result.bounds.tolist(),
        vertices=len(xyz),faces=len(faces),geometry_optimization_updates=0,
        inverse='actor_xyz = texture_xyz[:,[0,2,1]] * metres_per_texture_unit + actor_center_m')
    (a.output/'transform.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))
if __name__=='__main__':main()

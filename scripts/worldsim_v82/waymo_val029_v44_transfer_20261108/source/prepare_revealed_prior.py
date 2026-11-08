"""Build new-object PVD contracts from this scene's observed points, same normalization and seed."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
from plyfile import PlyData
ROOT=Path(__file__).resolve().parent.parent
BANK=Path('/root/autodl-tmp/codex_tmp/waymo_val029_cpu_20261108/inputs')
PVD=ROOT.parent/'scene008_a37_C2_20261103_r1/source/prior/pvd'
def main():
    out=ROOT/'revealed_prior';out.mkdir(exist_ok=False)
    identity=json.loads((ROOT/'identity_contract.json').read_text());source=next(x for x in identity['source_actor_rows'] if x['actor_id']==25)
    v=PlyData.read(source['ply'])['vertex'].data;world=np.column_stack([v[k] for k in ['x','y','z']]);raws=np.rint(np.array(v['nx'])/.02+98.5).astype(int)
    xyz=np.empty_like(world);rgb=np.zeros_like(world);valid=np.zeros(len(world),bool);sizes=[];counts=[]
    for raw in sorted(set(raws)):
        ids=np.flatnonzero(raws==raw);obs=json.loads((BANK/'observations'/f'{raw:03d}.json').read_text());box=next(b for b in obs['boxes'] if b['track']=='DwjZr_bYUSZOr-XDGregnA');sizes.append(box['dimensions'])
        inv=np.linalg.inv(np.array(box['world_from_actor']).reshape(4,4));xyz[ids]=world[ids]@inv[:3,:3].T+inv[:3,3]
        camera=np.array(obs['camera_pose_world']).reshape(4,4)@np.array(obs['camera_extrinsic']).reshape(4,4);inv=np.linalg.inv(camera);cam=world[ids]@inv[:3,:3].T+inv[:3,3]
        fx,fy,cx,cy,k1,k2,p1,p2,k3=obs['camera_intrinsic_Brown'];x=-cam[:,1]/cam[:,0];y=-cam[:,2]/cam[:,0];rr=x*x+y*y;rad=1+k1*rr+k2*rr**2+k3*rr**3
        uv=np.column_stack((fx*(x*rad+2*p1*x*y+p2*(rr+2*x*x))+cx,fy*(y*rad+p1*(rr+2*y*y)+2*p2*x*y)+cy));pixels=np.clip(uv.astype(int),[0,0],[1919,1279])
        im=np.array(Image.open(BANK/obs['image_path']).convert('RGB'));rgb[ids]=im[pixels[:,1],pixels[:,0]]/255;valid[ids]=True;counts.append(dict(raw=int(raw),points=len(ids)))
    size=np.median(sizes,axis=0);point_path=out/'observations.npz'
    np.savez_compressed(point_path,observed_points_actor_native_m=xyz,size_lwh_m=size,raw_frame=raws,
        sample_xyz_local_m=xyz,sample_raw_frame=raws,sample_rgb=rgb,sample_rgb_valid=valid,
        sample_source_ply_row=np.arange(len(xyz)))
    sha=hashlib.sha256(point_path.read_bytes()).hexdigest();norm0=json.loads((PVD/'normalization_contract.json').read_text())
    rotation=np.array(norm0['native_to_training_raw_rotation']);span=np.array(norm0['training_mean_full_bbox_span_xyz']);mean=np.array(norm0['training_mean_xyz']);center=np.array(norm0['training_mean_full_bbox_center_xyz']);std=norm0['training_scalar_std']
    scale=np.diag(size[[1,2,0]]/span);matrix=np.eye(4);matrix[:3,:3]=rotation.T@scale*std;matrix[:3,3]=rotation.T@scale@(mean-center)
    norm=dict(schema_version='pvd.fixed_wholebox.new_object.v1',whole_box_lwh_m=size.tolist(),partial_bounds_scaled=False,partial_bounds_centered=False,pvd_to_actor_native_4x4=matrix.tolist(),training_mean_xyz=mean.tolist(),training_scalar_std=std,training_mean_full_bbox_span_xyz=span.tolist(),training_mean_full_bbox_center_xyz=center.tolist(),native_to_training_raw_rotation=rotation.tolist(),normalization_source='same fixed whole-box and official PointFlow car/train stats as current method',front_sign_assumption=norm0['front_sign_assumption'])
    (out/'normalization_contract.json').write_text(json.dumps(norm,indent=2))
    (out/'input_contract.json').write_text(json.dumps(dict(points_npz=str(point_path),points_sha256=sha,conditioning_count=200,scope='white SUV actor25, BUILD-only real measurements',frames=counts,heldout_excluded=list(range(79,119))),indent=2))
    print('NEW_REVEALED_PRIOR_INPUT',len(xyz),size.tolist(),len(counts))
if __name__=='__main__':main()

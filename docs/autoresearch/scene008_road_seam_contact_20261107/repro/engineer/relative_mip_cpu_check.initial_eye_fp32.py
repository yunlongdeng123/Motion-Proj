"""CPU-only geometry/formula cross-check; no rasterizer/model/CUDA initialization."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from relative_mip import RelativeMip,projected_raw_covariance,filter_factor,ROOT,V44
from world_surface_sampling import read_surface,resample_surface


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stats(values):
    v=np.asarray(values,float)
    return dict(min=float(v.min()),median=float(np.median(v)),max=float(v.max()),mean=float(v.mean()))


def main():
    torch.set_num_threads(1)
    output=ROOT/'engineer/relative_mip_cpu_check.json'
    if output.exists():raise FileExistsError('Preserve prior CPU check')
    reference=ROOT/'red/geometry_cpu.py'
    tree=ast.parse(reference.read_text());functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('norm','projected_rows')]
    namespace={'np':np,'math':math};exec(compile(ast.Module(body=functions,type_ignores=[]),str(reference),'exec'),namespace)
    policy=RelativeMip();parent=read_surface(policy.parent_asset_path);child=resample_surface(parent,2)
    parent_rows=policy.parent_geometry(torch.device('cpu'))
    opacity=torch.sigmoid(torch.full((len(parent['active']),1),float(np.log(.99/.01)),dtype=torch.float32))
    parent_rows=dict(parent_rows,opacities=opacity)
    ids=torch.as_tensor(child['parent_indices'],dtype=torch.long)
    child_rows=dict(means3D=torch.as_tensor(child['model_xyz'],dtype=torch.float32),
        scales=torch.as_tensor(child['gaussian_sigma_m']*float(parent['meter_to_model']),dtype=torch.float32),
        rotations=parent_rows['rotations'][ids],opacities=opacity[ids],cov3D_precomp=None)
    normal=parent['normal_world']@parent['world_to_model'][:3,:3].T/float(parent['meter_to_model'])
    normal/=np.linalg.norm(normal,axis=1)[:,None]
    poses=[];source_hashes={str(reference):sha(reference),str(policy.parent_asset_path):sha(policy.parent_asset_path)}
    for raw in (105,113,128,130,137):
        metrics=V44/f'results/full49_r1/v44/{raw}/metrics.json'
        camera_json=json.loads(metrics.read_text())['actual_camera_pose'];source_hashes[str(metrics)]=sha(metrics)
        camera=SimpleNamespace(world_view_transform=torch.tensor(camera_json['world_view'],dtype=torch.float32),
            FoVx=camera_json['FoVx'],FoVy=camera_json['FoVy'],image_width=camera_json['width'],image_height=camera_json['height'])
        ours=projected_raw_covariance(camera,parent_rows,policy.neg_fov,policy.normalize_quaternion).numpy()
        red=namespace['projected_rows'](parent['model_xyz'],normal,
            parent['gaussian_sigma_m'][:,0].astype(float),parent['gaussian_sigma_m'][:,2].astype(float),
            parent['opacity'],camera_json,float(parent['meter_to_model']),policy.neg_fov)['cov2D']
        red[:,0,0]-=.3;red[:,1,1]-=.3
        # Native quaternions/model scales are fp32, independently normalized normal reference is fp64.
        maxabs=float(np.max(np.abs(ours-red)));assert maxabs<2e-5,maxabs
        identity=policy(camera,parent_rows,torch.arange(len(opacity)))
        assert identity is parent_rows['opacities'],'n1 must preserve the original opacity object'
        before={key:t.clone() for key,t in child_rows.items() if torch.is_tensor(t)}
        alpha=policy(camera,child_rows,ids)
        assert all(torch.equal(child_rows[key],value) for key,value in before.items())
        # Same contents in a distinct camera object must hit the content cache.
        cache_count=len(policy._parent_factors)
        clone=SimpleNamespace(**vars(camera));clone.world_view_transform=camera.world_view_transform.clone()
        assert torch.equal(alpha,policy(clone,child_rows,ids));assert len(policy._parent_factors)==cache_count
        altered=SimpleNamespace(**vars(camera));altered.world_view_transform=camera.world_view_transform.clone()
        altered.world_view_transform[3,2]+=.001
        changed=policy(altered,child_rows,ids);assert not torch.equal(alpha,changed)
        poses.append(dict(raw=raw,reference_covariance_maxabs_px2=maxabs,
            raw_minor_eigenvalue_px2=stats(np.linalg.eigvalsh(ours)[:,0]),
            ratio=stats((alpha/child_rows['opacities']).numpy()),alpha=stats(alpha.numpy()),
            n1_object_identity=True,input_rows_unmodified=True,camera_content_cache_check=True))
    degenerate_rejected=False
    try:filter_factor(torch.zeros((1,2,2),dtype=torch.float64))
    except ValueError:degenerate_rejected=True
    assert degenerate_rejected
    # At identical J, four quarter-covariance kernels preserve uncomposited 2D opacity integral.
    cov=torch.tensor([[[1.2,.3],[.3,.5]]],dtype=torch.float64);sub=cov/4
    ratio=filter_factor(sub)/filter_factor(cov)
    original=torch.linalg.det(cov+.3*torch.eye(2)).sqrt()
    refined=4*ratio*torch.linalg.det(sub+.3*torch.eye(2)).sqrt()
    integral_error=float((refined-original).abs().max());assert integral_error<1e-12
    record=dict(status='PASS_CPU_ONLY_NOT_GPU_VALIDATED',gpu_forwards=0,training_updates=0,
        source_sha256=source_hashes,policy=policy.receipt(),poses=poses,
        degenerate_covariance_rejected=degenerate_rejected,affine_opacity_integral_maxabs=integral_error,
        notes=['reference normal covariance differs from native fp32 quaternion/scale rounding',
            'No claim of composited alpha, cutoff, early-stop, visibility or pixel parity preservation.'])
    record['engineering_sources_sha256']={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('relative_mip.py'),Path(__file__).with_name('refined_road_runtime.py'))}
    output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),sha256=sha(output),poses=poses,affine_opacity_integral_maxabs=integral_error)))


if __name__=='__main__':main()

"""CPU-only projection calibration and generic wrapper contract verification."""
import hashlib,json,math,types,os
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
from relative_area_asset import RelativeAreaOpacityAsset,covariance_world,projected_covariance,relative_area_factors

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
ASSET=Path(os.environ.get('SCALE_ASSET',str(ROOT/'work/a6_appearance_formal/collected_publication/output_adapter/projective_surface/formal_observed/formal/complete_a6.npz')))
DATA=Path(os.environ.get('SCALE_DATA',str(HERE)))

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(x):
    a=np.asarray(x).ravel();q=np.quantile(a,[0,.1,.5,.9,1])
    return dict(min=float(q[0]),p10=float(q[1]),median=float(q[2]),p90=float(q[3]),max=float(q[4]),mean=float(a.mean()))
def camera(v,obs,res):
    a=v['cameras'][res]
    return types.SimpleNamespace(image_width=a['wh'][0],image_height=a['wh'][1],fx=a['fx'],fy=a['fy'],
        cx=a['cx'],cy=a['cy'],FoVx=a['FoVx'],FoVy=a['FoVy'],
        world_view_transform=torch.tensor(np.asarray(obs['model_to_camera']).T,dtype=torch.float32))

def generic_contract():
    rng=np.random.default_rng(73);checks=[]
    for n in (9,23):
        q=rng.normal(size=(n,4));q/=np.linalg.norm(q,axis=1,keepdims=True)
        scales=rng.uniform(.01,.08,(n,3));means=rng.normal(size=(n,3));means[:,2]=rng.uniform(2,6,n)
        T=lambda a:torch.tensor(a,dtype=torch.float32)
        rows=dict(means3D=T(means),scales=T(scales),rotations=T(q),cov3D_precomp=None,
            opacities=T(rng.uniform(.4,.95,(n,1))),features=T(rng.normal(size=(n,24))))
        # Independent scalar quaternion rotation formula in NumPy (row-major).
        S=[]
        for qi,si in zip(rows['rotations'].numpy().astype(float),rows['scales'].numpy().astype(float)):
            r,x,y,z=qi
            R=np.array([[1-2*(y*y+z*z),2*(x*y-r*z),2*(x*z+r*y)],
                [2*(x*y+r*z),1-2*(x*x+z*z),2*(y*z-r*x)],
                [2*(x*z-r*y),2*(y*z+r*x),1-2*(x*x+y*y)]])
            S.append(R@np.diag(si**2)@R.T)
        S=np.stack(S);allrows=torch.ones(n,dtype=torch.bool)
        err=float(np.max(np.abs(covariance_world(rows,allrows).numpy()-S)));assert err<1e-15
        p=torch.tensor(S[:,[0,0,0,1,1,2],[0,1,2,1,2,2]],dtype=torch.float64)
        packed=dict(rows,scales=None,rotations=None,cov3D_precomp=p)
        chart=torch.tensor(np.arange(n)%3==1)
        c=types.SimpleNamespace(image_width=480,image_height=320,FoVx=.87,FoVy=.60,world_view_transform=torch.eye(4))
        pipe=types.SimpleNamespace(neg_fov=True)
        a=relative_area_factors(rows,c,pipe,chart);b=relative_area_factors(packed,c,pipe,chart)
        assert torch.equal(a,b)
        class Base:pass
        base=Base();base.xyz=rows['means3D'];base.representation_kind=chart.to(torch.int16)
        base.rows=lambda *args,**kwargs:rows
        wrapped=RelativeAreaOpacityAsset(base)
        op=rows['opacities'].clone()
        low=wrapped.rows(None,c,pipe,16)
        assert torch.equal(rows['opacities'],op) and torch.equal(low['opacities'][~chart],op[~chart])
        assert all(low[k] is rows[k] for k in rows if k!='opacities')
        c.image_width=1920;c.image_height=1280
        assert wrapped.rows(None,c,pipe,16) is rows
        assert torch.equal(wrapped.last_area_factors,torch.ones(n))
        checks.append(dict(n=n,nonprefix_chart_mask=True,general_anisotropic_covariance_numpy_maxabs=err,
            packed_vs_scales_factors_exact=True,HR_rows_object_identity=True,
            nonchart_opacity_exact=True,source_opacity_exact=True,all_other_fields_object_identity=True))
    return checks

@torch.no_grad()
def main():
    receipt_path=DATA/'receipt.remote.json'
    if not receipt_path.is_file():receipt_path=DATA/'receipt.json'
    receipt=json.loads(receipt_path.read_text())
    obs=json.loads((DATA/'camera_geometry_cpu.json').read_text())
    assert sha(ASSET)==receipt['asset_sha256']
    with np.load(ASSET,allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in ('xyz','sigma','thickness','normal','opacity','representation_kind')}
    chart=torch.tensor(arrays['representation_kind']==1)
    pipe=types.SimpleNamespace(neg_fov=True)
    output=[]
    for v,o in zip(receipt['views'],obs['records']):
        assert v['raw']==o['raw']
        pose=torch.tensor(o['model_from_local'][0],dtype=torch.float32)
        means=torch.tensor(arrays['xyz'])@pose[:3,:3].T+pose[:3,3]
        scale=pose[:3,:3].norm(dim=0).mean()
        normal=torch.nn.functional.normalize(torch.tensor(arrays['normal'])@pose[:3,:3].T,dim=1)
        q=torch.stack((1+normal[:,2],-normal[:,1],normal[:,0],torch.zeros(len(normal))),1)
        q=torch.where((q.norm(dim=1)<1e-5)[:,None],q.new_tensor([0,1,0,0]),q)
        q=torch.nn.functional.normalize(q,dim=1)
        rows=dict(means3D=means,scales=torch.tensor(np.stack((arrays['sigma'],arrays['sigma'],arrays['thickness']),1))*scale,
            rotations=q,cov3D_precomp=None,opacities=torch.tensor(arrays['opacity'][:,None]))
        c=camera(v,o,'HR');lr=camera(v,o,'LR')
        actual=projected_covariance(rows,c,pipe,chart)
        low=projected_covariance(rows,lr,pipe,chart)
        t=means[chart].double()@c.world_view_transform[:3,:3].double()+c.world_view_transform[3,:3].double()
        z=t[:,2];idealJ=torch.zeros(len(t),2,3,dtype=torch.float64)
        idealJ[:,0,0]=c.fx/z;idealJ[:,0,2]=-c.fx*t[:,0]/z.square()
        idealJ[:,1,1]=c.fy/z;idealJ[:,1,2]=-c.fy*t[:,1]/z.square()
        T=c.world_view_transform[:3,:3].double()[None]@idealJ.transpose(-1,-2)
        ideal=T.transpose(-1,-2)@covariance_world(rows,chart)@T
        e=torch.linalg.eigvalsh(actual).numpy();ei=torch.linalg.eigvalsh(ideal).numpy()
        k=relative_area_factors(rows,lr,pipe,chart)[chart].numpy()
        prior=v['compensation']['chart_factor']
        assert abs(float(k.mean())-prior['mean'])<1e-7
        assert abs(float(e.min()/16)-v['compensation']['geometric_eigenvalues_LR']['min'])<1e-8
        assert abs(float(e.max()/16)-v['compensation']['geometric_eigenvalues_LR']['max'])<1e-8
        record=dict(raw=v['raw'],chart_rows=int(chart.sum()),source_sigma_HR_pixels=0.6,
            physical_pinhole_HR_unfiltered_sigma_minor=stats(np.sqrt(ei[:,0])),
            physical_pinhole_HR_unfiltered_sigma_major=stats(np.sqrt(ei[:,1])),
            active_legacy_HR_unfiltered_sigma_minor=stats(np.sqrt(e[:,0])),
            active_legacy_HR_unfiltered_sigma_major=stats(np.sqrt(e[:,1])),
            active_legacy_HR_filtered_sigma_minor=stats(np.sqrt(e[:,0]+.3)),
            active_legacy_HR_filtered_sigma_major=stats(np.sqrt(e[:,1]+.3)),
            active_legacy_LR_filtered_sigma_minor=stats(np.sqrt(e[:,0]/16+.3)),
            active_legacy_LR_filtered_sigma_major=stats(np.sqrt(e[:,1]/16+.3)),
            native_over_pinhole_unfiltered_ellipse_area=stats(np.sqrt(np.linalg.det(actual.numpy())/np.linalg.det(ideal.numpy()))),
            relative_LR_factor=stats(k),
            unfiltered_HR_vs_16LR_maxabs=float((actual-16*low).abs().max()),
            source_pose_and_view_float32=True,calibration_uses_known_camera=True,
            native_focal_HR=[c.image_width/(2*math.tan(-.5)),c.image_height/(2*math.tan(-.5))],
            physical_pinhole_focal_HR=[c.fx,c.fy])
        output.append(record)
    result=dict(status='PASS_CPU_CALIBRATION_AND_GENERIC_WRAPPER_CONTRACT',GPU_calls=0,
        cuda_initialized=torch.cuda.is_initialized(),source_asset_sha256=sha(ASSET),
        wrapper_sha256=sha(HERE/'relative_area_asset.py'),views=output,generic_contract=generic_contract(),
        conclusion='Legacy native footprint projection and the PVD-derived new-chart sigma calibration are inconsistent; no global projection/camera/.so change was made.',
        scope='CPU diagnostic only; positive-focal pinhole covariance is an analytic comparison, not an adopted renderer or an extra GPU experiment.',
        neg_fov_semantics='tan_fovx=tan_fovy=tan(-0.5); negative focal lengths and native min(lim,max(-lim,q)) clamp order preserved for active footprint.',
        neural_weight_updates=0,geometry_updates=0,colour_updates=0,
        source_opacity_updates=0,automatic_adoption=False)
    assert not result['cuda_initialized']
    (DATA/'covariance_cpu_review.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],views=[dict(raw=x['raw'],ideal_minor=x['physical_pinhole_HR_unfiltered_sigma_minor']['median'],ideal_major=x['physical_pinhole_HR_unfiltered_sigma_major']['median'],native_minor=x['active_legacy_HR_unfiltered_sigma_minor']['median'],native_major=x['active_legacy_HR_unfiltered_sigma_major']['median'],nativeLRfiltered_minor=x['active_legacy_LR_filtered_sigma_minor']['median'],nativeLRfiltered_major=x['active_legacy_LR_filtered_sigma_major']['median']) for x in output]),indent=2))

if __name__=='__main__':main()

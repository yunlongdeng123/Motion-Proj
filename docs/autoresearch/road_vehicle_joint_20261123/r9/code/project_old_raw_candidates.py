"""Original Brown-domain teacher RGB -> fixed world-road observations, CPU only."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from PIL import Image


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def linear(x):return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4).astype(np.float32)


def srgb(x):
    x=np.clip(x,0,1);return np.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055)


def bilinear(a,xy):
    h,w=a.shape[:2];x=np.clip(np.nan_to_num(xy[...,0]),0,w-1);y=np.clip(np.nan_to_num(xy[...,1]),0,h-1)
    x0=np.floor(x).astype(int);y0=np.floor(y).astype(int);x1=np.minimum(x0+1,w-1);y1=np.minimum(y0+1,h-1)
    dx=x-x0;dy=y-y0;out=np.zeros((*xy.shape[:-1],*a.shape[2:]),np.float64)
    for yy,xx,weight in [(y0,x0,(1-dx)*(1-dy)),(y0,x1,dx*(1-dy)),(y1,x0,(1-dx)*dy),(y1,x1,dx*dy)]:
        out+=a[yy,xx]*(weight[...,None] if a.ndim==3 else weight)
    valid=np.isfinite(xy).all(-1)&(xy[...,0]>=0)&(xy[...,1]>=0)&(xy[...,0]<=w-1)&(xy[...,1]<=h-1)
    return out.astype(np.float32),valid


def all_four(a,xy):
    h,w=a.shape;x=np.clip(np.nan_to_num(xy[...,0]),0,w-1);y=np.clip(np.nan_to_num(xy[...,1]),0,h-1)
    x0=np.floor(x).astype(int);y0=np.floor(y).astype(int);x1=np.minimum(x0+1,w-1);y1=np.minimum(y0+1,h-1)
    return a[y0,x0]&a[y0,x1]&a[y1,x0]&a[y1,x1]


def project_world_to_model_brown(world,view,model_wh,native_wh):
    E=view['world_to_camera'];K=view['camera_K'];camera=world@E[:3,:3].T+E[:3,3]
    z=camera[...,2];den=np.where(z>1e-6,z,1.);xy=camera[...,:2]/den[...,None]
    pin=camera@K.T;pin=pin[...,:2]/den[...,None]-.5
    fx,fy,cx,cy,k1,k2,p1,p2,k3=map(float,view['Brown_intrinsics'])
    x,y=xy[...,0],xy[...,1];r2=x*x+y*y;radial=1+k1*r2+k2*r2*r2+k3*r2*r2*r2
    xd=x*radial+2*p1*x*y+p2*(r2+2*x*x);yd=y*radial+p1*(r2+2*y*y)+2*p2*x*y
    # Both K and Brown calibration use boundary coordinates; image samples index centers.
    native_boundary=np.stack((fx*xd+cx,fy*yd+cy),-1)
    model_xy=native_boundary*np.asarray(model_wh)/np.asarray(native_wh)-.5
    return pin,model_xy,z


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',type=Path,required=True);p.add_argument('--r7',type=Path,required=True)
    p.add_argument('--rectified-masks',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--semantics',type=Path,required=True);p.add_argument('--surface-visibility',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(args.sources.read_text(encoding='utf-8-sig'));rows=manifest['sources']
    geo=load(args.r7/'shared_grid_geometry.npz');seed=load(args.r7/'quilt_seed/atlas_seed.npz')
    world=geo['world_xyz'];uv=geo['uv'];domain=seed['write_domain'].astype(bool)&geo['geometry_valid']
    gains={int(r['raw']):float(r['gain']) for r in json.loads((args.r7/'exposure_gain.json').read_text())['records']}
    observations=[]
    for row in rows:
        raw=int(row['raw']);path=Path(row['raw_model']['path'])
        if not path.is_file():path=Path(row['raw_model']['local_path'])
        assert sha(path)==row['raw_model']['sha256']
        assert 'final_native' not in path.name and row['seed']==101 and row['prompt_id']=='legacy'
        with Image.open(path) as im:encoded=np.asarray(im.convert('RGB'),np.float32)/255.
        model_wh=[encoded.shape[1],encoded.shape[0]];native_wh=row['native_wh']
        assert model_wh==row['actual_model_wh']==[1152,768] and native_wh==[1920,1280]
        view_path=args.r7/f'projection_cache_v1/{raw:03d}/view_cache.npz';view=load(view_path)
        assert int(view['raw'])==raw and str(view['geometry_sha256'])==sha(args.r7/'shared_grid_geometry.npz')
        mask_path=args.rectified_masks/f'{raw:03d}/rectified_masks.npz';masks=load(mask_path)
        surface_path=args.surface_visibility/f'{raw:03d}/surface.npz';surface=load(surface_path)
        semantic_path=args.semantics/f'{raw:03d}/road_eroded_model_Brown.png'
        semantic_road=np.asarray(Image.open(semantic_path))>0
        pin,model_xy,z=project_world_to_model_brown(world,view,model_wh,native_wh)
        candidate,bounds=bilinear(linear(encoded),model_xy)
        cached_uv,_=bilinear(np.where(view['projection_valid'][...,None],view['atlas_uv_map'],0),pin)
        geometry=all_four(view['projection_valid'].astype(bool)&view['geometry_valid'].astype(bool)&surface['triangle_all_corners_geometry_valid'],pin)
        blocked=masks['protected']|masks['actor_visible_proxy']|~masks['RGB_valid']
        if 'unknown_potential' in masks:blocked|=masks['unknown_potential']
        protected_free=all_four(~blocked,pin)
        uv_error=np.linalg.norm(cached_uv-uv,axis=-1)
        pin_bounds=(pin[...,0]>=0)&(pin[...,1]>=0)&(pin[...,0]<=1151)&(pin[...,1]<=767)
        surface_depth,_=bilinear(surface['camera_depth_m'],pin)
        depth_error=np.abs(surface_depth-z);depth_tolerance=.03+.002*z
        geometric_candidate=domain&bounds&pin_bounds&(z>1e-6)&geometry&protected_free&(uv_error<=1.)&(depth_error<=depth_tolerance)
        semantic_good=all_four(semantic_road,model_xy)
        valid=geometric_candidate&semantic_good
        assert np.isfinite(candidate[valid]).all()
        cached_brown,_=bilinear(view['photo_source_brown_uv'],pin)
        cached_model_xy=(cached_brown+.5)*np.asarray(model_wh)/np.asarray(native_wh)-.5
        brown_error=np.linalg.norm(cached_model_xy-model_xy,axis=-1)
        assert valid.any() and brown_error[valid].max()<.02,'Direct Brown projection disagrees with the pinned inverse camera map'
        camera_center=-view['world_to_camera'][:3,:3].T@view['world_to_camera'][:3,3]
        ray=camera_center-world;ray/=np.maximum(np.linalg.norm(ray,axis=-1,keepdims=True),1e-12)
        incidence=np.abs((ray*geo['normal_world']).sum(-1))
        du=np.gradient(pin,axis=1);dv=np.gradient(pin,axis=0)
        area=np.abs(du[...,0]*dv[...,1]-du[...,1]*dv[...,0])
        weight=np.clip(np.sqrt(area),.1,3.)/(1+np.clip(np.sqrt(area),.1,3.))*np.clip(incidence,.05,1)
        weight=np.where(valid,weight,0).astype(np.float32)
        common=(candidate*gains[raw]).astype(np.float32);common[~valid]=0
        out=args.output/f'{raw:03d}';out.mkdir()
        np.savez_compressed(out/'world_observation.npz',canonical_linear_RGB=common,valid_mask=valid,weight=weight,
            model_Brown_xy=model_xy.astype(np.float32),native_pinhole_xy=pin.astype(np.float32),UV_error_texels=uv_error.astype(np.float32),
            surface_depth_error_m=depth_error.astype(np.float32),pre_semantic_geometric_candidate=geometric_candidate)
        # One inverse Brown gather for the pinhole preview; no hard-final image is read.
        source_xy=(view['photo_source_brown_uv']+.5)*np.asarray(model_wh)/np.asarray(native_wh)-.5
        preview,preview_valid=bilinear(linear(encoded),source_xy)
        Image.fromarray(np.rint(srgb(preview)*255).astype(np.uint8)).save(out/'raw_teacher_pinhole.png')
        rec=dict(raw=raw,seed=row['seed'],prompt_id=row['prompt_id'],raw_model_path=str(path),raw_model_sha256=sha(path),
            view_cache_sha256=sha(view_path),mask_sha256=sha(mask_path),exposure_gain=gains[raw],
            semantic_road_sha256=sha(semantic_path),surface_visibility_sha256=sha(surface_path),
            candidate_texels=int(valid.sum()),candidate_real_anchor_texels=int((valid&seed['same_world_lock']).sum()),
            pre_semantic_geometric_candidate_texels=int(geometric_candidate.sum()),semantic_nonroad_rejected_texels=int((geometric_candidate&~semantic_good).sum()),
            world_geometry_valid_texels=int(domain.sum()),projected_in_bounds_texels=int((domain&bounds&pin_bounds&(z>0)).sum()),
            protected_rejected_texels=int((domain&bounds&pin_bounds&~protected_free).sum()),
            geometry_certificate_rejected_texels=int((domain&bounds&pin_bounds&protected_free&~geometry).sum()),
            actual_UV_consistency_max_texels=float(uv_error[valid].max()) if valid.any() else None,
            direct_vs_inverse_Brown_max_error_model_px=float(brown_error[valid].max()),
            depth_error_max_m=float(depth_error[valid].max()),depth_tolerance='0.03 + 0.002 * camera_depth_m',UV_tolerance_texels=1.,
            raw_native_or_hard_final_used_as_RGB=False,original_W_used_as_adoption_mask=False,
            observation_sha256=sha(out/'world_observation.npz'))
        write(out/'receipt.json',rec);observations.append(rec)
    write(args.output/'receipt.json',dict(complete=True,CPU_only=True,source_manifest_sha256=sha(args.sources),
        geometry_sha256=sha(args.r7/'shared_grid_geometry.npz'),seed_sha256=sha(args.r7/'quilt_seed/atlas_seed.npz'),
        frames=[int(r['raw']) for r in rows],observations=observations,
        geometry='unchanged actual nonplanar world-road XYZ; no flattening or Gaussian edits',
        coordinate_chain='world XYZ -> calibrated camera -> Brown distortion -> original model1152 pixel; linear RGB before bilinear',
        protection='separate Brown-rectified retained vehicles and A20 mask; old W/final composite never used for RGB',
        new_diffusion_calls=0,nativeF=0,nativeB=0,optimizer_updates=0))
    print(json.dumps(observations,ensure_ascii=False))


if __name__=='__main__':main()

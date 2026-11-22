"""Independent Brown mapping audit and explicitly labelled pinhole previews; CPU only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def pin(path):
    p=Path(path)
    return {'path':p.resolve().as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def eotf(x):
    x=np.asarray(x,dtype=np.float64)
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)


def oetf(x):
    x=np.maximum(x,0)
    return np.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055)


def sample(image,uv):
    uv=np.asarray(uv,dtype=np.float64)
    finite=np.isfinite(uv).all(-1)
    safe=np.where(finite[...,None],uv,0)
    x=np.floor(safe[...,0]).astype(np.int64);y=np.floor(safe[...,1]).astype(np.int64)
    h,w=image.shape[:2]
    valid=finite&(x>=0)&(y>=0)&(x+1<w)&(y+1<h)
    ix=np.clip(x,0,w-2);iy=np.clip(y,0,h-2)
    fx=np.clip(safe[...,0]-x,0,1);fy=np.clip(safe[...,1]-y,0,1)
    values=(image[iy,ix]*(1-fx)[...,None]*(1-fy)[...,None]
            +image[iy,ix+1]*fx[...,None]*(1-fy)[...,None]
            +image[iy+1,ix]*(1-fx)[...,None]*fy[...,None]
            +image[iy+1,ix+1]*fx[...,None]*fy[...,None])
    return np.where(valid[...,None],values,0),valid,(ix,iy)


def brown_uv(xy,intr):
    fx,fy,cx,cy,k1,k2,p1,p2,k3=intr
    x,y=xy[...,0],xy[...,1]; r2=x*x+y*y
    radial=1+k1*r2+k2*r2*r2+k3*r2*r2*r2
    xd=x*radial+2*p1*x*y+p2*(r2+2*x*x)
    yd=y*radial+p1*(r2+2*y*y)+2*p2*x*y
    return np.stack([fx*xd+cx-.5,fy*yd+cy-.5],-1)


def stats(x):
    x=np.asarray(x); x=x[np.isfinite(x)]
    return {'count':int(x.size),'mean':float(np.mean(x)),'p50':float(np.percentile(x,50)),
            'p95':float(np.percentile(x,95)),'p99':float(np.percentile(x,99)),'max':float(np.max(x))} if x.size else {'count':0}


def save_rgb(path,linear):
    Image.fromarray(np.clip(oetf(linear)*255+.5,0,255).astype(np.uint8)).save(path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work-root',type=Path,default=Path(__file__).resolve().parents[2]);a=ap.parse_args()
    work=a.work_root.resolve();r9=work/'old_raw_teacher_joint_20261123_r9';audit=r9/'audit'
    manifest_path=audit/'raw_teacher_sources.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    geom_path=work/'road_shared_appearance_20261122_r7/shared_grid_geometry.npz'
    geom=np.load(geom_path,allow_pickle=False);xyz=geom['world_xyz'];atlas_uv=geom['uv'];scope=geom['write_domain']&geom['geometry_valid']
    rows=[];contact=[]
    for source in manifest['sources']:
        raw=source['raw'];out=audit/'mapping'/f'{raw:03d}';out.mkdir(parents=True,exist_ok=True)
        cache_path=work/'road_shared_appearance_20261122_r7/projection_cache_v1'/f'{raw:03d}'/'view_cache.npz';c=np.load(cache_path,allow_pickle=False)
        k=c['camera_K'];brown=c['Brown_intrinsics'];E=c['world_to_camera'];H,W=c['projection_valid'].shape
        yy,xx=np.mgrid[:H,:W];rays=np.stack([xx+.5,yy+.5,np.ones_like(xx)],-1)@np.linalg.inv(k).T
        analytic_uv=brown_uv(rays[...,:2]/rays[...,2:],brown)
        cached_uv=c['photo_source_brown_uv'].astype(np.float64)
        max_error=np.max(np.abs(analytic_uv-cached_uv));assert max_error<.0002,max_error
        model_path=Path(source['raw_model']['local_path']);model=eotf(np.asarray(Image.open(model_path).convert('RGB'))/255.)
        model_uv=(analytic_uv+.5)*np.array([1152/1920,768/1280])-.5
        raw_linear,raw_valid,_=sample(model,model_uv)
        final_path=audit/'source_inputs'/f'{raw:03d}'/'final_native.png'
        hard_linear,hard_valid,_=sample(eotf(np.asarray(Image.open(final_path).convert('RGB'))/255.),analytic_uv)
        save_rgb(out/'raw_model_pinhole_preview.png',raw_linear)
        save_rgb(out/'hard_final_pinhole_audit_only.png',hard_linear)
        # No final-native RGB enters any teacher samples. This NPZ carries RAW only.
        np.savez_compressed(out/'raw_model_pinhole_preview.npz',raw_linear=raw_linear.astype(np.float32),
                            valid=raw_valid,brown_native_pixel_index_uv=analytic_uv.astype(np.float32),
                            brown_model_pixel_index_uv=model_uv.astype(np.float32),source_raw_model_sha256=source['raw_model']['sha256'])
        r8_path=work/'road_vehicle_joint_dggt_20261123_r8/audit/mask_coordinate_audit/rectified_masks'/f'{raw:03d}'/'rectified_masks.npz'
        r8=np.load(r8_path,allow_pickle=False);protected=r8['protected'];new_rgb_valid=r8['RGB_valid']
        overlay=np.asarray(Image.open(out/'raw_model_pinhole_preview.png').convert('RGB')).copy()
        overlay[protected]=np.clip(.4*overlay[protected]+.6*np.array([255,80,40]),0,255).astype(np.uint8)
        Image.fromarray(overlay).save(out/'raw_preview_R8_protected_overlay.png')
        # Independent world-center -> camera -> Brown -> model sample. Four-corner
        # cache UV agreement is a diagnostic distribution, not a silently chosen gate.
        camera=xyz@E[:3,:3].T+E[:3,3];z=camera[...,2]
        xy=camera[...,:2]/np.maximum(z[...,None],1e-12)
        p=xy@k[:2,:2].T+k[:2,2]-.5
        gathered_uv,bounded,(ix,iy)=sample(c['atlas_uv_map'].astype(np.float64),p)
        cache_valid=c['projection_valid']&c['geometry_valid']
        all4=(cache_valid[iy,ix]&cache_valid[iy,ix+1]&cache_valid[iy+1,ix]&cache_valid[iy+1,ix+1])
        eligible=scope&(z>0)&bounded&all4
        inverse_uv_error=np.linalg.norm(gathered_uv-atlas_uv,axis=-1)
        direct_model_uv=(brown_uv(xy,brown)+.5)*.6-.5
        direct_rgb,direct_valid,_=sample(model,direct_model_uv)
        wrong_same_pixel_shift=np.linalg.norm(model_uv-np.stack([xx,yy],-1),axis=-1)
        # Comparing the two camera routes uses Brown coordinate interpolation only;
        # RGB interpolation is not commutative and remains a separate operation.
        route_brown,route_valid,_=sample(analytic_uv,p)
        direct_brown=brown_uv(xy,brown)
        route_error=np.linalg.norm(route_brown-direct_brown,axis=-1)
        mask=eligible&route_valid
        row=dict(raw=raw,source=pin(model_path),projection_cache=pin(cache_path),R8_pinhole_protection=pin(r8_path),
            Brown_analytic_vs_cached_max_abs_native_pixels=float(max_error),
            pinhole_preview_valid_pixels=int(raw_valid.sum()),new_protected_pixels=int(protected.sum()),
            source_direct_world_samples_finite=bool(np.isfinite(direct_rgb[eligible&direct_valid]).all()),
            world_scope_texels=int(scope.sum()),four_corner_geometry_supported_scope_texels=int(eligible.sum()),
            inverse_UV_error_texels_diagnostic=stats(inverse_uv_error[eligible]),
            world_to_Brown_direct_vs_inverse_map_native_px=stats(route_error[mask]),
            wrong_Brown_as_pinhole_shift_model_px=stats(wrong_same_pixel_shift[new_rgb_valid]),
            crop_native_xyxy_half_open=source['crop_native_xyxy_half_open'],source_domain='Brown resized 1152x768',
            preview_domain='pinhole 1152x768, camera_K boundary convention; pixel indices subtract .5',
            interpolation='exact raw source sRGB EOTF first, bilinear in linear RGB; encode for PNG display',
            old_CS_W_K_used_for_qualification=False,hard_final_RGB_used_for_supervision=False,
            views_are_independent_timestamps=True,
            new_protection_is_proxy=True,new_protection_is_semantic_road_GT=False,
            outputs={name:pin(out/name) for name in ['raw_model_pinhole_preview.png','hard_final_pinhole_audit_only.png','raw_preview_R8_protected_overlay.png','raw_model_pinhole_preview.npz']})
        (out/'receipt.json').write_text(json.dumps(row,indent=2)+'\n',encoding='utf-8');rows.append(row)
        panel=Image.new('RGB',(1152*2,768+38),'#151a21');draw=ImageDraw.Draw(panel)
        draw.text((10,10),f'raw{raw:03d}  RAW TEACHER: Brown -> pinhole preview',fill='white')
        draw.text((1162,10),'HARD FINAL: audit only, no RGB supervision',fill='white')
        panel.paste(Image.open(out/'raw_model_pinhole_preview.png'),(0,38));panel.paste(Image.open(out/'hard_final_pinhole_audit_only.png'),(1152,38));contact.append(panel)
    image=Image.new('RGB',(2304,806*len(contact)))
    for i,panel in enumerate(contact):image.paste(panel,(0,i*806))
    image.save(audit/'mapping/pinhole_source_comparison.png')
    result=dict(schema='R9.Brown_teacher_mapping_audit.v1',complete=True,decision='PASS_COORDINATE_AUDIT',
        road_surface_acceptance='NOT_EVALUATED_BY_THIS_COORDINATE_AUDIT',source_manifest=pin(manifest_path),
        shared_geometry=pin(geom_path),rows=rows,contactsheet=pin(audit/'mapping/pinhole_source_comparison.png'),
        CPU_only=True,new_GPU_forward=0,new_diffusion_calls=0,training_updates=0,
        geometric_four_corner_counts_are_diagnostic=True,geometric_acceptance_threshold_not_chosen_here=True)
    (audit/'brown_mapping_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'receipt':pin(audit/'brown_mapping_audit.json'),'rows':[{k:v for k,v in x.items() if k in ['raw','Brown_analytic_vs_cached_max_abs_native_pixels','world_to_Brown_direct_vs_inverse_map_native_px','wrong_Brown_as_pinhole_shift_model_px','inverse_UV_error_texels_diagnostic']} for x in rows]}))


if __name__=='__main__':main()

"""Project original raw teachers onto an explicitly supplied local road model.

CPU candidate generation only. The supplied inferred model remains a separate
geometry hypothesis; raster visibility is not evidence of measured geometry.
"""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
from PIL import Image

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['r7','r9','rectified-masks','geometry','sources','output']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--scope-key',default='candidate_mask')
    a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
    sys.path.insert(0,str(a.r9/'code'))
    from project_old_raw_candidates import load,sha,write,linear,bilinear,all_four,project_world_to_model_brown
    from local_surface_projection import raster_surface
    started=time.time();g=load(a.geometry);world=g['world_xyz'];uv=g['uv_global']
    scope=g[a.scope_key].astype(bool);normal=g['normal_world']
    assert world.shape==(*scope.shape,3) and uv.shape==(*scope.shape,2)
    assert np.array_equal(uv[scope],np.rint(uv[scope]))
    assert np.isfinite(world[scope]).all() and np.isfinite(normal[scope]).all()
    gain={int(r['raw']):float(r['gain']) for r in json.loads((a.r7/'exposure_gain.json').read_text())['records']}
    records=[]
    for row in json.loads(a.sources.read_text(encoding='utf-8-sig'))['sources']:
        raw=int(row['raw']);path=Path(row['raw_model']['path'])
        if not path.is_file():path=Path(row['raw_model']['local_path'])
        assert sha(path)==row['raw_model']['sha256'] and path.name=='raw_model.png'
        assert row['seed']==101 and row['prompt_id']=='legacy'
        encoded=np.asarray(Image.open(path).convert('RGB'),np.float32)/255
        assert encoded.shape==(768,1152,3) and row['native_wh']==[1920,1280]
        vp=a.r7/f'projection_cache_v1/{raw:03d}/view_cache.npz';v=load(vp)
        mp=a.rectified_masks/f'{raw:03d}/rectified_masks.npz';m=load(mp)
        sp=a.r9/f'raw_road_semantic_v2/{raw:03d}/road_eroded_model_Brown.png'
        road=np.asarray(Image.open(sp))>0
        pin,brown,z=project_world_to_model_brown(world,v,[1152,768],[1920,1280])
        candidate,bounds=bilinear(linear(encoded),brown)
        # Perspective-correct, independently generated visibility of this model,
        # rather than the old mesh's unsupported hole coordinates.
        surface=raster_surface(world,uv,scope,v['world_to_camera'],v['camera_K'],[1152,768])
        mapped,_=bilinear(np.where(surface['projection_valid'][...,None],surface['atlas_uv_global'],0),pin)
        visible=all_four(surface['projection_valid'],pin)
        uv_error=np.linalg.norm(mapped-uv,axis=-1)
        depth,_=bilinear(surface['camera_depth_m'],pin);depth_error=np.abs(depth-z)
        blocked=m['protected']|m['actor_visible_proxy']|~m['RGB_valid']
        if 'unknown_potential' in m:blocked|=m['unknown_potential']
        free=all_four(~blocked,pin)
        pinbounds=(pin[...,0]>=0)&(pin[...,1]>=0)&(pin[...,0]<=1151)&(pin[...,1]<=767)
        pre=scope&bounds&pinbounds&(z>0)&visible&free&(uv_error<=1)&(depth_error<=.03+.002*z)
        valid=pre&all_four(road,brown)
        centre=-v['world_to_camera'][:3,:3].T@v['world_to_camera'][:3,3]
        ray=centre-world;ray/=np.maximum(np.linalg.norm(ray,axis=-1,keepdims=True),1e-12)
        incidence=np.abs((ray*normal).sum(-1));du=np.gradient(pin,axis=1);dv=np.gradient(pin,axis=0)
        area=np.abs(du[...,0]*dv[...,1]-du[...,1]*dv[...,0]);s=np.clip(np.sqrt(area),.1,3)
        weight=np.where(valid,s/(1+s)*np.clip(incidence,.05,1),0).astype(np.float32)
        common=(candidate*gain[raw]).astype(np.float32);common[~valid]=0
        out=a.output/f'{raw:03d}';out.mkdir()
        np.savez_compressed(out/'world_observation.npz',canonical_linear_RGB=common,
            valid_mask=valid,weight=weight,uv_global=uv,model_Brown_xy=brown.astype(np.float32),
            native_pinhole_xy=pin.astype(np.float32),UV_error_texels=uv_error.astype(np.float32),
            surface_depth_error_m=depth_error.astype(np.float32),pre_semantic_geometric_candidate=pre)
        np.savez_compressed(out/'local_surface_projection.npz',**surface)
        rec=dict(raw=raw,raw_model=dict(path=str(path),sha256=sha(path)),view_cache_sha256=sha(vp),
            mask_sha256=sha(mp),semantic_sha256=sha(sp),exposure_gain=gain[raw],
            local_scope_vertices=int(scope.sum()),candidate_texels=int(valid.sum()),
            before_semantic_texels=int(pre.sum()),protected_rejected=int((scope&bounds&pinbounds&~free).sum()),
            max_UV_error_texels=float(uv_error[valid].max()) if valid.any() else None,
            max_depth_error_m=float(depth_error[valid].max()) if valid.any() else None,
            observation_sha256=sha(out/'world_observation.npz'))
        write(out/'receipt.json',rec);records.append(rec)
    write(a.output/'receipt.json',dict(schema='R10.local_raw_teacher_world_candidates.v1',complete=True,
        CPU_only=True,geometry=dict(path=str(a.geometry),sha256=sha(a.geometry)),scope_key=a.scope_key,
        inferred_geometry_not_measured=True,geometry_acceptance='SEPARATE; not asserted by candidate projection',
        source_manifest_sha256=sha(a.sources),frames=[r['raw'] for r in records],observations=records,
        mapping='world -> calibrated Brown -> original raw model1152; linearize before bilinear',
        hard_final_or_old_W_used=False,first_writer_locks=False,new_diffusion_calls=0,new_semantic_calls=0,
        GPU_forward=0,GPU_backward=0,Gaussian_updates=0,seconds=time.time()-started))
    print(json.dumps(dict(complete=True,counts={r['raw']:r['candidate_texels'] for r in records})))

if __name__=='__main__':main()

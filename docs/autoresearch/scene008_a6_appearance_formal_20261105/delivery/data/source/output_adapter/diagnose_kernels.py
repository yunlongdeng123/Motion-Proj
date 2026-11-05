"""Two-arm frozen-point output-kernel diagnostic; no fitting or PVD updates."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import binary_erosion
import torch

CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
RUN=CAM/'scene008_a6_appearance_formal_20261105'
OLD=CAM/'scene008_a6_asset_controls_20261105'
SOURCE=RUN/'highres/formal/complete_a6.npz'
OUT=RUN/'output_adapter'
sys.path.insert(0,str(RUN/'source'))
from highres_fit import camera_hr
from scene_runtime import Runtime,raster,tagged
from sh_asset import SHAsset,hash_geometry
from full_pvd import GEOMETRY_FIELDS


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,v):Path(path).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rgb8(a):return np.round(np.clip(a,0,1)*255).astype(np.uint8)
def npz(path,arrays):
    with np.load(path,allow_pickle=False) as saved:return {k:saved[k].copy() for k in saved.files}


def stats(a):
    a=np.asarray(a,np.float64)
    q=np.quantile(a,[0,.01,.1,.5,.9,.99,1])
    return dict(min=q[0],p01=q[1],p10=q[2],median=q[3],p90=q[4],p99=q[5],max=q[6],mean=float(a.mean()))


def metric(a,mask):
    v=a[mask]
    return dict(count=int(len(v)),mean=float(v.mean()),min=float(v.min()),max=float(v.max())) if len(v) else dict(count=0)


def depth_vis(depth,alpha,low,high):
    t=np.clip((depth-low)/max(high-low,1e-6),0,1)
    # Shared blue-to-yellow ramp; exact depths are in theNPZ.
    value=np.stack((t,.25+.6*(1-np.abs(2*t-1)),1-t),-1)
    value[alpha<=.1]=0
    return rgb8(value)


@torch.no_grad()
def main():
    OUT.mkdir(exist_ok=False);start=time.time()
    original_hash=sha(SOURCE)
    arrays=npz(SOURCE,None);m2=npz(OLD/'m2/complete_a6.npz',None)
    assert len(arrays['xyz'])==18432
    assert all(np.array_equal(arrays[k],m2[k]) for k in GEOMETRY_FIELDS)
    original_geometry_hash=hash_geometry(arrays)
    assert original_geometry_hash==str(arrays['geometry_hash'])
    oldparent=CAM/'scene008_a6_complete_asset_20261105/appearance/complete_a6_fitted.npz'
    parent=npz(oldparent,None)
    geometry_stats=dict(units='canonical local metres before actor pose uniform scale',
        M2=dict(rows=len(m2['xyz']),sigma_m=stats(m2['sigma']),thickness_m=stats(m2['thickness']),
                thickness_over_sigma=stats(m2['thickness']/m2['sigma']),opacity=stats(m2['opacity'])),
        parent2048=dict(rows=len(parent['xyz']),sigma_m=stats(parent['sigma']),
                        thickness_m=stats(parent['thickness']),thickness_over_sigma=stats(parent['thickness']/parent['sigma'])),
        input_HR_geometry_identical_to_M2=True)
    write(OUT/'sigma_thickness_stats.json',geometry_stats)
    print(json.dumps(dict(stage='maternal_geometry_stats',**geometry_stats)),flush=True)
    sphere_arrays={k:v.copy() for k,v in arrays.items()}
    sphere_arrays['thickness']=sphere_arrays['sigma'].copy()
    sphere_hash=hash_geometry(sphere_arrays)
    assert sphere_hash!=original_geometry_hash
    sphere_arrays['geometry_hash']=np.asarray(sphere_hash)
    metadata=json.loads(str(arrays.get('metadata_json','{}')))
    metadata['output_kernel_diagnostic']=dict(arm='sphere_sigma',source_asset=str(SOURCE),
        source_sha256=original_hash,source_geometry_hash=original_geometry_hash,
        changed_fields=['thickness'],thickness_policy='thickness=sigma; sigma unchanged',
        PVD_weight_updates=0,appearance_updates=0,point_positions_changed=False,
        point_count_changed=False,opacity_changed=False,parent_provenance_changed=False,
        geometry_hash_changed=True)
    surface=SHAsset(arrays,json.loads(str(arrays.get('metadata_json','{}')))).cuda()
    sphere=SHAsset(sphere_arrays,metadata).cuda()
    assert surface.geometry_hash==original_geometry_hash and sphere.geometry_hash==sphere_hash
    assert surface.provenance_hash==sphere.provenance_hash
    for key in GEOMETRY_FIELDS:
        if key!='thickness':assert torch.equal(getattr(surface,key),getattr(sphere,key)),key
    assert torch.equal(surface.coefficients,sphere.coefficients)
    sphere_export=sphere.export(OUT/'sphere_sigma_asset.npz')
    # Round-trip requires the new correct hash and identical point/material data.
    reloaded=SHAsset.from_geometry(OUT/'sphere_sigma_asset.npz')
    assert reloaded.geometry_hash==sphere_hash
    assert np.array_equal(reloaded.xyz.numpy(),arrays['xyz'])
    assert np.array_equal(reloaded.coefficients.numpy(),arrays['canonical_sh'])
    rt=Runtime();frames=[]
    for raw in (106,113):
        d=rt.raw(raw);c=camera_hr(d['cam']);renders={}
        for name,asset in (('surface',surface),('sphere_sigma',sphere)):
            per=OUT/f'raw{raw:03d}'/name;per.mkdir(parents=True)
            rows=asset.rows(d['pose'],c,rt.pipe,1,precomputed=True)
            out=raster(c,rt.pipe,0,tagged(rows,'asset'),rt.n.bg.new_zeros(3))
            alpha=out['alpha'][0].cpu().numpy()
            rgb=out['rgb'].permute(1,2,0).cpu().numpy()
            depth_sum=out['depth_sum'][0].cpu().numpy()
            depth=depth_sum/np.maximum(alpha,1e-6)/rt.scale
            depth[alpha<=1e-6]=0
            assert np.isfinite(rgb).all() and np.isfinite(alpha).all() and np.isfinite(depth).all()
            Image.fromarray(rgb8(rgb)).save(per/'rgb_black_hr.png')
            Image.fromarray(np.round(np.clip(alpha,0,1)*65535).astype(np.uint16)).save(per/'alpha16_hr.png')
            Image.fromarray(rgb8(alpha)).save(per/'alpha_hr.png')
            np.savez_compressed(per/'raster_hr.npz',rgb_premultiplied=rgb,alpha=alpha,
                expected_depth_m=depth,depth_sum_model=depth_sum,
                meter_to_model=np.float64(rt.scale),geometry_hash=np.asarray(asset.geometry_hash),
                raw=np.int64(raw),source_rgb_coefficients_sha256=np.asarray(hashlib.sha256(arrays['canonical_sh'].tobytes()).hexdigest()))
            renders[name]=dict(rgb=rgb,alpha=alpha,depth=depth)
        s=renders['surface'];b=renders['sphere_sigma']
        union=(s['alpha']>.1)|(b['alpha']>.1)
        ys,xs=np.where(union);side=int(max(xs.max()-xs.min()+1,ys.max()-ys.min()+1)+48)
        left=int(np.floor((xs.min()+xs.max()+1-side)/2));top=int(np.floor((ys.min()+ys.max()+1-side)/2))
        box=(left,top,left+side,top+side)
        low,high=np.quantile(np.concatenate((s['depth'][s['alpha']>.1],b['depth'][b['alpha']>.1])),[.01,.99])
        panel=Image.new('RGB',(3*460,2*(460+30)),(236,242,246));draw=ImageDraw.Draw(panel)
        for j,name in enumerate(('surface','sphere_sigma')):
            v=renders[name]
            depthimage=depth_vis(v['depth'],v['alpha'],float(low),float(high))
            Image.fromarray(depthimage).save(OUT/f'raw{raw:03d}'/name/'depth_vis_hr.png')
            images=(rgb8(v['rgb']),np.repeat(rgb8(v['alpha'])[...,None],3,2),depthimage)
            for i,(image,label) in enumerate(zip(images,('black RGB','alpha','expected camera z (m)'))):
                draw.text((i*460+8,j*490+8),f'{name}: {label}',fill=(12,20,27))
                tile=Image.fromarray(image).crop(box).resize((460,460),Image.Resampling.BILINEAR)
                panel.paste(tile,(i*460,j*490+30))
        panel.save(OUT/f'raw{raw:03d}_compare.png')
        # Fixed baseline-defined domains prevent footprint changes from hiding
        # low-alpha holes in this mechanism diagnostic.
        interior=binary_erosion(s['alpha']>.1,iterations=4)
        visible=np.repeat(np.repeat(d['a6_visible'].cpu().numpy(),4,0),4,1)
        common=(s['alpha']>.9)&(b['alpha']>.9)
        rows=[]
        for name in ('surface','sphere_sigma'):
            v=renders[name]
            rows.append(dict(arm=name,geometry_hash=original_geometry_hash if name=='surface' else sphere_hash,
                alpha_fixed_surface_interior=metric(v['alpha'],interior),
                interior_below05=int((v['alpha'][interior]<.5).sum()),
                interior_below09=int((v['alpha'][interior]<.9).sum()),
                alpha_fixed_original_visible=metric(v['alpha'],visible),
                object_alpha_gt01=int((v['alpha']>.1).sum()),object_alpha_gt09=int((v['alpha']>.9).sum())))
        record=dict(raw=raw,arms=rows,crop_xyxy=list(box),depth_visualization_range_m=[float(low),float(high)],
            alpha_maxabs_change=float(np.abs(b['alpha']-s['alpha']).max()),
            alpha_meanabs_change_fixed_surface_interior=float(np.abs(b['alpha'][interior]-s['alpha'][interior]).mean()),
            expected_depth_meanabs_change_common_opaque_m=float(np.abs(b['depth'][common]-s['depth'][common]).mean()),
            RGB_meanabs_change_fixed_surface_interior=float(np.abs(b['rgb'][interior]-s['rgb'][interior]).mean()),
            note='Kernel-shape output diagnostic only; no hidden-surface RGB truth or quality acceptance')
        write(OUT/f'raw{raw:03d}'/'comparison.json',record);frames.append(record)
        print(json.dumps(dict(stage='frame_complete',**record)),flush=True)
        del d,out,rows,renders
        rt._data.clear();torch.cuda.empty_cache()
    assert surface.assert_geometry_unchanged() and sphere.assert_geometry_unchanged() and rt.modelunchanged()
    assert sha(SOURCE)==original_hash
    write(OUT/'receipt.json',dict(status='PASS',source_asset=str(SOURCE),source_sha256=original_hash,
        source_geometry_hash=original_geometry_hash,sphere_geometry_hash=sphere_hash,
        source_RGB_coefficients_identical=True,xyz_identical=True,sigma_identical=True,opacity_identical=True,
        source_and_parent_provenance_identical=True,rows=18432,arms=['surface','sphere_sigma'],
        changed_geometry_fields=['thickness'],PVD_weight_updates=0,appearance_updates=0,
        density_updates=0,no_HELD_RGB_used_for_fitting=True,no_fitting_performed=True,
        sphere_export=sphere_export,frames=frames,seconds=time.time()-start,
        peak_vram_bytes=int(torch.cuda.max_memory_allocated())))
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='file_manifest.json']
    write(OUT/'file_manifest.json',dict(files=[dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)]))
    print(json.dumps(dict(stage='complete',status='PASS',seconds=time.time()-start)),flush=True)


if __name__=='__main__':main()

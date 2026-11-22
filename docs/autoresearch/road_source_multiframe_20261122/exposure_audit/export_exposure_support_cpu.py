"""导出真实gain拟合/外部验证取样的Brown原图像素支撑；只写新审计文件。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def exact(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)


def brown(q,intrinsic):
    fx,fy,cx,cy,k1,k2,p1,p2,k3=intrinsic;x,y=(q[:,:2]/q[:,2:]).T;r=x*x+y*y
    radial=1+k1*r+k2*r*r+k3*r*r*r
    return np.c_[fx*(x*radial+2*p1*x*y+p2*(r+2*x*x))+cx-.5,
                 fy*(y*radial+p1*(r+2*y*y)+2*p2*x*y)+cy-.5]


def pixel_support(uv):
    mask=np.zeros((1280,1920),bool)
    flo=np.floor(uv).astype(int);fraction=uv-flo
    for dy in (0,1):
        for dx in (0,1):
            weight=(fraction[:,0] if dx else 1-fraction[:,0])*(fraction[:,1] if dy else 1-fraction[:,1])
            x=flo[:,0]+dx;y=flo[:,1]+dy;valid=(weight>0)&(x>=0)&(x<1920)&(y>=0)&(y<1280)
            mask[y[valid],x[valid]]=True
    return mask


def main():
    C=Path('/root/autodl-tmp/drivingeditor_repro/campaign');R=C/'road_source_multiframe_20261122_r3';out=R/'exposure_audit'
    exposure=json.loads((out/'shared_exposure.json').read_text());table=load(out/'external_correspondence_samples.npz')
    source1=C/'road_gaussian_transfer_20261122_r1/source_selection/candidates/patch_02.npz'
    source2=C/'road_gaussian_transfer_20261122_r2_isometric/source_selection/candidates/patch_02.npz'
    p1,p2=load(source1),load(source2)
    checks={k:exact(v,p2[k]) for k,v in p1.items()};assert set(p1)==set(p2) and all(checks.values())
    equivalence=dict(schema='source02_packet_path_equivalence.v1',source_in_exposure=str(source1),source_in_pool=str(source2),
        exposure_resolved=str(source1.resolve()),pool_resolved=str(source2.resolve()),same_physical_file=source1.samefile(source2),
        all_fields_exact=True,field_checks=checks,source_core_count=int(p1['core_mask'].sum()),
        world_core_ROI_exact=True,source_center_exact=True,non_RGB_exact=True)
    (out/'source_packet_equivalence.json').write_text(json.dumps(equivalence,indent=2)+'\n')
    F=p1['source_tangent_frame_world'][:3,:3];O=p1['source_center_world'];half=p1['core_size_m']/2
    union=table['world_query_ids'];xyz=table['world_xyz'];rows=[];fitkeys=set();valkeys=set()
    bank=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs')
    for r in exposure['records']:
        raw=r['raw'];prefix=f'raw_{raw:03d}_'
        qids=table[prefix+'query_id'];shared=table[prefix+'shared_query_id'];uv=table[prefix+'Brown_uv']
        positions=np.searchsorted(qids,shared);assert np.array_equal(qids[positions],shared)
        worldpos=np.searchsorted(union,shared);assert np.array_equal(union[worldpos],shared)
        local=(xyz[worldpos]-O)@F
        assert ((abs(local[:,:2])>half+.20).any(1)).all()
        keys=np.floor(local[:,:2]/.08).astype(np.int64);unique,inverse=np.unique(keys,axis=0,return_inverse=True)
        assert np.array_equal(unique,table[prefix+'cell_keys'])
        val=table[prefix+'validation_cells'];fitpoints=~val[inverse];valpoints=val[inverse]
        fitkeys.update(map(tuple,unique[~val]));valkeys.update(map(tuple,unique[val]))
        fitmask=pixel_support(uv[positions][fitpoints]);valmask=pixel_support(uv[positions][valpoints])
        qualified=load(C/f'road_multisource_20261122_r1/sources/qualified_observations/{raw:03d}.npz')
        T=qualified['world_to_camera'];corners=p1['source_world_roi_corners']@T[:3,:3].T+T[:3,3]
        obs=json.loads((bank/f'observations/{raw:03d}.json').read_text());projected=brown(corners,obs['camera_intrinsic_Brown'])
        im=Image.new('L',(1920,1280));ImageDraw.Draw(im).polygon([tuple(q) for q in projected],fill=255);core=np.asarray(im)>0
        assert not (fitmask&core).any() and not (valmask&core).any()
        folder=out/'masks'/f'{raw:03d}';folder.mkdir(parents=True,exist_ok=True)
        fitpath=folder/'fit_Brown_bilinear_support.png';valpath=folder/'validation_Brown_bilinear_support.png'
        Image.fromarray(fitmask.astype(np.uint8)*255).save(fitpath);Image.fromarray(valmask.astype(np.uint8)*255).save(valpath)
        rows.append(dict(raw=raw,calibration_id='SOURCE02-EXTERNAL-GAIN-20261122-R3-ANCHOR115',
            fit_mask=str(fitpath),validation_mask=str(valpath),
            fit_raw_bilinear_support_pixels=int(fitmask.sum()),validation_raw_bilinear_support_pixels=int(valmask.sum()),
            fit_validation_raw_support_overlap_pixels=int((fitmask&valmask).sum()),
            fit_matching_world_queries=int(fitpoints.sum()),validation_matching_world_queries=int(valpoints.sum()),
            fit_world_cells=int((~val).sum()),validation_world_cells=int(val.sum()),
            fit_core_overlap_pixels=0,validation_core_overlap_pixels=0,
            correspondence_table=str(out/'external_correspondence_samples.npz'),
            world_cell_keys_field=prefix+'cell_keys',validation_world_cell_flags_field=prefix+'validation_cells'))
    assert not fitkeys&valkeys
    result=dict(schema='shared_exposure_raw_support_masks.v1',complete=True,
        calibration_id='SOURCE02-EXTERNAL-GAIN-20261122-R3-ANCHOR115',records=rows,
        all_fit_and_validation_world_cell_keys_disjoint=True,all_masks_outside_source_core=True,
        fit_unique_world_cells=len(fitkeys),validation_unique_world_cells=len(valkeys),
        pixel_counts_are_not_regression_denominators=True,query_core_samples_used=0,
        all_gains_unchanged=True,GPU_calls=0,optimizer_updates=0)
    (out/'calibration_support_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(support=result,source_packet_equivalence=equivalence),indent=2))


if __name__=='__main__':main()

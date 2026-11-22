"""核心之外真实同世界道路对应的共享标量曝光；只读原始输入，CPU 中位数估计。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import argparse,json,time
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates,binary_dilation,binary_erosion,gaussian_filter

TRAIN=[88,90,92,94,96,99,104,115]
QUERY=[86,100,110,117]
FRAMES=sorted(set(TRAIN+QUERY))
LUMA=np.array([.2126,.7152,.0722])


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def read(path):return json.loads(Path(path).read_text())


def sample(a,uv,order=1,cval=0):
    if a.ndim==2:return map_coordinates(a,[uv[:,1],uv[:,0]],order=order,mode='constant',cval=cval,prefilter=False)
    return np.stack([sample(a[:,:,c],uv,order,cval) for c in range(a.shape[2])],1)


def linear(x):return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)


def guard(shape,rectangles,fg):
    mask=np.zeros(shape,bool)
    for x0,y0,x1,y1 in rectangles:
        x0,x1=max(0,int(np.floor(x0))),min(shape[1],int(np.ceil(x1)))
        y0,y1=max(0,int(np.floor(y0))),min(shape[0],int(np.ceil(y1)))
        if x1>x0 and y1>y0:mask[y0:y1,x0:x1]=True
    return binary_dilation(mask,iterations=2)|binary_dilation(fg,iterations=2)


def brown(q,intrinsic):
    fx,fy,cx,cy,k1,k2,p1,p2,k3=intrinsic
    x,y=(q[:,:2]/q[:,2:]).T;r=x*x+y*y
    radial=1+k1*r+k2*r*r+k3*r*r*r
    return np.c_[fx*(x*radial+2*p1*x*y+p2*(r+2*x*x))+cx-.5,
                 fy*(y*radial+p1*(r+2*y*y)+2*p2*x*y)+cy-.5]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--campaign',type=Path,default=Path('/root/autodl-tmp/drivingeditor_repro/campaign'))
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();started=time.monotonic();C=args.campaign
    out=args.output
    if (out/'shared_exposure.json').exists():raise FileExistsError('Preserve frozen exposure file')
    out.mkdir(parents=True,exist_ok=True)
    bank=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs');B=bank.parent
    S=C/'road_multisource_20261122_r1/sources'
    source_path=C/'road_gaussian_transfer_20261122_r1/source_selection/candidates/patch_02.npz'
    source=load(source_path);F=source['source_tangent_frame_world'][:3,:3];O=source['source_center_world'];half=source['core_size_m']/2
    with np.load(S/'geometry.npz',allow_pickle=False) as z:X=z['world_xyz']
    local=(X-O)@F
    # 物理核心0.6m方形外再留0.20m；不接触任何source核心观测。
    exterior=(abs(local[:,:2])<=1.5).all(1)&(abs(local[:,:2])>half+.20).any(1)&(abs(local[:,2])<=.12)
    frames={};frame_audit=[];tables={}
    for raw in FRAMES:
        obs=read(bank/f'observations/{raw:03d}.json');qualified=load(S/f'qualified_observations/{raw:03d}.npz')
        ids=qualified['query_id'];keep=exterior[ids]
        keep &= qualified['road_clear']&qualified['shadow_proxy_clear']&~qualified['measured_conflict']&~qualified['box_blocked']
        ids=ids[keep];q=X[ids]@qualified['world_to_camera'][:3,:3].T+qualified['world_to_camera'][:3,3]
        uv=brown(q,obs['camera_intrinsic_Brown'])
        photo=np.asarray(Image.open(bank/obs['image_path']).convert('RGB'),np.float32)/255
        sem=np.asarray(Image.open(B/f'semantics/{raw:03d}.png').resize((480,320),Image.Resampling.NEAREST))
        road=binary_erosion(sem==0,iterations=1)
        fg=np.asarray(Image.open(B/f'repaired_masks_all_r4/{raw:03d}/dynamic_union.png'))>0
        labels=np.asarray(Image.open(B/f'repaired_masks_all_r4/{raw:03d}/labels.png'))>0
        ann=read(C/f'waymo_val065_road_completion_20261111_r1/raw_road_annotations/{raw:03d}.json')
        rectangles=np.asarray([b['xyxy'] for b in ann['front_camera_rectangles']]).reshape(-1,4)*.25
        foreground=guard((320,480),rectangles,fg|labels)
        low=np.asarray(Image.fromarray(np.rint(photo*255).astype(np.uint8)).resize((480,320),Image.Resampling.LANCZOS),np.float32)/255
        lum=linear(low)@LUMA;clean=road&~foreground;den=gaussian_filter(clean.astype(np.float32),12)
        local_luma=gaussian_filter(lum*clean,12)/np.maximum(den,1e-6)
        shadow=(lum<.72*local_luma)&(local_luma-lum>.02)&(den>.2)&road
        cert_uv=(uv+.5)/4-.5
        fresh=(q[:,2]>.1)&(uv[:,0]>=0)&(uv[:,0]<=1919)&(uv[:,1]>=0)&(uv[:,1]<=1279)
        fresh &= sample(road.astype(np.float32),cert_uv,0)>.5
        fresh &= sample(foreground.astype(np.float32),cert_uv,1,1)==0
        fresh &= sample(shadow.astype(np.float32),cert_uv,1,1)==0
        rgb=sample(photo,uv)
        # 曝光标定只用中性沥青；保留标线和阴影为未使用，而非做曝光解释。
        fresh &= (rgb.max(1)<.65)&(np.ptp(rgb,axis=1)<.18)&(rgb.min(1)>.04)
        ids,uv,rgb=ids[fresh],uv[fresh],rgb[fresh]
        assert not ((abs(local[ids,:2])<=half+.20).all(1)).any()
        frames[raw]=dict(ids=ids,rgb=rgb,uv=uv)
        cells=np.floor(local[ids,:2]/.08).astype(np.int64)
        frame_audit.append(dict(raw=raw,qualified_exterior_before_Brown_gate=int(keep.sum()),after_Brown_photo_gate=len(ids),
            external_8cm_world_cells=len(np.unique(cells,axis=0)),source_core_samples_used=0,
            local_external_bbox_xy=[local[ids,:2].min(0).tolist(),local[ids,:2].max(0).tolist()]))
        tables[f'raw_{raw:03d}_query_id']=ids;tables[f'raw_{raw:03d}_RGB']=rgb;tables[f'raw_{raw:03d}_Brown_uv']=uv
    anchor=frames[115];records=[];pair_audit=[]
    for raw in FRAMES:
        f=frames[raw];common,fi,ai=np.intersect1d(f['ids'],anchor['ids'],return_indices=True)
        keys=np.floor(local[common,:2]/.08).astype(np.int64);unique,inverse=np.unique(keys,axis=0,return_inverse=True)
        fcell=[];acell=[];count=[]
        for cell in range(len(unique)):
            take=inverse==cell;fcell.append(np.median(f['rgb'][fi[take]],axis=0));acell.append(np.median(anchor['rgb'][ai[take]],axis=0));count.append(int(take.sum()))
        fcell=np.asarray(fcell);acell=np.asarray(acell);x=fcell@LUMA;y=acell@LUMA
        validation=((unique[:,0]*73856093)^(unique[:,1]*19349663))%3==0;fit=~validation
        enough=int(fit.sum())>=16 and int(validation.sum())>=8
        a=float(np.median(y[fit]/x[fit])) if fit.any() else None
        rng=np.random.default_rng(20261122+raw)
        ratios=y[fit]/x[fit]
        bootstrap=np.median(rng.choice(ratios,size=(1000,len(ratios)),replace=True),axis=1) if len(ratios) else np.array([np.nan])
        ci=np.quantile(bootstrap,[.025,.975]).tolist()
        raw_luma=float(abs(x[validation]-y[validation]).mean()) if validation.any() else None
        new_luma=float(abs(a*x[validation]-y[validation]).mean()) if validation.any() and a else None
        raw_rgb=float(abs(fcell[validation]-acell[validation]).mean()) if validation.any() else None
        new_rgb=float(abs(a*fcell[validation]-acell[validation]).mean()) if validation.any() and a else None
        stable=enough and a is not None and .8<=a<=1.25 and ci[1]-ci[0]<=.03 and new_luma<=.006 and new_rgb<=.012
        if raw==115:a=1.;ci=[1.,1.]
        record=dict(raw=raw,gain=a if stable else None,bias=0.,calibrated=bool(stable),
            status=('anchor' if raw==115 else 'calibrated_gain_only') if stable else 'uncalibratable_external_support_or_residual',
            role='TRAIN_POOL' if raw in TRAIN else 'QUERY_external_exposure_only_no_core',
            anchor_raw=115,fit_world_cells=int(fit.sum()),validation_world_cells=int(validation.sum()),
            matching_world_queries=len(common),gain_bootstrap95=ci,
            validation_raw_luma_MAE=raw_luma,validation_canonical_luma_MAE=new_luma,
            validation_raw_RGB_MAE=raw_rgb,validation_canonical_RGB_MAE=new_rgb,
            source_core_RGB_samples_used=0)
        records.append(record)
        pair_audit.append(dict(raw=raw,shared_8cm_cells=len(unique),sample_count_per_cell=count,
            low_frequency_definition='Equal-weight 8cm world cells; per-cell median RGB over exactly the same world queries in frame and anchor',
            fit_cell_keys=unique[fit].tolist(),validation_cell_keys=unique[validation].tolist(),
            luma_span_p90_minus_p10=float(np.quantile(x,.9)-np.quantile(x,.1)),
            affine_design_condition_number=float(np.linalg.cond(np.c_[x,np.ones(len(x))])),
            bias_fitted=False,per_channel_gain_fitted=False))
        tables[f'raw_{raw:03d}_shared_query_id']=common;tables[f'raw_{raw:03d}_cell_keys']=unique
        tables[f'raw_{raw:03d}_cell_RGB']=fcell;tables[f'raw_{raw:03d}_anchor_cell_RGB']=acell;tables[f'raw_{raw:03d}_validation_cells']=validation
    union=np.unique(np.concatenate([v['ids'] for v in frames.values()]));tables['world_query_ids']=union;tables['world_xyz']=X[union]
    np.savez_compressed(out/'external_correspondence_samples.npz',**tables)
    result=dict(schema='source_shared_external_scalar_exposure.v1',complete=True,calibration_frozen=True,
        source='patch_02',source_packet=str(source_path),anchor_raw=115,color_space='encoded_sRGB',
        forward='canonical = gain[raw] * raw_RGB + 0',inverse='raw_RGB = canonical / gain[raw]',
        bias_fixed=0.,gain_shared_between_training_arms=True,train_pool8=TRAIN,train_pool2=[88,115],query_frames=QUERY,
        query_core_used_for_calibration=False,query_external_real_road_used_for_exposure_only=True,
        calibration_domain=dict(outer_half_extent_m=1.5,excluded_source_half_extent_m=(half+.20).tolist(),
            max_abs_original_chart_height_m=.12,world_cell_m=.08,source_geometry_modified=False),
        fit_validation_split='world-cell deterministic hash modulo3: remainder0 external validation, others fit',
        minimum_fit_cells=16,minimum_validation_cells=8,
        acceptance=dict(gain_bounds=[.8,1.25],max_gain_bootstrap95_width=.03,
            max_external_validation_luma_MAE=.006,max_external_validation_RGB_MAE=.012),
        records=records,all_requested_frames_calibrated=all(r['calibrated'] for r in records),
        exposure_consistent_subpool=[r['raw'] for r in records if r['calibrated'] and abs(r['gain']-1)<=.015 and r['validation_raw_luma_MAE']<=.006],
        GPU_calls=0,optimizer_updates=0,original_images_modified=False,source_asset_modified=False,
        seconds=time.monotonic()-started)
    diagnostic=dict(schema='source_shared_exposure_external_diagnostic.v1',frame_footprints=frame_audit,pair_cells=pair_audit,
        existing_query_geometry=str(S/'geometry.npz'),qualification_inputs=str(S/'qualified_observations/{raw:03d}.npz'),
        fresh_photo_gate='Brown-correct raw coordinate; original road semantics eroded1certpx, full dynamic+labels dilation2certpx, original rectangles margin2certpx, generic local shadow proxy; inherited fresh LiDAR/OBB qualification preserved',
        core_exclusion_exact=True,bias_fitted=False,camera_fitted=False,query_core_read=False)
    (out/'shared_exposure.json').write_text(json.dumps(result,ensure_ascii=True,indent=2,allow_nan=False)+'\n')
    (out/'external_support_diagnostic.json').write_text(json.dumps(diagnostic,ensure_ascii=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,ensure_ascii=True,indent=2,allow_nan=False))


if __name__=='__main__':main()

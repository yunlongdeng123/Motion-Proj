"""A1原残73：按继承fixed-Σ和实际native alpha cutoff审计inactive人口。

CPU近似投影先用保存的真实逐行alphaT核对；候选不是新材质/几何，
不执行激活，不把足迹预测当作native贡献或K资格。
"""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
from pathlib import Path
import argparse
import json
import numpy as np
from road_activation_builder import CAMPAIGN,CPU_CAMPAIGN,read_npz,write_json


def projection(xyz,normal,sigma,camera_K,world_to_camera,*,negative_fov=False,width=480,height=320):
    R=world_to_camera[:3,:3]
    X=xyz@R.T+world_to_camera[:3,3]
    h=X@camera_K.T
    xy=h[:,:2]/h[:,2:]-.5
    z=X[:,2]
    # 当前候选的两个切向sigma完全相等，故协方差与切向基的旋转无关。
    if not np.allclose(sigma[:,0],sigma[:,1],atol=1e-12,rtol=0):raise ValueError('本CPU实现要求轴对称切向sigma')
    n=(normal/np.linalg.norm(normal,axis=1)[:,None])@R.T
    # 继承engine effective_config.neg_fov=True时，实际settings的Cov2D焦距
    # 来自tan(-.5)，而mean projection仍来自full_proj_transform；不能用K焦距代替。
    tx=np.tan(-.5) if negative_fov else width/(2*camera_K[0,0])
    ty=np.tan(-.5) if negative_fov else height/(2*camera_K[1,1])
    fx,fy=width/(2*tx),height/(2*ty)
    clipped_x=np.minimum(1.3*tx,np.maximum(-1.3*tx,X[:,0]/z))*z
    clipped_y=np.minimum(1.3*ty,np.maximum(-1.3*ty,X[:,1]/z))*z
    j0=np.stack([fx/z,np.zeros(len(z)),-fx*clipped_x/z**2],1)
    j1=np.stack([np.zeros(len(z)),fy/z,-fy*clipped_y/z**2],1)
    st=sigma[:,0]**2;dn=sigma[:,2]**2-st
    a=st*np.sum(j0*j0,1)+dn*np.sum(j0*n,1)**2+.3
    c=st*np.sum(j1*j1,1)+dn*np.sum(j1*n,1)**2+.3
    b=st*np.sum(j0*j1,1)+dn*np.sum(j0*n,1)*np.sum(j1*n,1)
    det=a*c-b*b
    conic=np.stack([c/det,-b/det,a/det],1)
    mid=.5*(a+c);lam=mid+np.sqrt(np.maximum(.1,mid*mid-det))
    radius=np.ceil(3*np.sqrt(lam)).astype(np.int64)
    return xy,z,conic,radius


def run(native,visibility,endpoint,output,calibration):
    native,visibility,endpoint,output=map(Path,(native,visibility,endpoint,output))
    output.mkdir(parents=True,exist_ok=False)
    isolated=json.loads(Path(calibration).read_text())
    if isolated.get('GPU_executed') is not True or isolated['max_CPU_raw_alpha_error']>2e-4:
        raise ValueError('缺少通过的真实单行fragment校准；不产出beam候选')
    road=read_npz(native/'asset_A1_guarded/road.npz')
    p=read_npz(native/'asset_A1_guarded/provenance.npz')
    source_model=Path(json.loads((native/'asset_A1_guarded/persistent/scene.json').read_text())['model'])
    native_settings=json.loads((source_model/'effective_config.json').read_text())
    negative_fov=bool(native_settings['neg_fov'])
    n=len(road['active']);world=p['world_xyz']
    material=p['known']&(p['source']==1)&~p['unknown']
    physical=p['surface_supported']&p['legal_write']&~p['guarded_world_rows']&(road['minimum_model_depth']<=0)
    eligible=~road['active']&material&physical
    common=np.zeros(n,bool);rows=[];pairs=[]
    for raw in (86,110):
        saved=read_npz(native/f'{raw:03d}_native.npz')
        remaining=saved['fixed_low_S']&(saved['road_alpha']<=.9)
        aftermax=read_npz(endpoint/f'{raw:03d}_endpoint.npz')['combined_max_road_alpha']
        ys,xs=np.nonzero(remaining)
        geom=read_npz(CPU_CAMPAIGN/f'ideal_targets_r2/geometry/{raw:03d}.npz')
        sparse=read_npz(visibility/f'{raw:03d}_pixel_contribution.npz')
        xy,z,conic,radius=projection(world,road['normal_world'],road['gaussian_sigma_m'],geom['camera_K'],geom['world_to_camera'],
            negative_fov=negative_fov,width=remaining.shape[1],height=remaining.shape[0])
        real=sparse['contributing_world_rows']
        xyerror=float(np.abs(xy[real]-sparse['projected_xy']).max())
        radius_mismatch=int((radius[real]!=sparse['native_radii']).sum())
        errors=[];aggregate_errors=[]
        for pi,(y,x) in enumerate(zip(ys,xs)):
            hit=sparse['sparse_pixel_index']==pi
            wi=sparse['sparse_world_row'][hit];mass=sparse['sparse_alphaT'][hit].astype(np.float64)
            order=np.argsort(z[wi],kind='stable');wi,mass=wi[order],mass[order]
            T=1-np.r_[0.,np.cumsum(mass[:-1])]
            actual_alpha=mass/T
            d=xy[wi]-[x,y]
            power=-.5*(conic[wi,0]*d[:,0]**2+2*conic[wi,1]*d[:,0]*d[:,1]+conic[wi,2]*d[:,1]**2)
            predicted=np.minimum(.99,road['road_opacity'][wi]*np.exp(power))
            errors.extend(np.abs(actual_alpha-predicted).tolist())
            aggregate_errors.append(abs(1-np.prod(1-predicted)-float(mass.sum())))
        error=float(max(errors,default=0.))
        aggregate_error=float(max(aggregate_errors,default=0.))
        if xyerror>1e-4 or radius_mismatch or aggregate_error>2e-4:
            raise ValueError('CPU fixed-Σ足迹尚未与实际native匹配: '+str((raw,xyerror,radius_mismatch,aggregate_error)))
        selected=np.zeros(n,bool);pixel_counts=np.zeros(len(ys),np.int32)
        candidates=np.flatnonzero(eligible&(z>.2)&np.isfinite(xy).all(1))
        local_pairs=[]
        for start in range(0,len(candidates),10000):
            ids=candidates[start:start+10000]
            dx=xy[ids,0,None]-xs[None,:];dy=xy[ids,1,None]-ys[None,:]
            power=-.5*(conic[ids,0,None]*dx*dx+2*conic[ids,1,None]*dx*dy+conic[ids,2,None]*dy*dy)
            alpha=np.minimum(.99,road['road_opacity'][ids,None]*np.exp(np.minimum(power,0)))
            # 继承kernel tile-expanded radius矩形；仅作为额外裁剪，不用它替代alpha cutoff。
            lo=np.maximum(np.trunc((xy[ids]-radius[ids,None])/16).astype(np.int64),0)*16
            hi=np.maximum(np.trunc((xy[ids]+radius[ids,None]+15)/16).astype(np.int64),0)*16-1
            tile=(xs[None,:]>=lo[:,0,None])&(xs[None,:]<=hi[:,0,None])&(ys[None,:]>=lo[:,1,None])&(ys[None,:]<=hi[:,1,None])
            valid=(alpha>=1/255)&tile
            ir,ip=np.nonzero(valid)
            selected[ids[ir]]=True
            np.add.at(pixel_counts,ip,1)
            if len(ir):local_pairs.append((ids[ir],ip,alpha[ir,ip]))
        common|=selected
        ids=np.flatnonzero(selected)
        if local_pairs:
            wr=np.concatenate([v[0] for v in local_pairs]);pi=np.concatenate([v[1] for v in local_pairs]);aa=np.concatenate([v[2] for v in local_pairs])
        else:wr=np.empty(0,np.int64);pi=np.empty(0,np.int64);aa=np.empty(0)
        np.savez_compressed(output/f'{raw:03d}_fixed_sigma_footprint.npz',pixel_y=ys,pixel_x=xs,
            A1_residual_alpha=saved['road_alpha'][ys,xs],endpoint_alpha=aftermax[ys,xs],
            fixed_sigma_candidate_world_rows=ids,candidate_world_row_per_pair=wr,pixel_index_per_pair=pi,
            predicted_fragment_alpha_per_pair=aa.astype(np.float32),candidate_count_per_pixel=pixel_counts,
            proposed_source=p['source'][ids],known=p['known'][ids],surface_supported=p['surface_supported'][ids],
            legacy_legal_write=p['legal_write'][ids],world_xyz=world[ids],fixed_sigma_m=road['gaussian_sigma_m'][ids])
        rows.append(dict(raw=raw,A1_residual_pixels=len(ys),endpoint_still_thin_pixels=int((aftermax[ys,xs]<=.9).sum()),
            candidate_rows=len(ids),pixels_with_observed_inactive_footprint=int((pixel_counts>0).sum()),
            actual_native_projection_error_px=xyerror,actual_native_radii_mismatch_rows=radius_mismatch,
            naive_double_depth_raw_inference_max_error=error,
            order_independent_alpha_product_max_error_vs_actual_native=aggregate_error))
    ids=np.flatnonzero(common)
    np.savez_compressed(output/'observed_inactive_fixed_sigma_candidates.npz',proposed_world_rows=ids,
        original_active=road['active'],proposed_active=road['active']|common,world_xyz=world[ids],
        source=p['source'][ids],known=p['known'][ids],surface_supported=p['surface_supported'][ids],
        parent_effective_rgb_sparse=road['reference_srgb'][ids],observed_reference_rgb_sparse=p['observed_reference_srgb'][ids])
    result=dict(schema='road_activation_fixed_sigma_footprint_CPU.v1',CPU_only=True,additional_native_calls=0,optimizer_updates=0,
        source_asset=str(native/'asset_A1_guarded'),A1_residual_pixels=73,endpoint_strata=dict(improved=21,remaining=52),
        candidate_rows=len(ids),rows=rows,no_metric_radius_or_dilation_used=True,
        selection='existing inactive source1/known material-qualified supported rows of the same parent world chart whose original fixed-Σ native fragment prediction passes1/255 cutoff at an A1 residual pixel',
        inherited_native_negative_fov=negative_fov,
        native_covariance_settings_source=str(source_model/'effective_config.json'),
        isolated_fragment_calibration=str(calibration),isolated_fragment_CPU_alpha_max_error=isolated['max_CPU_raw_alpha_error'],
        no_new_geometry_or_material=True,activation_authorized=False,new_packet_K_qualified=False,
        boundary='CPU footprint prediction is numerically crosschecked against saved native VJP, not an actual inactive-row contribution or strict certificate; fresh native K/near and coverage are required')
    write_json(output/'results.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native',type=Path,default=CAMPAIGN/'native_r1');p.add_argument('--visibility',type=Path,default=CAMPAIGN/'visibility_r2')
    p.add_argument('--endpoint',type=Path,default=CAMPAIGN/'endpoint_r1');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--calibration',type=Path,default=CAMPAIGN/'footprint_calibration_native_r1/results.json')
    a=p.parse_args();print(json.dumps(run(a.native,a.visibility,a.endpoint,a.output,a.calibration),ensure_ascii=False),flush=True)

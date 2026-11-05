"""Observed-anchored C2 cubic B-spline camera-depth posterior, CPU only.

One data-derived physical control pitch; fixed 5 Huber IRLS solves. No PVD
training, parameter search, colour fitting or source-asset overwrite.
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.interpolate import BSpline
from scipy.ndimage import binary_erosion
from scipy.sparse import coo_matrix, eye, vstack
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

CAM = Path("/root/autodl-tmp/drivingeditor_repro/campaign")
OLD = CAM/"scene008_a6_appearance_formal_20261105"
ART = CAM/"scene008_a6_insert_artifacts_20261105"
HELD = np.array([89, 95, 101, 105, 113, 121, 125, 130, 137])
BUILD = np.setdiff1d(np.arange(89, 138), HELD)
FLOAT = ("xyz", "normal", "sigma", "opacity", "thickness", "box", "knn_spacing", "normal_radius", "normal_confidence")
BOOL = ("generated", "conditioning", "normal_valid", "out_of_box", "conditioning_held")
INT = ("source_index", "conditioning_observation_index", "conditioning_raw_frame", "conditioning_cloud_row")
GEOMETRY = FLOAT+BOOL+INT
APPEARANCE = {"canonical_sh", "rgb", "colour_inferred", "metadata_json"}
RESERVED = set(GEOMETRY)|APPEARANCE|{"coefficients", "sh", "sh_degree", "geometry_hash", "provenance_hash"}
C0 = np.float32(.28209479177387814)


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bank(p):
    with np.load(p, allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}
def write(p, v): Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
def same(a,b): return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def stats(a):
    a=np.asarray(a,float); a=a[np.isfinite(a)]
    if not len(a): return dict(count=0)
    q=np.quantile(a,[0,.1,.5,.9,.99,1]);return dict(count=len(a),min=q[0],p10=q[1],median=q[2],p90=q[3],p99=q[4],max=q[5],mean=float(a.mean()))
def geometry_hash(a):
    h=hashlib.sha256()
    for k in sorted(GEOMETRY):
        v=np.ascontiguousarray(a[k]);h.update(k.encode());h.update(str((v.dtype.str,v.shape)).encode());h.update(v.tobytes())
    return h.hexdigest()
def provenance_hash(a):
    h=hashlib.sha256()
    for k in sorted(set(a)-RESERVED):
        v=np.ascontiguousarray(a[k]);h.update(k.encode());h.update(str((v.shape,v.dtype.str)).encode());h.update(v.tobytes())
    return h.hexdigest()


def nearest3_pitch(x):
    d=cKDTree(x).query(x,k=4)[0][:,1:4]
    return float(np.median(d[d>1e-8]))


def knots_for(lo,hi,step):
    left=np.floor(lo/step)*step-step
    intervals=int(np.ceil((hi-left)/step))+1
    return left+np.arange(-3,intervals+4)*step


def tensor_basis(uv,tu,tv):
    u=BSpline.design_matrix(uv[:,0],tu,3,extrapolate=False).tocsr()
    v=BSpline.design_matrix(uv[:,1],tv,3,extrapolate=False).tocsr()
    assert np.all(np.diff(u.indptr)==4) and np.all(np.diff(v.indptr)==4)
    ui,vi=u.indices.reshape(-1,4),v.indices.reshape(-1,4)
    ud,vd=u.data.reshape(-1,4),v.data.reshape(-1,4)
    nv=len(tv)-4
    col=(ui[:,:,None]*nv+vi[:,None,:]).reshape(-1)
    values=(ud[:,:,None]*vd[:,None,:]).reshape(-1)
    out=coo_matrix((values,(np.repeat(np.arange(len(uv)),16),col)),shape=(len(uv),(len(tu)-4)*nv)).tocsr()
    out.eliminate_zeros()
    if len(uv): assert np.max(abs(np.asarray(out.sum(1)).ravel()-1))<2e-12
    return out


def curvature(nu,nv):
    rows=[];cols=[];vals=[];n=0
    for i in range(1,nu-1):
        for j in range(nv):
            rows.extend([n]*3);cols.extend([(i-1)*nv+j,i*nv+j,(i+1)*nv+j]);vals.extend([1.,-2.,1.]);n+=1
    for i in range(nu):
        for j in range(1,nv-1):
            rows.extend([n]*3);cols.extend([i*nv+j-1,i*nv+j,i*nv+j+1]);vals.extend([1.,-2.,1.]);n+=1
    return coo_matrix((vals,(rows,cols)),shape=(n,nu*nv)).tocsr()


def huber_objective(B,y,x,sigma):
    if not len(y):return 0.
    e=np.abs((B@x-y)/sigma)
    return float(np.mean(np.where(e<=1,.5*e*e,e-.5)))


def neighbor_stats(depth_map,mask):
    interior=binary_erosion(mask,iterations=1)
    result={}
    for name,m in (("same_query_support",mask),("same_support_eroded1",interior)):
        dx=(m[:,1:]&m[:,:-1]);dy=(m[1:]&m[:-1])
        a=np.r_[np.abs(np.diff(depth_map,axis=1))[dx],np.abs(np.diff(depth_map,axis=0))[dy]]
        result[name]=dict(abs_adjacent_z_difference_m=stats(a),pairs_gt_01m=int((a>.1).sum()))
    return result


def export_asset(path,base,original_rows,new_local,new_z,fx,chart_row_mask,metadata,matte_rgb=None):
    n=len(base["xyz"])
    out={k:(v[original_rows].copy() if v.ndim and v.shape[0]==n else v.copy()) for k,v in base.items()}
    if new_local is not None:
        out["xyz"][chart_row_mask]=new_local.astype(np.float32)
        s=(.6*new_z/fx).astype(np.float32)
        out["sigma"][chart_row_mask]=s;out["thickness"][chart_row_mask]=s
        out["normal_radius"][chart_row_mask]=3*s
        out["knn_spacing"][chart_row_mask]=(new_z/fx).astype(np.float32)
        out["out_of_box"][chart_row_mask]=(abs(new_local)>out["box"]/2).any(1)
    if matte_rgb is not None:
        real=out["resample_rgb_real"][chart_row_mask]
        chart_ids=np.flatnonzero(chart_row_mask)[~real]
        rgb=matte_rgb[~real].astype(np.float32)
        out["canonical_sh"][chart_ids,0]=(rgb-np.float32(.5))/C0
        out["rgb"][chart_ids]=out["canonical_sh"][chart_ids,0]*C0+np.float32(.5)
    if matte_rgb is None:
        for key in ("canonical_sh","rgb","colour_inferred"):
            assert same(out[key],base[key][original_rows]),"Depth-only appearance changed: "+key
    known=out["resample_rgb_real"]
    for key in ("canonical_sh","rgb","colour_inferred"):
        assert same(out[key][known],base[key][original_rows][known]),"Real106 appearance changed: "+key
    for key in GEOMETRY:
        if key=="box":continue
        assert same(out[key][~chart_row_mask],base[key][original_rows][~chart_row_mask]),"Retained PVD geometry changed: "+key
    for key in ("opacity","resample_source_row","resample_source_parent","representation_kind"):
        assert same(out[key],base[key][original_rows]),"Source/opacity changed: "+key
    out["geometry_hash"]=np.asarray(geometry_hash(out));out["provenance_hash"]=np.asarray(provenance_hash(out))
    old_meta=json.loads(str(base["metadata_json"]));old_meta["surface_depth_posterior"]=metadata
    old_meta["geometry_hash"]=str(out["geometry_hash"]);old_meta["provenance_hash"]=str(out["provenance_hash"])
    old_meta["representation"]="Observed-anchored C2 reference-depth posterior"
    out["metadata_json"]=np.asarray(json.dumps(old_meta,ensure_ascii=False,sort_keys=True,allow_nan=False))
    np.savez_compressed(path,**out)
    reloaded=bank(path);assert all(same(v,reloaded[k]) for k,v in out.items())
    return dict(path=str(path),sha256=sha(path),geometry_hash=str(out["geometry_hash"]),provenance_hash=str(out["provenance_hash"]),rows=len(out["xyz"]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    p.add_argument("--matte-samples",default=str(ART/"vehicle/matte_contract/matte_texture_samples_filled_v2.npz"))
    p.add_argument("--prototype",default=str(OLD/"output_adapter/projective_surface/projective_surface_asset.npz"))
    p.add_argument("--chart",default=str(OLD/"output_adapter/projective_surface/reference_chart.npz"))
    p.add_argument("--observations",default=str(CAM/"scene008_instances_20261105_r1/observations/a6.npz"))
    p.add_argument("--sigma-lidar",type=float,default=.05)
    args=p.parse_args();start=time.time();dest=Path(args.output);dest.mkdir(parents=True,exist_ok=False)
    paths=[Path(args.prototype),Path(args.chart),Path(args.observations),Path(args.matte_samples),OLD/"output_adapter/sphere_strong_v2/complete_a6.npz",CAM/"scene008_instances_20261105_r1/observations/frames/106.npz"]
    source_hashes={str(f):sha(f) for f in paths}
    base,chart,obs,matte,sphere,frame=map(bank,paths)
    assert geometry_hash(base)==str(base["geometry_hash"])
    assert provenance_hash(base)==str(base["provenance_hash"])
    xy=chart["pixel_xy"].astype(float);pixels=chart["pixels"]
    assert np.array_equal(matte["chart_frame_pixels"],pixels)
    assert np.array_equal(matte["chart_pixel_xy"],xy)
    assert np.array_equal(matte["real106_alwayskeep"],chart["color_real"])
    allowed=matte["effective_allowed_real_priority"].astype(bool)
    assert np.array_equal(allowed,matte["matte_allowed_nearest"]|chart["color_real"])
    assert allowed[chart["color_real"]].all()
    cut=len(chart["source_retained"]);original_rows=np.r_[np.arange(cut),cut+np.flatnonzero(allowed)]
    assert np.array_equal(matte["original_asset_row"],np.arange(cut,len(base["xyz"])))
    prior_z=chart["expected_depth_m"].astype(float)
    K=frame["K"].astype(float).copy();K[:2]*=4
    pose=chart["reference_pose"].astype(float);view=chart["reference_world_view"].astype(float)
    scale=float(np.linalg.norm(pose[:3,:3],axis=0).mean())
    camera=view.T@pose;camera[:3,:4]/=scale
    xyz=obs["sample_xyz_local_m"].astype(float);raw=obs["sample_raw_frame"]
    legal=obs["sample_rgb_valid"].astype(bool)&np.isin(raw,BUILD)
    legal_ids=np.flatnonzero(legal);camera_xyz=xyz[legal]@camera[:3,:3].T+camera[:3,3]
    uv=(camera_xyz@K.T)[:,:2]/camera_xyz[:,2,None]-.5
    ij=np.floor(uv+.5).astype(int);z=camera_xyz[:,2]
    fullmap=np.full((1280,1920),-1,np.int32);fullmap.ravel()[pixels]=np.arange(len(pixels))
    inside=(z>.1)&(ij[:,0]>=0)&(ij[:,0]<1920)&(ij[:,1]>=0)&(ij[:,1]<1280)
    support=np.zeros(len(z),bool);support[inside]=fullmap[ij[inside,1],ij[inside,0]]>=0
    eligible=np.flatnonzero(support)
    pixel_keys=ij[eligible,1]*1920+ij[eligible,0]
    order=np.lexsort((legal_ids[eligible],z[eligible],pixel_keys));ordered=eligible[order]
    ordered_pixels=ij[ordered,1]*1920+ij[ordered,0]
    first=np.r_[True,ordered_pixels[1:]!=ordered_pixels[:-1]];front=ordered[first]
    prior_index=fullmap[ij[front,1],ij[front,0]]
    offset=z[front]-prior_z[prior_index]
    certain=np.abs(offset)<=.75;accepted=front[certain];uncertain=front[~certain]
    assert len(accepted)>0 and not np.isin(raw[legal_ids[accepted]],HELD).any()
    mothers=sphere["xyz"][sphere["sampling_exact_source_point"]]
    pvd_pitch=nearest3_pitch(mothers)
    perframe=[]
    for r in BUILD:
        x=xyz[legal&(raw==r)]
        if len(x)>=4:perframe.append(nearest3_pitch(x))
    lidar_pitch=float(np.median(perframe));pitch=max(pvd_pitch,lidar_pitch)
    zref=float(np.median(prior_z));step_u=float(K[0,0]*pitch/zref);step_v=float(K[1,1]*pitch/zref)
    tu=knots_for(xy[:,0].min(),xy[:,0].max(),step_u);tv=knots_for(xy[:,1].min(),xy[:,1].max(),step_v)
    Bp=tensor_basis(xy[allowed],tu,tv);Bm=tensor_basis(uv[accepted],tu,tv);Bq=tensor_basis(xy[allowed],tu,tv)
    D=curvature(len(tu)-4,len(tv)-4)
    term=[(Bp,prior_z[allowed]-zref,.20,"prior"),(Bm,z[accepted]-zref,args.sigma_lidar,"BUILD_visible_LiDAR"),(D,np.zeros(D.shape[0]),.05,"D2_control_curvature")]
    n=Bp.shape[1];coef=np.zeros(n);ridge=1e-8;trace=[]
    for iteration in range(5):
        H=eye(n,format="csc")*(ridge/n);rhs=np.zeros(n)
        for B,y,sigma,name in term:
            residual=B@coef-y;robust=np.minimum(1.,sigma/np.maximum(abs(residual),1e-15))
            weight=robust/(len(y)*sigma*sigma)
            H=H+B.T@B.multiply(weight[:,None]);rhs+=B.T@(weight*y)
        updated=spsolve(H.tocsc(),rhs)
        assert np.isfinite(updated).all();coef=updated
        trace.append(dict(iteration=iteration+1,objective={name:huber_objective(B,y,coef,sigma) for B,y,sigma,name in term},coefficient_maxabs=float(abs(coef).max())))
    new_z=zref+Bq@coef;measured_pred=zref+Bm@coef
    assert np.isfinite(new_z).all() and (new_z>.1).all()
    rays=np.column_stack((xy[allowed]+.5,np.ones(int(allowed.sum()))))@np.linalg.inv(K).T
    local=(rays*new_z[:,None]-camera[:3,3])@np.linalg.inv(camera[:3,:3]).T
    reconstructed=local@camera[:3,:3].T+camera[:3,3]
    projection=(reconstructed@K.T)[:,:2]/reconstructed[:,2,None]-.5
    reproj=float(abs(projection-xy[allowed]).max());assert reproj<1e-8
    domain=np.zeros((1280,1920),bool);domain.ravel()[pixels[allowed]]=True
    oldmap=np.full(domain.shape,np.nan);newmap=oldmap.copy();oldmap.ravel()[pixels[allowed]]=prior_z[allowed];newmap.ravel()[pixels[allowed]]=new_z
    metadata=dict(method="Observed-anchored tensor cubic B-spline C2 depth",reference_raw=106,
        original_prototype_sha256=source_hashes[str(paths[0])],original_chart_sha256=source_hashes[str(paths[1])],
        matte_sha256=source_hashes[str(paths[3])],physical_control_pitch_m=pitch,control_spacing_HR_pixels=[step_u,step_v],
        sigma_prior_m=.20,sigma_LiDAR_m=args.sigma_lidar,sigma_D2_curvature_m=.05,
        term_normalization="Each prior/LiDAR/curvature term divided by its own row count",Huber_delta_sigma=1.,IRLS_solves=5,ridge=ridge,
        curvature="Second differences of depth control coefficients per u/v knot cell; residual units metres",
        geometry_changed=True,only_reference_ray_depths_fitted=True,sigma_reference_HR_pixels=.6,
        no_free_pose_or_xy_or_scale_adjustment=True,old_formal_compensation_colours_used=False,
        matte_prior_and_query_filter=True,real106_known_always_retained=True,PVD_updates=0,colour_fit_updates=0,
        source_distance_arrays="Original resampling ancestry descriptors retained, not reassigned to posterior positions")
    chart_mask=np.arange(len(original_rows))>=cut
    exported={}
    exported["masked_prototype"]=export_asset(dest/"masked_prototype.npz",base,original_rows,None,None,K[0,0],chart_mask,dict(metadata,geometry_posterior_applied=False))
    exported["smooth_depth_only"]=export_asset(dest/"smooth_depth_only.npz",base,original_rows,local,new_z,K[0,0],chart_mask,dict(metadata,initial_RGB="Prototype coefficients bitwise unchanged"))
    effective_RGB=matte["effective_RGB_real_priority"][allowed]
    exported["masked_matte_initialization"]=export_asset(dest/"masked_matte_initialization.npz",base,original_rows,None,None,K[0,0],chart_mask,dict(metadata,geometry_posterior_applied=False,initial_RGB="Approved matte-unpremultiplied fixed teacher; true106 protected"),effective_RGB)
    exported["smooth_depth_matte_initialization"]=export_asset(dest/"smooth_depth_matte_initialization.npz",base,original_rows,local,new_z,K[0,0],chart_mask,dict(metadata,initial_RGB="Approved matte-unpremultiplied fixed teacher; true106 protected"),effective_RGB)
    np.savez_compressed(dest/"posterior.npz",control_delta_z_m=coef,z_reference_m=zref,u_knots_pixels=tu,v_knots_pixels=tv,
        query_original_asset_rows=original_rows,chart_query_keep=allowed,query_depth_m=new_z,query_local_xyz=local,
        effective_RGB_real_priority=effective_RGB,prior_depth_m=prior_z,prior_depth_map_m=oldmap,posterior_depth_map_m=newmap,
        reference_K=K,reference_camera_from_local_m=camera,physical_pitch_m=pitch,
        accepted_sample_ids=legal_ids[accepted],accepted_sample_raw=raw[legal_ids[accepted]],accepted_sample_uv=uv[accepted],
        accepted_measured_z_m=z[accepted],accepted_prior_z_m=prior_z[fullmap[ij[accepted,1],ij[accepted,0]]],accepted_posterior_z_m=measured_pred,
        uncertain_sample_ids=legal_ids[uncertain],uncertain_depth_offset_m=offset[~certain],
        all_frontmost_sample_ids=legal_ids[front],all_frontmost_sample_uv=uv[front],all_frontmost_sample_z_m=z[front])
    before=prior_z[prior_index[certain]]-z[accepted];after=measured_pred-z[accepted]
    receipt=dict(status="PASS_CPU_C2_DEPTH_POSTERIOR",contract=metadata,source_SHA256=source_hashes,
        source_PVD_mother_rows=len(mothers),PVD_median_nearest3_m=pvd_pitch,BUILD_per_frame_median_nearest3_m=lidar_pitch,
        reference_depth_for_spacing_m=zref,control_shape=[len(tu)-4,len(tv)-4],control_coefficients=n,
        prior_fit_samples=int(allowed.sum()),curvature_rows=D.shape[0],all_measured_samples=len(xyz),BUILD_visible_samples=int(legal.sum()),
        reference_supported_samples=int(support.sum()),frontmost_reference_pixels=len(front),association_certain_samples=len(accepted),
        association_uncertain_over075m=len(uncertain),association_uncertain_not_forced_into_fit=True,
        accepted_samples_outside_matte_query_support=int((~allowed[fullmap[ij[accepted,1],ij[accepted,0]]]).sum()),
        initial_chart_rows=len(xy),matte_retained_chart_rows=int(allowed.sum()),real106_rows_protected=int(chart["color_real"].sum()),
        measured_training_camera_z_abs_error_m=dict(prior=stats(abs(before)),posterior=stats(abs(after))),
        measured_training_camera_z_signed_error_m=dict(prior=stats(before),posterior=stats(after)),
        posterior_minus_prior_z_m=stats(new_z-prior_z[allowed]),
        neighbors=dict(prior_matched_support=neighbor_stats(oldmap,domain),posterior_matched_support=neighbor_stats(newmap,domain)),
        reference_reprojection_error_HR_pixels_max=reproj,IRLS_trace=trace,assets=exported,
        camera_z_measurement_error_is_training_fit_not_HELD_generalization=True,no_HELD_data_used=True,
        PVD_training_updates=0,colour_optimization_updates=0,old_sources_unchanged=all(sha(f)==h for f,h in zip(paths,source_hashes.values())),
        seconds=time.time()-start,source_sha256=sha(__file__))
    assert receipt["old_sources_unchanged"]
    write(dest/"receipt.json",receipt)
    print(json.dumps({k:receipt[k] for k in ["status","control_shape","prior_fit_samples","association_certain_samples","association_uncertain_over075m","measured_training_camera_z_abs_error_m","neighbors","seconds"]}),flush=True)


if __name__=="__main__":main()

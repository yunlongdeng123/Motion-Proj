"""CPU-only, fixed-native-rectangle C2 chart / measured ground ordering probe.

Reuses immutable prior CPU helpers. No GT completion or renderer/model update.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial import Delaunay
from scipy.ndimage import distance_transform_edt
from PIL import Image, ImageDraw
from geometry_cpu import (LodGeometry, Surface, rays, load, trans, norm,
                          projected_rows, alphas, weights, stat)


ROI = [232, 190, 284, 218]  # half-open xyxy, 480x320 native pixels


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def opaque_hits(vertices, triangles, origin, directions, alpha, matte):
    v0, v1, v2 = (vertices[triangles[:, i]] for i in range(3))
    e1, e2 = v1-v0, v2-v0
    tv = origin-v0
    qv = np.cross(tv, e1)
    num = np.einsum("ij,ij->i", e2, qv)
    out = np.full((len(matte), len(directions)), np.nan)
    for j, direction in enumerate(directions):
        pv = np.cross(direction[None], e2)
        det = np.einsum("ij,ij->i", e1, pv)
        safe = np.where(np.abs(det)>1e-12, det, 1.)
        u = np.einsum("ij,ij->i", tv, pv)/safe
        v = (qv@direction)/safe
        d = num/safe
        base = (np.abs(det)>1e-12)&(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)&(d>.1)
        for mi, threshold in enumerate(matte):
            valid = base & (alpha[triangles].min(1)>=threshold)
            if valid.any():
                out[mi,j] = d[valid].min()
    return out


def summary(mask, labels, loss, bw, nw, ext, gap, gt, newrgb, basergb):
    n = int(mask.sum())
    return dict(pixels=n,chart_first=int((mask&(labels==1)).sum()),
        ground_first=int((mask&(labels==2)).sum()),near_contact_005m=int((mask&(labels==3)).sum()),
        no_chart_hit_or_ground_unsupported=int((mask&(labels==0)).sum()),
        baseline_assetW=stat(bw[mask]),current_assetW=stat(nw[mask]),assetW_loss=stat(loss[mask]),
        assetW_loss_gt001=int((mask&(loss>.01)).sum()),assetW_loss_gt01=int((mask&(loss>.1)).sum()),
        assetW_loss_gt05=int((mask&(loss>.5)).sum()),newroad_Talpha=stat(ext[mask]),
        ground_minus_chart_depth_m=stat(gap[mask]),
        observed_RGB_MAE_baseline=stat(np.abs(basergb-gt).mean(0)[mask]),
        observed_RGB_MAE_current=stat(np.abs(newrgb-gt).mean(0)[mask]))


def make_panels(folder, obs, base, current, labels, loss, roi):
    palette = np.array([[110,110,110],[20,180,95],[225,100,40],[240,200,40]], np.uint8)
    x0,y0,x1,y1=roi
    layers = [np.rint(obs["rgb"]*255).clip(0,255).astype(np.uint8),
        np.rint(base["rgb"].transpose(1,2,0)*255).clip(0,255).astype(np.uint8),
        np.rint(current["rgb"].transpose(1,2,0)*255).clip(0,255).astype(np.uint8)]
    marked=[]
    for im in layers:
        p=Image.fromarray(im); ImageDraw.Draw(p).rectangle((x0,y0,x1-1,y1-1),outline=(255,50,50),width=1);marked.append(p)
    canvas=Image.new("RGB",(480*3,350),"white"); d=ImageDraw.Draw(canvas)
    for i,(im,title) in enumerate(zip(marked,("Original real FRONT RGB", "Frozen V44 delete", "R0 cellfit1000 delete"))):
        canvas.paste(im,(i*480,30));d.text((i*480+8,8),title,fill="black")
    canvas.save(folder/"raw128_native_registered_red.png")
    crops=[Image.fromarray(a[y0:y1,x0:x1]).resize(((x1-x0)*10,(y1-y0)*10),Image.Resampling.NEAREST) for a in layers]
    lab=Image.fromarray(palette[labels[y0:y1,x0:x1]]).resize(crops[0].size,Image.Resampling.NEAREST)
    heat=np.zeros((320,480,3),np.uint8);heat[:,:,0]=np.rint(np.clip(loss,0,1)*255)
    h=Image.fromarray(heat[y0:y1,x0:x1]).resize(crops[0].size,Image.Resampling.NEAREST)
    crops.extend([lab,h]);cw,ch=crops[0].size
    canvas=Image.new("RGB",(cw*3,(ch+45)*2),"white");d=ImageDraw.Draw(canvas)
    titles=["Original RGB (a37 can occlude hidden C2)","Frozen V44 delete","Frozen R0 cellfit1000 delete",
        "Chart first=green; ground first=orange", "V44 assetW minus R0 assetW (red)"]
    for i,im in enumerate(crops):
        xx=(i%3)*cw;yy=(i//3)*(ch+45);canvas.paste(im,(xx,yy+40));d.text((xx+5,yy+6),titles[i],fill="black")
    d.text((2*cw+8,ch+54),"Yellow: |ground-chart| <= 0.05m\nGray: chart miss / unsupported ground\nOpaque open chart reference, not hidden GT",fill="black")
    canvas.save(folder/"raw128_contact_ordering_crops.png")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run",type=Path,required=True);args=parser.parse_args()
    run=args.run;out=run/"red";out.mkdir(exist_ok=True);start=time.time()
    target=out/"raw128_contact_audit_v2.json"
    if target.exists():raise FileExistsError(target)
    old=run.parent/"scene008_road_r0_v44_20261107_r1";diag=run/"diagnostic/layers_v1/128"
    obs=load(diag/"observations.npz");base=load(diag/"v44_delete.npz");current=load(diag/"r0_delete.npz")
    solo=load(diag/"asset_solo.npz");roadsolo=load(diag/"newroad_solo.npz")
    asset_path=run.parent/"scene008_a6_insert_artifacts_20261105/surface_fit/combined_v1/observed_fit/formal/complete_a6.npz"
    posterior_path=run.parent/"scene008_a6_insert_artifacts_20261105/surface_fit/masked_depth_v1/posterior.npz"
    asset=load(asset_path);chart=asset["representation_kind"]==1
    uv=asset["resample_reference_pixel_xy"][chart];vertices=asset["xyz"][chart].astype(float)
    alpha=np.minimum(asset["opacity"][chart]/np.float32(.99),1)
    triangles=Delaunay(uv).simplices
    edge=np.stack([np.linalg.norm(uv[triangles[:,i]]-uv[triangles[:,j]],axis=1) for i,j in ((0,1),(1,2),(2,0))],1)
    triangles=triangles[edge.max(1)<=1.5]
    bundle=json.loads((run.parent/"r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json").read_text())
    scale=float(bundle["meter_to_model"]);T=np.asarray(bundle["world_to_model"])@np.asarray(bundle["coordinate_anchor"]);inverse=np.linalg.inv(T)
    plane=np.asarray(json.loads(Path(bundle["original_readiness"]).read_text())["SGCM_geometry"]["ground_fit"]["plane_anchor"])
    surface=Surface(load(old/"surface/measured_ground_support_pool.npz")["local_xyz"],plane)
    frame=load(run.parent/"r15_gn11_fullclip_cpu_20261026/frames/128.npz")
    ground,ground_valid=rays(surface,frame["K"],frame["w2l"])
    ground_depth=trans(ground.reshape(-1,3),frame["w2l"])[:,2].reshape(320,480)
    roi=np.zeros((320,480),bool);x0,y0,x1,y1=ROI;roi[y0:y1,x0:x1]=True;pixels=np.argwhere(roi)[:,::-1]
    camera_to_actor=np.linalg.inv(obs["model_from_local"][0])@np.linalg.inv(obs["model_to_camera"])
    direction=(np.c_[pixels+.5,np.ones(len(pixels))]@np.linalg.inv(obs["K"]).T)@camera_to_actor[:3,:3].T*scale
    hits=opaque_hits(vertices,triangles,camera_to_actor[:3,3],direction,alpha,(.01,.5))
    print(json.dumps(dict(stage="triangle_hits",seconds=time.time()-start)),flush=True)
    gaps=np.full((2,320,480),np.nan);labels=np.zeros((2,320,480),np.uint8)
    for i in range(2):
        gaps[i,roi]=ground_depth[roi]-hits[i];valid=ground_valid&np.isfinite(gaps[i]);labels[i,valid&(gaps[i]>.05)]=1
        labels[i,valid&(gaps[i]<-.05)]=2;labels[i,valid&(np.abs(gaps[i])<=.05)]=3
    actor_hits=camera_to_actor[:3,3]+direction*hits[0,:,None]
    chart_anchor=trans(trans(actor_hits,obs["model_from_local"][0]),inverse)
    finite=np.isfinite(chart_anchor).all(1);chart_height=np.full((320,480),np.nan)
    ch=np.full(len(pixels),np.nan)
    if finite.any():
        sh,ss=surface.height(chart_anchor[finite,:2]);ch[finite]=np.where(ss>=3,chart_anchor[finite,2]-sh,np.nan)
    chart_height[roi]=ch
    bw=base["group_alpha_T"][0];nw=current["group_alpha_T"][0];loss=bw-nw;ext=current["newroad_alpha_T"]
    masks={"registered_rectangle":roi,"a6_candidate_hidden":roi&obs["box_candidate"][0]&~obs["visible"][0],
        "baseline_assetW_gt05_loss_gt001":roi&(bw>.5)&(loss>.01),
        "baseline_assetW_gt09_loss_gt001":roi&(bw>.9)&(loss>.01),
        "baseline_assetW_gt09_loss_gt05":roi&(bw>.9)&(loss>.5),
        "observed_a6":roi&obs["visible"][0],"observed_a37":roi&obs["visible"][1],
        "original_a6_unknown":roi&obs["unknown"][0],"original_road":roi&obs["road"],
        "asset_solo_alpha_gt05":roi&(solo["alpha"]>.5)}
    strata={str(matte):{k:summary(m,labels[i],loss,bw,nw,ext,gaps[i],obs["rgb"].transpose(2,0,1),current["rgb"],base["rgb"]) for k,m in masks.items()} for i,matte in enumerate((.01,.5))}
    v44=json.loads((run.parent/"scene008_v44_rgb_asset_20261107_r1/results/full49_r1/v44/128/metrics.json").read_text())
    camera=v44["actual_camera_pose"];saved=v44["states"]["delete"]["meta"]["LOD"];lod=LodGeometry(asset,load(posterior_path))
    local,n,s,thin,op,kind=lod.samples(float(saved["du"]));assert len(local)==saved["render_rows"]
    pose=obs["model_from_local"][0].astype(np.float32);model=trans(local,pose).astype(np.float32);anchor=trans(model,inverse)
    normal_model=norm(n@pose[:3,:3].T);normal_anchor=norm(normal_model@inverse[:3,:3].T)
    heights,support=surface.height(anchor[:,:2]);heights=anchor[:,2]-heights
    actor_scale=float(np.linalg.norm(pose[:3,:3],axis=0).mean());sigma=s.astype(float)*actor_scale/scale;thickness=thin.astype(float)*actor_scale/scale
    sigma_z=np.sqrt(sigma*sigma+(thickness*thickness-sigma*sigma)*normal_anchor[:,2]**2)
    cfg=json.loads((run.parent/"scene008_blue_global_rgb_probe_20261105_r1/global_10k/effective_config.json").read_text())
    body_rows=projected_rows(model,normal_model,sigma,thickness,op,camera,scale,cfg["neg_fov"])
    baked_path=old/"candidates/baked_cellfit_strip1000_candidate_v1/new_road_canonical_candidate.npz";road=load(baked_path)
    nr=norm(road["normal_world"]@np.asarray(bundle["world_to_model"])[:3,:3].T/scale)
    road_rows=projected_rows(road["model_xyz"].astype(np.float32),nr,road["gaussian_sigma_m"][:,0],road["gaussian_sigma_m"][:,2],road["opacity"],camera,scale,cfg["neg_fov"])
    cov0=road_rows["cov2D"]-.3*np.eye(2);eig=np.linalg.eigvalsh(cov0)
    def factor(c):return np.sqrt(np.maximum(np.linalg.det(c),0)/np.maximum(np.linalg.det(c+.3*np.eye(2)),1e-30))
    fp=factor(cov0)
    # Registration is fixed first; evidence cohorts do not change the asset.
    chosen=masks["baseline_assetW_gt05_loss_gt001"]
    records=[]
    for y,x in np.argwhere(chosen):
        aa=alphas(body_rows,[x,y]);ra=alphas(road_rows,[x,y]);jw=weights(np.r_[aa,ra],np.r_[body_rows["depth"],road_rows["depth"]])
        bodyw=jw[:len(aa)];roadw=jw[len(aa):]
        isolated_bodyw=weights(aa,body_rows["depth"])
        bi=int(np.argmax(isolated_bodyw));ri=int(np.argmax(roadw));cov=road_rows["cov2D"][ri]
        delta=road_rows["uv"][ri]-[x,y];a0=min(.99,float(road["opacity"][ri]*np.exp(-.5*delta@np.linalg.inv(cov0[ri])@delta)))
        dist=float(distance_transform_edt(~obs["visible_reviewed"][0])[y,x])
        records.append(dict(pixel_xy=[int(x),int(y)],baseline_assetW=float(bw[y,x]),current_assetW=float(nw[y,x]),assetW_loss=float(loss[y,x]),
            ordering_matte_001=int(labels[0,y,x]),ground_minus_chart_depth_m=float(gaps[0,y,x]) if np.isfinite(gaps[0,y,x]) else None,
            chart_hit_height_above_pool_m=float(chart_height[y,x]) if np.isfinite(chart_height[y,x]) else None,
            original_visible_actor=int(obs["actor_labels"][y,x]),a6_unknown=bool(obs["unknown"][0,y,x]),a6_occluded=bool(obs["occluded"][0,y,x]),
            a6_reviewed_edge_distance_px=dist,original_road=bool(obs["road"][y,x]),
            dominant_actual_C2_LOD_row=bi,dominant_C2_center_height_m=float(heights[bi]),C2_center_height_support=int(support[bi]),
            dominant_C2_row_selection="argmax asset-solo CPU Talpha, including when native combined asset is fully occluded",
            dominant_C2_3sigma_lower_height_m=float(heights[bi]-3*sigma_z[bi]),
            road_parent_row=ri,road_center_to_pixel_px=float(np.linalg.norm(delta)),road_covariance_eigenvalues_before_filter_px2=eig[ri].tolist(),
            road_alpha_at_pixel=float(ra[ri]),road_alpha_without_03_filter=float(a0),
            road_center_depth_m=float(road_rows["depth"][ri]),dominant_C2_center_depth_m=float(body_rows["depth"][bi]),
            CPU_body_plus_road_assetW=float(bodyw.sum()),CPU_error_vs_native_current_assetW=float(bodyw.sum()-nw[y,x]),
            predicted_parent_relative_filter_factor_n2=float(factor(cov0[ri:ri+1]/4)[0]/fp[ri]),
            predicted_parent_relative_filter_factor_n4=float(factor(cov0[ri:ri+1]/16)[0]/fp[ri])))
    # Check road normal transformation and local surface compatibility globally.
    rh,rs=surface.height(road["local_xyz"][:,:2]);normal_len=np.linalg.norm(road["normal_local"],axis=1)
    delta=.01;xy=road["local_xyz"][:,:2].astype(float);grad=[]
    for axis in (0,1):
        shift=np.zeros_like(xy);shift[:,axis]=delta
        plus,_=surface.height(xy+shift);minus,_=surface.height(xy-shift);grad.append((plus-minus)/(2*delta))
    numeric_normal=norm(np.c_[-grad[0],-grad[1],np.ones(len(xy))]);dot=np.abs((numeric_normal*road["normal_local"]).sum(1))
    positive=road["normal_local"][:,2]>0
    native_asset_mean=np.divide(base["asset_centre_depth_T_m"],bw,out=np.full_like(bw,np.nan),where=bw>1e-6)
    native_road_mean=np.divide(current["newroad_centre_depth_T_m"],ext,out=np.full_like(ext,np.nan),where=ext>1e-6)
    depth_summary={}
    strong=masks["baseline_assetW_gt09_loss_gt001"]
    for label,title in ((1,"chart_first"),(2,"ground_first"),(3,"near_contact"),(0,"uncertain")):
        m=strong&(labels[0]==label)
        depth_summary[title]=dict(pixels=int(m.sum()),chart_surface_height_above_pool_m=stat(chart_height[m]),
            true_reference_ground_camera_z_m=stat(ground_depth[m]),
            native_asset_solo_expected_center_depth_m=stat(solo["depth_m"][m]),
            native_v44_asset_component_expected_center_depth_m=stat(native_asset_mean[m]),
            native_r0_newroad_component_expected_center_depth_m=stat(native_road_mean[m]),
            road_component_center_depth_minus_ground_hit_m=stat((native_road_mean-ground_depth)[m]))
    closures={}
    for name,d in (("v44_delete",base),("r0_delete",current)):
        reconstructed=d["group_rgb_T"].sum(0)+d["env_rgb_T"]+d["constant_background_rgb_T"]
        closures[name]=dict(RGB_layer_sum_max_abs_error=float(np.max(np.abs(reconstructed-d["rgb"]))),
            alpha_layer_sum_max_abs_error=float(np.max(np.abs(d["group_alpha_T"].sum(0)-d["alpha"][0]))))
    report=dict(schema="RAW128_FIXED_RED_C2_CHART_GROUND_CONTACT_CPU_AUDIT_V2",raw=128,native_resolution=[480,320],
        replaces_diagnostic_v1="v1 canonical hit/order/native W statistics are unchanged; dominant C2 row in v2 uses asset-solo weights to avoid argmax(all-zero combined weights) when occluded.",
        registered_roi_xyxy_halfopen=ROI,roi_registration="Native unmodified 480x320 source, parent supplied screenshot location; fixed rectangle before contribution/geometry evidence selection.",
        input_sha256={str(p.relative_to(run.parent)):sha(p) for p in (diag/"v44_delete.npz",diag/"r0_delete.npz",diag/"observations.npz",asset_path,posterior_path,baked_path)},
        ordered_labels={"0":"uncertain: no active chart triangle or unsupported ground","1":"chart_first (>0.05m)","2":"ground_first (>0.05m)","3":"near_contact <=0.05m"},
        canonical_vertices=len(vertices),triangles=len(triangles),UV_max_edge_px=1.5,matte_thresholds=[.01,.5],two_sided=True,
        strata=strata,matte_ordering_changed_pixels=int(np.count_nonzero(roi&(labels[0]!=labels[1]))),
        strong_loss_native_component_depth_summaries=depth_summary,native_readout_layer_closure=closures,
        reconstructed_actual_LOD=dict(du=saved["du"],render_rows=len(local),saved_render_rows=saved["render_rows"],
            all_supported_C2_center_height_m=stat(heights[support>=3])),
        road_geometry=dict(rows=len(xy),height_residual_vs_independent_pool_m=stat(road["local_xyz"][:,2]-rh),
            normal_lengths=stat(normal_len),positive_local_normal_z=int(positive.sum()),normal_vs_numeric_IDW_height_angle_deg=stat(np.degrees(np.arccos(np.clip(dot,0,1)))),
            raw_projected_covariance_min_eigenvalue_px2=stat(eig[:,0]),filter_03_fraction_in_minor_axis=stat(.3/(eig[:,0]+.3))),
        evidence_pixels=records,
        native_component_centre_depth_note="GPU component fields are sum(T alpha center_cameraZ_metres); divide by their component alpha before comparing. Not a first-hit depth.",
        limitations=["Open immutable canonical C2 chart only; retained non-chart Gaussian rows are not meshed. No triangle hit does not imply physical road ownership.",
            "Real observed pixels can contain the removed a37; they are not hidden C2/body or deleted-road truth.",
            "Measured-pool ground IDW surface is independently constrained but approximate, 8 neighbours within0.8m and ray residual<0.03m; no first-hit GT.",
            "CPU Gaussian composition excludes other native scene rows and is only a mechanism diagnostic; native GPU readouts remain authoritative.",
            "Relative-Mip prediction assumes same parent local projection Jacobian; exact child projection factors need current-camera evaluation. No renderer/model change here."],
        GPU_used=False,model_updates=0,elapsed_seconds=time.time()-start)
    target.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    np.savez_compressed(out/"raw128_fixed_red_geometry_v2.npz",roi=roi,labels=labels,ground_valid=ground_valid,ground_depth_m=ground_depth,
        chart_depth_m=hits,ground_minus_chart_depth_m=gaps,assetW_loss=loss,baseline_assetW=bw,current_assetW=nw,newroad_Talpha=ext)
    make_panels(out,obs,base,current,labels[0],loss,ROI)
    print(json.dumps(dict(strong=strata["0.01"]["baseline_assetW_gt09_loss_gt001"],depth_summary=depth_summary,closure=closures,elapsed_seconds=report["elapsed_seconds"])),flush=True)


if __name__=="__main__":main()

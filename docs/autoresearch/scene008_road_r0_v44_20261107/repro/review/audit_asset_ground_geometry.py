"""CPU C2-LOD Gaussian/road geometry diagnosis at saved contribution-loss pixels.

Reconstructs the existing immutable LOD samples from saved du, S and alpha F.
CPU splat weights are an explicitly approximate diagnostic, not a new GPU run
or hidden-body ground truth. World Gaussian height and ground rays are direct.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.interpolate import BSpline
from localize_lateral_domain import Surface,rays


def load(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def trans(x,T):return x@T[:3,:3].T+T[:3,3]
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=1)[:,None],1e-12)
def count(x):return int(np.count_nonzero(x))
def stat(v):
    v=np.asarray(v,float);v=v[np.isfinite(v)]
    return dict(count=len(v),min=float(v.min()) if len(v) else None,median=float(np.median(v)) if len(v) else None,max=float(v.max()) if len(v) else None,mean=float(v.mean()) if len(v) else None)


class LodGeometry:
    def __init__(self,asset,posterior):
        self.a=asset;self.p=posterior;self.chart=asset["representation_kind"]==1;self.keep=~self.chart
        self.uv=asset["resample_reference_pixel_xy"][self.chart].astype(float)
        self.K=posterior["reference_K"];self.C=posterior["reference_camera_from_local_m"]
        self.tu=posterior["u_knots_pixels"];self.tv=posterior["v_knots_pixels"]
        self.coeff=posterior["control_delta_z_m"].reshape(len(self.tu)-4,len(self.tv)-4)
        self.zref=float(posterior["z_reference_m"])
        alpha=np.minimum(asset["opacity"][self.chart]/np.float32(.99),1)
        tex=np.zeros((1280,1920),np.float32);ij=np.rint(self.uv).astype(int);tex[ij[:,1],ij[:,0]]=alpha
        self.mips=[tex]
        while min(self.mips[-1].shape)>1:
            a=self.mips[-1];h,w=a.shape;a=np.pad(a,((0,h%2),(0,w%2)))
            self.mips.append(a.reshape(a.shape[0]//2,2,a.shape[1]//2,2).mean((1,3)))
    def material_alpha(self,uv,du):
        lod=max(0.,math.log2(du));lo=min(math.floor(lod),len(self.mips)-1);hi=min(lo+1,len(self.mips)-1);f=lod-math.floor(lod) if hi!=lo else 0
        def sample(level):
            t=self.mips[level];xy=(uv+.5)/(2**level)-.5;ij=np.floor(xy).astype(int);frac=xy-ij;out=np.zeros(len(uv))
            for oy in (0,1):
                for ox in (0,1):
                    x=ij[:,0]+ox;y=ij[:,1]+oy;valid=(x>=0)&(x<t.shape[1])&(y>=0)&(y<t.shape[0])
                    weight=(frac[:,0] if ox else 1-frac[:,0])*(frac[:,1] if oy else 1-frac[:,1]);out[valid]+=t[y[valid],x[valid]]*weight[valid]
            return out
        return (sample(lo)*(1-f)+sample(hi)*f).clip(0,1).astype(np.float32)
    def samples(self,du):
        a=self.a
        if du==1:return a["xyz"],a["normal"],a["sigma"],a["thickness"],a["opacity"],a["representation_kind"]
        low=self.uv.min(0)-3*du;high=self.uv.max(0)+3*du
        u=np.arange(math.floor((low[0]+.5)/du-.5),math.ceil((high[0]+.5)/du-.5)+1)*du+.5*du-.5
        v=np.arange(math.floor((low[1]+.5)/du-.5),math.ceil((high[1]+.5)/du-.5)+1)*du+.5*du-.5
        xx,yy=np.meshgrid(u,v);uv=np.c_[xx.ravel(),yy.ravel()];alpha=self.material_alpha(uv,du)
        supported=(uv[:,0]>=self.tu[3])&(uv[:,0]<=self.tu[-4])&(uv[:,1]>=self.tv[3])&(uv[:,1]<=self.tv[-4])
        keep=(alpha>1e-12)&supported;uv=uv[keep];alpha=alpha[keep]
        U=BSpline(self.tu,np.eye(len(self.tu)-4),3,extrapolate=False)(uv[:,0])
        V=BSpline(self.tv,np.eye(len(self.tv)-4),3,extrapolate=False)(uv[:,1])
        z=self.zref+np.sum((U@self.coeff)*V,axis=1)
        ray=np.c_[uv+.5,np.ones(len(uv))]@np.linalg.inv(self.K).T
        xyz=((ray*z[:,None]-self.C[:3,3])@np.linalg.inv(self.C[:3,:3]).T).astype(np.float32)
        sigma=(.6*du*z/self.K[0,0]).astype(np.float32)
        k=self.keep;n=len(xyz)
        return (np.r_[a["xyz"][k],xyz],np.r_[a["normal"][k],np.tile([0,0,1.],(n,1))],
                np.r_[a["sigma"][k],sigma],np.r_[a["thickness"][k],sigma],np.r_[a["opacity"][k],np.float32(.99)*alpha],
                np.r_[a["representation_kind"][k],np.ones(n,np.int64)])


def projected_rows(model_xyz,normal_model,sigma_m,thin_m,opacity,camera,scale,neg_fov):
    WV=np.asarray(camera["world_view"],float);P=np.asarray(camera["full_projection"],float)
    camera_xyz=(np.c_[model_xyz,np.ones(len(model_xyz))]@WV)[:,:3]/scale
    clip=np.c_[model_xyz,np.ones(len(model_xyz))]@P;uv=(clip[:,:2]/clip[:,3:]+1)*[240,160]-.5
    nc=norm(normal_model@WV[:3,:3]);s2=sigma_m*sigma_m;t2=thin_m*thin_m
    cov=s2[:,None,None]*np.eye(3)+(t2-s2)[:,None,None]*nc[:,:,None]*nc[:,None,:]
    tx=math.tan(-.5) if neg_fov else math.tan(float(camera["FoVx"])/2)
    ty=math.tan(-.5) if neg_fov else math.tan(float(camera["FoVy"])/2)
    fx,fy=480/(2*tx),320/(2*ty);z=camera_xyz[:,2]
    # Match visible native kernel's min(lim,max(-lim,x/z)) even for its
    # unchanged legacy negative-FOV setting; this is not a proposed fix.
    xx=np.minimum(1.3*tx,np.maximum(-1.3*tx,camera_xyz[:,0]/z))*z
    yy=np.minimum(1.3*ty,np.maximum(-1.3*ty,camera_xyz[:,1]/z))*z
    J=np.zeros((len(z),2,3));J[:,0,0]=fx/z;J[:,1,1]=fy/z;J[:,0,2]=-fx*xx/z**2;J[:,1,2]=-fy*yy/z**2
    c2=J@cov@J.transpose(0,2,1);c2[:,0,0]+=.3;c2[:,1,1]+=.3
    return dict(uv=uv,depth=z,inv_cov=np.linalg.inv(c2),opacity=opacity)


def alphas(rows,pixel):
    d=rows["uv"]-pixel;power=-.5*np.einsum("ni,nij,nj->n",d,rows["inv_cov"],d)
    a=np.minimum(.99,rows["opacity"]*np.exp(np.minimum(power,0)))
    return np.where((rows["depth"]>.2)&(a>=1/255),a,0)


def weights(alpha,depth):
    order=np.argsort(depth,kind="stable");out=np.zeros(len(alpha));T=1.
    for i in order:
        if alpha[i]==0:continue
        nxt=T*(1-alpha[i])
        if nxt<.0001:break
        out[i]=T*alpha[i];T=nxt
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);args=p.parse_args();run=args.run;start=time.time()
    out=run/"review/asset_ground_conflict";target=out/"geometry_depth_evidence.json"
    if target.exists():raise FileExistsError("Preserve geometry diagnosis")
    asset_path=run.parent/"scene008_a6_insert_artifacts_20261105/surface_fit/combined_v1/observed_fit/formal/complete_a6.npz"
    post_path=run.parent/"scene008_a6_insert_artifacts_20261105/surface_fit/masked_depth_v1/posterior.npz"
    assert sha(asset_path)=="0b1f69e13d0e54328734325c8f118bc27dc7b75080869f2d52f55038afc0b20d"
    assert sha(post_path)=="b98a0d932730fbf1462e555eb7d5eb931030b51fe68848990f073b9752879e06"
    asset=load(asset_path);lod=LodGeometry(asset,load(post_path))
    meta=json.loads((run.parent/"r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json").read_text())
    T=np.asarray(meta["world_to_model"])@np.asarray(meta["coordinate_anchor"]);inverse=np.linalg.inv(T);scale=float(meta["meter_to_model"])
    plane=np.asarray(json.loads(Path(meta["original_readiness"]).read_text())["SGCM_geometry"]["ground_fit"]["plane_anchor"])
    surface=Surface(load(run/"surface/measured_ground_support_pool.npz")["local_xyz"],plane)
    road=load(run/"surface/frontfill_v2/road_reference_199.npz");ids=np.flatnonzero(road["active"]&(road["local_xyz"][:,0]>19.75)&(road["local_xyz"][:,1]>=-3.35)&(road["local_xyz"][:,1]<=2.71))
    cfg=json.loads((run.parent/"scene008_blue_global_rgb_probe_20261105_r1/global_10k/effective_config.json").read_text())
    rows=[];lod_checks=[]
    for raw in (130,137):
        obs=load(run.parent/f"scene008_instances_20261105_r1/observations/frames/{raw:03d}.npz")
        frame=load(run.parent/f"r15_gn11_fullclip_cpu_20261026/frames/{raw:03d}.npz")
        metrics=json.loads((run.parent/f"scene008_v44_rgb_asset_20261107_r1/results/full49_r1/v44/{raw:03d}/metrics.json").read_text())
        camera=metrics["actual_camera_pose"];saved_lod=metrics["states"]["delete"]["meta"]["LOD"]
        local,n,s,thin,op,kind=lod.samples(float(saved_lod["du"]))
        assert len(local)==saved_lod["render_rows"]
        pose=obs["model_from_local"][0].astype(np.float32);model=trans(local,pose).astype(np.float32);anchor=trans(model.astype(float),inverse)
        normal_model=norm(n@pose[:3,:3].T);normal_anchor=norm(normal_model@inverse[:3,:3].T)
        h,support=surface.height(anchor[:,:2]);height=anchor[:,2]-h
        actor_scale=float(np.linalg.norm(pose[:3,:3],axis=0).mean());sigma=s.astype(float)*actor_scale/scale;thickness=thin.astype(float)*actor_scale/scale
        sigma_z=np.sqrt(sigma*sigma+(thickness*thickness-sigma*sigma)*normal_anchor[:,2]**2)
        body_rows=projected_rows(model,normal_model,sigma,thickness,op,camera,scale,cfg["neg_fov"])
        nr=norm(road["normal_world"][ids]@np.asarray(meta["world_to_model"])[:3,:3].T/scale)
        road_rows=projected_rows(road["model_xyz"][ids].astype(np.float32),nr,road["gaussian_sigma_m"][ids,0],road["gaussian_sigma_m"][ids,2],np.full(len(ids),.99),camera,scale,cfg["neg_fov"])
        ground_map,ground_valid=rays(surface,frame["K"],frame["w2l"])
        gdepth=trans(ground_map.reshape(-1,3),frame["w2l"])[:,2].reshape(320,480)
        b=load(run/f"results/full49_corridor/{raw:03d}/baseline_delete_float.npz");ca=load(run/f"results/full49_corridor/{raw:03d}/calibrated_delete_float.npz")
        bw=b["group_alpha_T"][0];nw=ca["group_alpha_T"][0];loss=bw-nw
        chosen=obs["box_candidate"][0].astype(bool)&~obs["visible"][0].astype(bool)&(bw>.5)&(loss>.01)
        lod_checks.append(dict(raw=raw,du=saved_lod["du"],render_rows_reconstructed=len(local),render_rows_saved=saved_lod["render_rows"],
            Gaussian_mean_height_to_ground_m=stat(height[support>=3]),Gaussian_3sigma_lower_height_m=stat((height-3*sigma_z)[support>=3])))
        for y,x in np.argwhere(chosen):
            aa=alphas(body_rows,[x,y]);ww=weights(aa,body_rows["depth"]);isoalpha=ww.sum();den=max(isoalpha,1e-20)
            iso_depth=float((ww*body_rows["depth"]).sum()/den);dominant=int(np.argmax(ww))
            ra=alphas(road_rows,[x,y]);jw=weights(np.r_[aa,ra],np.r_[body_rows["depth"],road_rows["depth"]]);joint_asset=float(jw[:len(aa)].sum())
            supported=(support>=3);mass=np.where(supported,ww,0);sm=max(mass.sum(),1e-20)
            top=np.argsort(-ww)[:4]
            rec=dict(raw=raw,pixel_xy=[int(x),int(y)],baseline_assetW=float(bw[y,x]),new_assetW=float(nw[y,x]),assetW_loss=float(loss[y,x]),strong_baseline_gt09=bool(bw[y,x]>.9),
                road_ray_geometry_supported=bool(ground_valid[y,x]),road_ray_local_xyz_m=ground_map[y,x].tolist(),road_ray_camera_z_m=float(gdepth[y,x]) if ground_valid[y,x] else None,
                saved_baseline_expected_depth_m=float(b["depth_m"][y,x]),saved_new_expected_depth_m=float(ca["depth_m"][y,x]),
                CPU_isolated_asset_alpha= float(isoalpha),CPU_isolated_asset_expected_depth_m=iso_depth,CPU_joint_assetW=joint_asset,
                CPU_joint_vs_saved_new_assetW_absdiff=abs(joint_asset-float(nw[y,x])),
                road_ray_minus_CPU_asset_expected_depth_m=float(gdepth[y,x]-iso_depth) if ground_valid[y,x] else None,
                road_ray_minus_dominant_Gaussian_mean_depth_m=float(gdepth[y,x]-body_rows["depth"][dominant]) if ground_valid[y,x] else None,
                isolated_weighted_Gaussian_mean_height_m=float((mass*height).sum()/sm),
                isolated_weight_fraction_mean_below_road_minus002=float(mass[height<-.02].sum()/sm),
                isolated_weight_fraction_3sigma_crosses_road=float(mass[height-3*sigma_z<0].sum()/sm),
                dominant_Gaussian_mean_height_m=float(height[dominant]),dominant_Gaussian_lower3sigma_height_m=float(height[dominant]-3*sigma_z[dominant]),
                original_a37_visible=bool(obs["visible"][1,y,x]),a6_measured_occluded=bool(obs["occluded"][0,y,x]),
                original_scene_first_return_m=float(obs["scene_depth_m"][y,x]),
                top_Gaussians=[dict(row=int(i),kind=int(kind[i]),isotropic_LOD_chart=bool(kind[i]==1),camera_z_m=float(body_rows["depth"][i]),
                    source_support_returns=int(support[i]),local_anchor_xyz_m=anchor[i].tolist(),height_to_road_m=float(height[i]),lower3sigma_height_m=float(height[i]-3*sigma_z[i]),
                    isolated_alphaT=float(ww[i]),projected_mean_xy=body_rows["uv"][i].tolist()) for i in top])
            rows.append(rec)
        print(json.dumps(dict(raw=raw,diagnosed_pixels=count(chosen),seconds=time.time()-start)),flush=True)
    summary=[]
    for raw in (130,137):
        for strong in (False,True):
            selected=[r for r in rows if r["raw"]==raw and (not strong or r["strong_baseline_gt09"])]
            summary.append(dict(raw=raw,selection="baselineW>.9 and loss>.01" if strong else "baselineW>.5 and loss>.01",pixels=len(selected),
                road_ray_valid=sum(r["road_ray_geometry_supported"] for r in selected),
                road_ray_before_isolated_expected_by005=sum(r["road_ray_minus_CPU_asset_expected_depth_m"] is not None and r["road_ray_minus_CPU_asset_expected_depth_m"]<-.05 for r in selected),
                dominant_Gaussian_mean_below_road_minus002=sum(r["dominant_Gaussian_mean_height_m"]<-.02 for r in selected),
                road_ray_minus_isolated_expected_depth_m=stat([r["road_ray_minus_CPU_asset_expected_depth_m"] for r in selected if r["road_ray_minus_CPU_asset_expected_depth_m"] is not None]),
                dominant_Gaussian_mean_height_m=stat([r["dominant_Gaussian_mean_height_m"] for r in selected]),
                dominant_Gaussian_lower3sigma_height_m=stat([r["dominant_Gaussian_lower3sigma_height_m"] for r in selected]),
                CPU_joint_vs_saved_assetW_absdiff=stat([r["CPU_joint_vs_saved_new_assetW_absdiff"] for r in selected])))
    report=dict(schema="R0_C2_LOD_AND_MEASURED_ROAD_GEOMETRY_CONFLICT_DIAGNOSTIC_V1",asset_sha256=sha(asset_path),posterior_sha256=sha(post_path),
        lod_reconstruction=lod_checks,summary=summary,pixels=rows,
        ground="Same measured LiDAR-pool IDW surface as R0, not new road semantics or dense GT; road ray intersections checked independently of rendered expected depth.",
        CPU_splat_approximation="Existing C2 LOD geometry/alpha reconstructed from saved native du; camera cov follows visible native legacy negative-FOV clamp+0.3 low-pass. Approximate CPU isolated and body+road composition excludes other native scene rows and CUDA tiling/FP details; report its discrepancy against saved real GPU assetW.",
        height_semantics="Gaussian mean vertical anchor-height above measured ground; lower3sigma is finite-support diagnostic, not a mesh surface intersection.",
        interpretation="If measured ground is ahead of inferred body depth and the model Gaussian means/extents reach below ground, road-first ordering can explain lost predicted lower body. This does not establish correct hidden vehicle geometry or that every road replacement is physically valid.",
        no_hidden_GT=True,GPU_used=False,checkpoint_loads=0,model_updates=0,seconds=time.time()-start)
    target.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n");print(json.dumps(summary),flush=True)


if __name__=="__main__":main()

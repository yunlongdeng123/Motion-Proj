"""Complete the existing front candidate domain with a fixed rectangular grid.

The rectangle uses only existing candidate XY bounds and the old road's front
extent. It is fixed before any RGB access and uses no raw137 ROI information.
All added cells undergo the unchanged FRONT199 geometric/guard/box-source gates.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4194304),b""): h.update(b)
    return h.hexdigest()


def transform(x,T): return x@T[:3,:3].T+T[:3,3]


def sample(a,uv):
    if a.ndim==3: return np.stack([sample(a[:,:,c],uv) for c in range(a.shape[-1])],1)
    return map_coordinates(a.astype(float),[uv[:,1],uv[:,0]],order=1,mode="constant",cval=0)


def ground(f,plane):
    p=f["cloud"]
    good=(np.abs(p@plane[:3]+plane[3])<=.15)&(p[:,0]>=.5)&(p[:,0]<=90)&(np.abs(p[:,1])<=8)
    for c,h,ax in zip(f["box_center"],f["box_half"],f["box_axes"]):
        ids=np.flatnonzero(good)
        good[ids]&=~(np.abs((p[ids]-c)@ax)<=h).all(1)
    return good


class Surface:
    def __init__(self,points,plane):
        self.p=points.astype(float);self.tree=cKDTree(points[:,:2]);self.plane=-plane[[0,1,3]]/plane[2]
    def height(self,xy):
        d,ids=self.tree.query(xy,k=8,distance_upper_bound=.8);finite=np.isfinite(d)
        points=self.p[np.minimum(ids,len(self.p)-1)]
        residual=points[:,:,2]-points[:,:,:2]@self.plane[:2]-self.plane[2]
        weights=np.where(finite,1/np.maximum(d,.05)**2,0)
        return xy@self.plane[:2]+self.plane[2]+(weights*residual).sum(1)/np.maximum(weights.sum(1),1e-12),finite.sum(1)>=3
    def normal(self,xy):
        dx=(self.height(xy+[.05,0])[0]-self.height(xy-[.05,0])[0])/.1
        dy=(self.height(xy+[0,.05])[0]-self.height(xy-[0,.05])[0])/.1
        normal=np.c_[-dx,-dy,np.ones(len(xy))]
        return normal/np.linalg.norm(normal,axis=1)[:,None]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--base",type=Path,default=Path("/root/autodl-tmp/drivingeditor_repro/campaign"))
    args=parser.parse_args();base,run=args.base,args.run
    out=run/"surface/frontfill_v2";out.mkdir(parents=True,exist_ok=True)
    target=out/"evidence.npz"
    if target.exists():raise FileExistsError("Preserve prior frontfill evidence")
    started=time.time()
    sys.path.insert(0,str(base/"r15_gn11_stage1_v41_20261023/tools"))
    from v41_dense_geometry import measured_patch_evidence,ray_box_blocked,project
    source=base/"shadow_a5_a37_199_20261103/inputs/expanded/evidence.npz"
    source_sha=sha256(source)
    meta=json.loads((base/"r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json").read_text())
    plane=np.asarray(json.loads(Path(meta["original_readiness"]).read_text())["SGCM_geometry"]["ground_fit"]["plane_anchor"])
    anchor=np.asarray(meta["coordinate_anchor"]);world_to_model=np.asarray(meta["world_to_model"])
    with np.load(source,allow_pickle=False) as f:a={k:f[k] for k in f.files}
    with np.load(base/"scene008_native_road1k_20261105_r1/inputs/road_reference.npz",allow_pickle=False) as f:
        old_front=float(f["local_xyz"][f["active"].astype(bool),0].max())
    cell=float(a["cell_m"]);keys0=a["keys"].astype(np.int64)
    # Coordinate-only construction. Every cell in the existing y range and
    # strictly in front of the old road through the existing x maximum is used.
    xkeys=np.arange(int(np.floor(old_front/cell)),int(keys0[:,0].max())+1)
    xkeys=xkeys[(xkeys+.5)*cell>old_front]
    ykeys=np.arange(int(keys0[:,1].min()),int(keys0[:,1].max())+1)
    xx,yy=np.meshgrid(xkeys,ykeys,indexing="ij")
    rectangle=np.c_[xx.ravel(),yy.ravel()]
    existing={tuple(k) for k in keys0}
    added=np.array([k for k in rectangle if tuple(k) not in existing],np.int64)
    assert len(added)>0
    keys=np.r_[keys0,added];xy=(added+.5)*cell
    with np.load(run/"surface/measured_ground_support_pool.npz",allow_pickle=False) as f:pool=f["local_xyz"]
    surface=Surface(pool,plane);height,supported=surface.height(xy);normal=surface.normal(xy);xyz=np.c_[xy,height]
    extraction=json.loads((base/"r15_gn11_fullclip_cpu_20261026/extraction.json").read_text())
    sharp=np.array([r["sharpness"] for r in extraction["frames"]]);quality=np.minimum(sharp/max(np.median(sharp[89:138]),1e-9),1.)
    allvalid=[];allrgb=[];rows=[]
    for raw in range(199):
        with np.load(base/f"r15_gn11_fullclip_cpu_20261026/frames/{raw:03d}.npz",allow_pickle=False) as f:
            K,w=f["K"],f["w2l"];uv,z=project(xyz,K,w);origin=np.linalg.inv(w)[:3,3]
            ray=origin-xyz;length=np.linalg.norm(ray,axis=1)
            cosine=np.abs((ray/np.maximum(length[:,None],1e-9)*normal).sum(1))
            area=K[0,0]*K[1,1]*.2**2*cosine/np.maximum(z*z,1e-8)
            candidate=supported&(z>.1)&(uv[:,0]>=0)&(uv[:,0]<479)&(uv[:,1]>=0)&(uv[:,1]<319)&(sample(f["guard"],uv)==0)&(cosine>=.12)&(area>=.25)&(quality[raw]>=.2)
            ids=np.flatnonzero(candidate);valid=np.zeros(len(xyz),bool)
            if len(ids):
                ok,_,_,_=measured_patch_evidence(f["cloud"],ground(f,plane),K,w,xyz[ids],uv[ids],normal[ids])
                boxes=[dict(center=c,half=h,axes=ax) for c,h,ax in zip(f["box_center"],f["box_half"],f["box_axes"])]
                valid[ids]=ok&~ray_box_blocked(xyz[ids],origin,boxes)
            # Sample only valid source positions: no vehicle or rejected pixel
            # enters the added-cell RGB cache, even as an unused finite value.
            rgb=np.full((len(xyz),3),np.nan,np.float32)
            if valid.any():
                image=np.asarray(Image.open(base/f"r15_gn11_fullclip_cpu_20261026/rgb/{raw:03d}.png"),np.float32)/255
                rgb[valid]=sample(image,uv[valid])
            allvalid.append(valid);allrgb.append(rgb)
            rows.append(dict(raw=raw,added_qualified_sources=int(valid.sum())))
        if raw%25==0 or raw==198:
            print(json.dumps(dict(raw=raw,added_cells=len(added),qualified=int(valid.sum()),seconds=time.time()-started)),flush=True)
    extra_valid=np.asarray(allvalid);extra_rgb=np.asarray(allrgb)
    valid=np.c_[a["source_valid"],extra_valid]
    rgb=np.concatenate([np.where(a["source_valid"][:,:,None],a["source_rgb"],np.nan),extra_rgb],axis=1)
    local=np.r_[a["local_xyz"],xyz].astype(np.float32)
    world=np.r_[a["world_xyz"],transform(xyz,anchor)]
    model=np.r_[a["model_xyz"],transform(transform(xyz,anchor),world_to_model)]
    np.savez_compressed(target,keys=keys,local_xyz=local,world_xyz=world,model_xyz=model,
        normal_local=np.r_[a["normal_local"],normal].astype(np.float32),
        normal_world=np.r_[a["normal_world"],normal@anchor[:3,:3].T].astype(np.float32),
        gaussian_sigma_m=np.r_[a["gaussian_sigma_m"],np.tile([cell*.65,cell*.65,.015],(len(added),1))],
        surface_supported=np.r_[a["surface_supported"],supported],source_valid=valid,source_rgb=rgb,
        source_raw=np.arange(199,dtype=np.int16),cell_m=cell,
        candidate_origin=np.r_[np.zeros(len(keys0),np.uint8),np.ones(len(added),np.uint8)])
    assert sha256(source)==source_sha
    report=dict(schema="R0_FRONTFILL_FIXED_GEOMETRIC_DOMAIN_V2",candidate_domain_fix=True,
        original_candidates=len(keys0),added_candidates=len(added),total_candidates=len(keys),
        rectangle_candidate_cells=len(rectangle),old_road_front_m=old_front,
        fixed_rectangle_center_bounds_xy_m=[[(xkeys.min()+.5)*cell,(ykeys.min()+.5)*cell],[(xkeys.max()+.5)*cell,(ykeys.max()+.5)*cell]],
        rectangle_rule="All8cm centers with x>old-current-active-xmax through existing-expanded-xmax, and y spanning existing-expanded min/max; union with all immutable original candidates. Geometry-only rule fixed before RGB access.",
        added_geometry_supported=int(supported.sum()),added_cells_with_any_source=int(extra_valid.any(0).sum()),
        added_source_records=int(extra_valid.sum()),source_frames=199,source_camera="FRONT",
        source_path=str(source),source_sha256=source_sha,source_unchanged=True,
        evidence_path=str(target),evidence_sha256=sha256(target),per_frame=rows,
        RGB_policy="Only qualified positions sampled. No rejected/vehicle pixels, residual/model RGB, neighbour colour or new stable-colour thresholds.",
        source_rules="Unchanged original expanded evidence: measured triangle edges<=1.6m/24px, normalcos>=.9, depthdiff<=.3m, conservative2D guard clear,3Dboxes ray clear, cosine>=.12,20cm evidence area>=.25px2,quality>=.2.",
        fixed_sigmas_m=[cell*.65,cell*.65,.015],GPU_used=False,checkpoint_loads=0,optimizer_updates=0,
        target137_ROI_used_in_generation=False,seconds=time.time()-started)
    (out/"frontfill_evidence_receipt.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:report[k] for k in ("total_candidates","added_candidates","added_geometry_supported","added_cells_with_any_source","evidence_sha256","seconds")}),flush=True)


if __name__=="__main__":main()

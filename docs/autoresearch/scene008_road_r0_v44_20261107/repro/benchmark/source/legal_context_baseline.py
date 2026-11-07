"""Independent CPU estimator: one patch, one explicit input allowlist.

No imports of preparation/evaluator modules. All model-context reads after the
audit hook must be exact files in this one copied track. This is auditable
Python path isolation, not an OS/container security certificate.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[key]="1"
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import tempfile
import sysconfig
# The runner uses -S: skip site/.pth startup side effects and expose only the
# existing environment's explicit scientific packages, without project paths.
if sys.flags.no_site:
    package_path=sysconfig.get_paths()["purelib"]
    if package_path not in sys.path:
        sys.path.append(package_path)
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree, Delaunay, QhullError
from scipy.ndimage import map_coordinates, binary_erosion
sys.dont_write_bytecode=True


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def xyinside(p,box):
    return (p[:,0]>=box[0])&(p[:,0]<=box[2])&(p[:,1]>=box[1])&(p[:,1]<=box[3])


def project(points,camera):
    w=np.asarray(camera["anchor_local_to_camera"]);q=points@w[:3,:3].T+w[:3,3]
    positive=q[:,2]>.1
    x=np.divide(q[:,0],q[:,2],out=np.zeros(len(q)),where=positive)
    y=np.divide(q[:,1],q[:,2],out=np.zeros(len(q)),where=positive)
    fu,fv,cu,cv,k1,k2,p1,p2,k3=camera["intrinsic_Brown"]
    r=x*x+y*y;rad=1+k1*r+k2*r*r+k3*r*r*r
    xd=x*rad+2*p1*x*y+p2*(r+2*x*x);yd=y*rad+p1*(r+2*y*y)+2*p2*x*y
    uv=np.c_[(fu*xd+cu)/4-.5,(fv*yd+cv)/4-.5]
    uv[~positive]=-1e6
    return uv,q[:,2]


def fit_plane(points,settings):
    cell=settings["geometry_bin_m"];keys=np.floor(points[:,:2]/cell).astype(int)
    unique,inverse,counts=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
    order=np.argsort(inverse,kind="stable");ends=np.cumsum(counts);begins=np.r_[0,ends[:-1]]
    bank=[]
    for i,(start,end) in enumerate(zip(begins,ends)):
        if counts[i]>=settings["min_points_per_geometry_bin"]:
            z=float(np.quantile(points[order[start:end],2],settings["bin_height_quantile"]))
            bank.append([*(unique[i]+.5)*cell,z,int(counts[i])])
    bank=np.asarray(bank,float)
    if len(bank)<settings["minimum_plane_bins"]:
        return None,bank,dict(reason="insufficient_legal_geometry_bins")
    design=np.c_[bank[:,:2],np.ones(len(bank))];z=bank[:,2]
    rng=np.random.default_rng(settings["RANSAC_seed"]);best=None
    for _ in range(settings["RANSAC_trials"]):
        ids=rng.choice(len(bank),3,replace=False)
        try:
            plane=np.linalg.solve(design[ids],z[ids])
        except np.linalg.LinAlgError:
            continue
        if np.degrees(np.arctan(np.linalg.norm(plane[:2])))>settings["maximum_slope_degrees"]:
            continue
        residual=np.abs(design@plane-z);inside=residual<=settings["plane_inlier_threshold_m"]
        score=(int(inside.sum()),-float(np.median(residual[inside])) if inside.any() else -999.)
        if best is None or score>best[0]:
            best=(score,plane,inside)
    if best is None or best[0][0]<settings["minimum_plane_bins"]:
        return None,bank,dict(reason="no_robust_context_plane")
    plane=best[1];keep=best[2]
    for _ in range(settings["IRLS_steps"]):
        residual=np.abs(design@plane-z)
        weight=np.minimum(1.,settings["Huber_delta_m"]/np.maximum(residual,1e-9))*keep
        plane=np.linalg.lstsq(design*np.sqrt(weight[:,None]),z*np.sqrt(weight),rcond=None)[0]
    road=np.abs(design@plane-z)<=settings["road_context_plane_residual_max_m"]
    return plane,bank[road],dict(reason=None,input_geometry_bins=len(bank),plane_inlier_bins=int(road.sum()),
                               context_fit_median_abs_m=float(np.median(np.abs(design@plane-z)[road])))


def install_read_allowlist(inputs,outputs,manifest):
    allowed=set()
    for rel in manifest["input_files"]:
        relative=Path(rel);path=(inputs/relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or (inputs not in path.parents) or (inputs/relative).is_symlink():
            raise ValueError("Nonlocal/symlinked input in estimator allowlist")
        allowed.add(str(path))
    allowed.add(str((inputs/"input_manifest.json").resolve()))
    records=[];denied=[]
    library_roots=[Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve(),Path("/usr/lib").resolve(),Path("/usr/local/lib").resolve()]
    def under(path,root):
        return path==root or root in path.parents
    def hook(event,args):
        if event in ("subprocess.Popen","os.system","socket.connect"):
            raise PermissionError("R1 estimator disallows subprocess/network escapes")
        if event!="open" or not args or isinstance(args[0],int):
            return
        path=Path(os.fsdecode(args[0])).resolve();mode=args[1] if len(args)>1 else "r"
        flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
        writes=(isinstance(mode,str) and any(x in mode for x in ("w","a","+","x"))) or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if under(path,outputs):
            records.append(dict(kind="output_write" if writes else "output_read",path=str(path)))
            return
        if not writes and str(path) in allowed:
            records.append(dict(kind="allowlisted_input_read",path=str(path)));return
        if not writes and any(under(path,root) for root in library_roots) and path.suffix.lower() in (".py",".pyc",".so",".txt",".json",".pem"):
            records.append(dict(kind="runtime_library_read",path=str(path)));return
        denied.append(dict(path=str(path),write=bool(writes)))
        raise PermissionError("R1 estimator blocked a non-allowlisted path")
    sys.addaudithook(hook)
    return records,denied


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--inputs",type=Path,required=True);parser.add_argument("--patch-id",required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--denial-probe",type=Path,required=True)
    args=parser.parse_args();started=time.time();inputs=args.inputs.resolve();out=args.output.resolve()
    if out.exists():
        raise FileExistsError("Each estimator runs in a new patch work directory")
    out.mkdir(parents=True)
    # Qhull's diagnostic stream uses a real temporary file. Keep that file
    # inside this independent output directory rather than weakening reads.
    tempfile.tempdir=str(out)
    os.environ["TMPDIR"]=str(out)
    manifest=read(inputs/"input_manifest.json")
    # Reading the allowlist itself is the only pre-hook data operation.
    records,denied=install_read_allowlist(inputs,out,manifest)
    for rel,expected in manifest["input_files"].items():
        if sha(inputs/rel)!=expected["sha256"]:
            raise ValueError("Allowlisted estimator input content changed")
    blocked=False
    try:
        with args.denial_probe.open("rb") as forbidden:
            forbidden.read(1)
    except PermissionError:
        blocked=True
    if not blocked:
        raise AssertionError("Explicit forbidden-path probe was not denied")
    settings=read(inputs/"baseline_settings.json");patches=read(inputs/"patch_manifest.json")["patches"]
    patch=next(p for p in patches if p["patch_id"]==args.patch_id);center=np.asarray(patch["center_local_xy_m"])
    zinterval=patch["context_z_interval_m"];all_points=[]
    for raw in manifest["raw_ids"]:
        with np.load(inputs/"geometry"/f"{raw:03d}.npz",allow_pickle=False) as source:
            xyz=source["local_xyz"].astype(float)
        use=(np.linalg.norm(xyz[:,:2]-center,axis=1)<=settings["context_radius_m"])&(xyz[:,2]>=zinterval[0])&(xyz[:,2]<=zinterval[1])
        all_points.append(xyz[use])
    points=np.concatenate(all_points) if all_points else np.empty((0,3))
    plane,bank,geometry=fit_plane(points,settings)
    size=.08;box=patch["xy_bounds_local_m"]
    xx=np.arange(int(np.floor(box[0]/size))-1,int(np.ceil(box[2]/size))+1)
    yy=np.arange(int(np.floor(box[1]/size))-1,int(np.ceil(box[3]/size))+1)
    keys=np.array([[x,y] for x in xx for y in yy]);xy=(keys+.5)*size;pick=xyinside(xy,box);keys=keys[pick];xy=xy[pick]
    predicted_height=np.zeros(len(keys));height_supported=np.zeros(len(keys),bool)
    predicted_RGB=np.full((len(keys),3),.5);colour_supported=np.zeros(len(keys),bool)
    sample_rgb=[];sample_bank=[];sample_raw=[];legal_colour_rows=[]
    if plane is not None:
        predicted_height=np.c_[xy,np.ones(len(xy))]@plane
        if len(bank)>=3:
            try:
                height_supported=Delaunay(bank[:,:2]).find_simplex(xy)>=0
            except QhullError:
                height_supported[:]=False
        legal=np.ones(len(bank),bool)
        for other in patches:
            legal&=~xyinside(bank,other["guard_xy_bounds_local_m"])
        queries=bank[legal,:3]
        bank_ids=np.flatnonzero(legal)
        for raw in manifest["raw_ids"]:
            camera=read(inputs/"camera"/f"{raw:03d}.json")
            with np.load(inputs/"geometry"/f"{raw:03d}.npz",allow_pickle=False) as source:
                cloud=source["local_xyz"].astype(float)
            uv,z=project(queries,camera);cuv,cz=project(cloud,camera)
            cloudinside=(cz>.1)&(cuv[:,0]>=0)&(cuv[:,0]<480)&(cuv[:,1]>=0)&(cuv[:,1]<320)
            cids=np.flatnonzero(cloudinside)
            if not len(cids) or not len(queries):
                legal_colour_rows.append(dict(raw=raw,accepted_road_samples=0));continue
            near_dist,near=cKDTree(cuv[cids]).query(uv);near=cids[near]
            with np.load(inputs/"rgb"/f"{raw:03d}_valid.npz",allow_pickle=False) as source:
                valid=binary_erosion(source["valid"].astype(bool),iterations=1)
            with np.load(inputs/"guard"/f"{raw:03d}.npz",allow_pickle=False) as source:
                guard=source["vehicle_guard"].astype(bool)
            inside=(z>.1)&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<479)&(uv[:,1]>=0)&(uv[:,1]<319)
            ix=np.zeros(len(uv),int);iy=np.zeros(len(uv),int)
            ids=np.flatnonzero(inside)
            ix[ids]=np.floor(uv[ids,0]+.5).astype(int);iy[ids]=np.floor(uv[ids,1]+.5).astype(int)
            clean=np.zeros(len(queries),bool);ids=np.flatnonzero(inside)
            clean[ids]=valid[iy[ids],ix[ids]]&~guard[iy[ids],ix[ids]]
            near_height=np.c_[cloud[near,:2],np.ones(len(near))]@plane
            accept=inside&clean&(near_dist<=settings["nearest_return_max_distance_px"])&(np.abs(cz[near]-z)<=settings["nearest_return_depth_tolerance_m"])&(np.abs(cloud[near,2]-near_height)<=settings["road_context_plane_residual_max_m"])
            ids=np.flatnonzero(accept)
            with Image.open(inputs/"rgb"/f"{raw:03d}.png") as image:
                rgb=np.asarray(image,float)/255.
            if len(ids):
                colours=np.stack([map_coordinates(rgb[:,:,c],[uv[ids,1],uv[ids,0]],order=1,mode="nearest") for c in range(3)],axis=1)
                sample_rgb.append(colours);sample_bank.append(bank_ids[ids]);sample_raw.append(np.full(len(ids),raw,int))
            legal_colour_rows.append(dict(raw=raw,accepted_road_samples=len(ids)))
        colour_points=[];colours=[];frame_counts=[]
        if sample_rgb:
            srgb=np.concatenate(sample_rgb);sb=np.concatenate(sample_bank);sraw=np.concatenate(sample_raw)
            for i in np.unique(sb):
                mask=sb==i;unique_frames=len(np.unique(sraw[mask]))
                if unique_frames>=settings["minimum_colour_source_frames"]:
                    colour_points.append(bank[i,:2]);colours.append(np.median(srgb[mask],axis=0));frame_counts.append(unique_frames)
        if len(colour_points)>=settings["minimum_nearest_colour_cells"]:
            colour_points=np.asarray(colour_points);colours=np.asarray(colours)
            count=min(settings["nearest_colour_cells"],len(colour_points))
            distance,near=cKDTree(colour_points).query(xy,k=count)
            if distance.ndim==1:
                distance=distance[:,None];near=near[:,None]
            for i in range(len(keys)):
                legal_near=distance[i]<=settings["maximum_colour_continuation_distance_m"]
                if int(legal_near.sum())>=settings["minimum_nearest_colour_cells"]:
                    predicted_RGB[i]=np.median(colours[near[i,legal_near]],axis=0);colour_supported[i]=True
        geometry["legal_colour_world_cells"]=len(colour_points)
    path=out/"prediction.npz"
    np.savez_compressed(path,keys=keys,rgb_srgb=predicted_RGB.astype(np.float32),height_local_m=predicted_height.astype(np.float32),
                        RGB_supported=colour_supported,height_supported=height_supported,
                        normal_local=np.tile(np.array([-plane[0],-plane[1],1.])/np.linalg.norm(np.array([-plane[0],-plane[1],1.])) if plane is not None else [0.,0.,1.],(len(keys),1)))
    receipt=dict(schema="scene008.R1.independent_estimator_receipt.v1",method=settings["method"],patch_id=args.patch_id,track=manifest["track"],
        process_id=os.getpid(),independent_subprocess=True,working_directory=str(Path.cwd()),input_root=str(inputs),
        Python_no_site_startup=bool(sys.flags.no_site),
        input_manifest_sha256=sha(inputs/"input_manifest.json"),settings_sha256=sha(inputs/"baseline_settings.json"),prediction_sha256=sha(path),
        all_allowlisted_input_hashes_verified=True,
        candidate_seeds=[settings["RANSAC_seed"]],parameter_search=False,target_GT_read=False,target_geometry_for_filter_or_mask=False,
        geometry_context_points=len(points),geometry_context=geometry,target_cells=len(keys),RGB_supported_cells=int(colour_supported.sum()),
        height_supported_cells=int(height_supported.sum()),plane_local_coefficients=None if plane is None else plane.tolist(),
        plane_slope_degrees=None if plane is None else float(np.degrees(np.arctan(np.linalg.norm(plane[:2])))),
        legal_colour_source_rows=legal_colour_rows,forbidden_path_probe_denied=blocked,denied_open_attempts=denied,
        actual_open_events=records,all_actual_context_reads_in_one_track_allowlist=True,
        actual_filesystem_security="Python audit hook and explicit file allowlist only; not OS/container isolation or hostile-code sandbox",
        unused_allowed_raws=sorted(set(manifest["raw_ids"])-set(manifest["raw_ids"])),
        numerical_output_is_not_quality_acceptance=True,seconds=time.time()-started,GPU_used=False,neural_weight_updates=0)
    (out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({k:receipt[k] for k in ("patch_id","track","RGB_supported_cells","height_supported_cells","target_cells","seconds")}),flush=True)


if __name__=="__main__":
    main()

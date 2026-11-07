"""Matched frozen-world-strip R0 V2 material field: same 11x9 ridge method.

prepare/cpu-check/solve are CPU-only. smoke and resume require explicit dispatch.
No optimizer, geometry/opacity/native update, screen compositing or hidden GT.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

CAM=Path("/root/autodl-tmp/drivingeditor_repro/campaign")
RUN=CAM/"scene008_road_r0_v44_20261107_r1"
V2_SHA="6112be03108b5a9b1a6deb2f605ef850dcd8ba5018346fa822993177815807a2"
HELD=(89,95,101,105,113,121,125,130,137)
BUILD=tuple(r for r in range(89,138) if r not in HELD)
GRID=(11,9)
MODE_BATCH=21
PRIOR=.01
SMOOTH=.05
MAX_DELTA=.15
STATES=("keep","delete")


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as s:
        for b in iter(lambda:s.read(4194304),b""):h.update(b)
    return h.hexdigest()


def ahash(a):
    a=np.ascontiguousarray(a)
    h=hashlib.sha256(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes());return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def basis(xy,low,high):
    scaled=np.clip((np.asarray(xy)-low)/(high-low)*(np.array(GRID)-1),0,np.array(GRID)-1)
    index=np.minimum(scaled.astype(int),np.array(GRID)-2);fraction=scaled-index
    out=np.zeros((len(xy),99),np.float32)
    for ox in (0,1):
        for oy in (0,1):
            w=(fraction[:,0] if ox else 1-fraction[:,0])*(fraction[:,1] if oy else 1-fraction[:,1])
            out[np.arange(len(xy)),(index[:,0]+ox)*9+index[:,1]+oy]=w
    assert np.max(np.abs(out.sum(1)-1))<2e-6
    return out


def edge_matrix():
    rows=[]
    for x in range(11):
        for y in range(9):
            for dx,dy in ((1,0),(0,1)):
                if x+dx<11 and y+dy<9:
                    r=np.zeros(99);r[x*9+y]=1;r[(x+dx)*9+y+dy]=-1;rows.append(r)
    result=np.asarray(rows);assert result.shape==(178,99);return result


def system(normal,rhs,pixels):
    d=edge_matrix()
    return normal/pixels+(PRIOR/99)*np.eye(99)+(SMOOTH/len(d))*(d.T@d),rhs/pixels


def constrain(coeff,b,source):
    delta=b.astype(np.float64)@coeff
    scale=min(1.,MAX_DELTA/max(float(np.abs(delta).max()),1e-30))
    positive=delta>0;negative=delta<0
    if positive.any():scale=min(scale,float(((1-source)[positive]/delta[positive]).min()))
    if negative.any():scale=min(scale,float((source[negative]/-delta[negative]).min()))
    scale=max(0.,scale)
    if scale<1:scale*=1-1e-7
    final_coeff=coeff*scale;final_delta=b.astype(np.float64)@final_coeff
    colours=(source+final_delta).astype(np.float32)
    assert np.isfinite(colours).all() and colours.min()>=0 and colours.max()<=1
    assert np.abs(final_delta).max()<=MAX_DELTA+1e-12
    return final_coeff,colours,dict(global_scale=scale,raw_delta_maxabs=float(np.abs(delta).max()),
        final_delta_maxabs=float(np.abs(final_delta).max()),colour_min=float(colours.min()),colour_max=float(colours.max()))


def cpu_check(args):
    rng=np.random.default_rng(7);xy=rng.uniform([0,0],[10,8],(310,2));b=basis(xy,np.array([0,0]),np.array([10,8]))
    assert np.all(b>=0) and np.all((b>0).sum(1)<=4)
    phi=b[:211].astype(float)*rng.uniform(0,1,(211,1));res=rng.normal(0,.02,(211,3))
    normal=phi.T@phi;rhs=phi.T@res;a,y=system(normal,rhs,len(phi));co=np.linalg.solve(a,y)
    d=edge_matrix();gradient=(normal@co-rhs)/len(phi)+(PRIOR/99)*co+(SMOOTH/178)*(d.T@d@co)
    assert abs(gradient).max()<1e-12
    source=rng.uniform(.05,.95,(310,3));limited,colours,policy=constrain(co*1000,b,source)
    assert policy["global_scale"]<1
    args.output.mkdir(parents=True,exist_ok=True)
    target=args.output/"cpu_check_state.npz"
    np.savez_compressed(target,normal=normal,rhs=rhs,coeff=co)
    restored=load(target)
    assert all(ahash(restored[k])==ahash(v) for k,v in dict(normal=normal,rhs=rhs,coeff=co).items())
    result=dict(status="PASS_WORLD_FIELD_CPU_CHECKS",basis_partition=True,edge_count=178,
        objective_gradient_maxabs=float(abs(gradient).max()),constraint=policy,save_restore_bitwise=True,GPU_used=False)
    write(args.output/"cpu_checks.json",result);print(json.dumps(result),flush=True)


def prepare(args):
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/"field_contract.json").exists():raise FileExistsError("Preserve prepared field contract")
    assert sha(args.asset)==V2_SHA
    asset=load(args.asset);ids=np.flatnonzero(asset["active"].astype(bool)&(asset["local_xyz"][:,0]>19.75))
    xy=asset["local_xyz"][ids,:2].astype(np.float64)
    v44=read(CAM/"scene008_v44_rgb_asset_20261107_r1/contract/input_contract.json")
    fine_design=Path(v44["paths"]["fine"])/"design.json"
    old_knots=read(fine_design)["knots"]
    strip_low=float(old_knots["low_m"][1]);strip_high=float(old_knots["high_m"][1])
    support=(xy[:,1]>=strip_low)&(xy[:,1]<=strip_high)
    assert support.sum()==10098 and (~support).sum()==4563
    low=xy[support].min(0);high=xy[support].max(0);b=basis(xy,low,high)
    source=asset["reference_srgb"][ids].copy();assert np.isfinite(source).all()
    input_files={str(args.asset):V2_SHA,str(Path(__file__).resolve()):sha(__file__)}
    r0source=RUN/"source/r0_runtime.py";input_files[str(r0source)]=sha(r0source)
    corridor=RUN/"source/corridor_runtime.py";input_files[str(corridor)]=sha(corridor)
    input_files[str(fine_design)]=sha(fine_design)
    masks={};counts={}
    observations=CAM/"scene008_instances_20261105_r1/observations/frames"
    road=CAM/"scene008_blue_global_road1k_20261105_r1/inputs/views"
    for raw in BUILD:
        op=observations/f"{raw:03d}.npz";vp=road/f"{raw:03d}.npz";obs=load(op);view=load(vp)
        mask=obs["road"].astype(bool)&~obs["visible"].astype(bool).any(0)
        mask&=~obs["visible_reviewed"].astype(bool).any(0)&~obs["box_candidate"].astype(bool).any(0)
        mask&=~view["other_vehicle_guard"].astype(bool)
        assert mask.shape==(320,480) and mask.any()
        masks[f"raw{raw:03d}"]=mask;counts[str(raw)]=int(mask.sum())
        input_files[str(op)]=sha(op);input_files[str(vp)]=sha(vp)
    np.savez_compressed(args.output/"prepared.npz",extension_ids=ids,basis=b,source_colours_srgb=source,
        local_xy_m=xy,low_m=low,high_m=high,grid=np.array(GRID),frame_order=np.array(BUILD),held_raws=np.array(HELD),
        support_active=support,strip_lateral_bounds_m=np.array([strip_low,strip_high]))
    np.savez_compressed(args.output/"supervision_masks.npz",**masks)
    contract=dict(schema="R0_MATCHED_STRIP_WORLD_MATERIAL_FIELD_11X9_V1",created_utc=datetime.now(timezone.utc).isoformat(),
        asset=str(args.asset),asset_sha256=V2_SHA,r0_source=str(r0source),grid=list(GRID),modes=99,mode_batches=5,
        extension_ids_count=len(ids),extension_ids_sha256=ahash(ids),knots=dict(low_m=low.tolist(),high_m=high.tolist()),
        support=dict(policy="same CorridorRuntime persistent world strip; not per-frame ROI",active_rows=int(support.sum()),
            inactive_rows=int((~support).sum()),lateral_bounds_m=[strip_low,strip_high],mask_sha256=ahash(support),
            knots="bounds of active XY only",inactive_material="exact original V2 source RGB bytes"),
        frame_order=list(BUILD),HELD9_excluded_from_field_fit=list(HELD),smoke_frames=list(BUILD[:3]),resume_frames=list(BUILD[3:]),
        features=dict(retain_depth_one=[0,1],basis_columns=[2,23],extension_one_column=23,
            raster_extra_basis_columns=[0,21],raster_extra_extension_one=21,
            padding="all other feature columns zero; all non-extension rows columns2+ zero"),
        supervision="Original observed BUILD40 road AND exclude visible/reviewed/box-candidate A6/A37 and frozen other_vehicle_guard. Fixed independent of output alpha/RGB.",
        states=list(STATES),states_equal_weight=True,
        per_frame_per_state_supervision_pixels=counts,
        unique_frame_supervision_pixels=sum(counts.values()),total_supervision_pixels=2*sum(counts.values()),
        objective=dict(data="sum_pixel_RGB((Phi@C-(GT-currentRGB))^2)/(3*P)",
            prior="0.01*mean(C^2)",smooth="0.05*mean((D@C)^2), 178 undirected grid edges",
            normal_equation="(XtX/P+(.01/99)*I+(.05/178)*DtD) C = Xty/P",
            pixel_weight="Every legal road pixel equally weighted; Phi contains actual composited extension visibility"),
        constraint="One global coefficient scale on support_active rows ensures |basis@C|<=.15 and all active sourceRGB+basis@C in[0,1]; inactive colours copied exactly from source; no pointwise colour clipping",
        preserved="V2 xyz/normal/sigma/opacity, registered CorridorRuntime world support mask, old road/native/C2/gates/env/camera; shared field in keep/delete; no retirement",
        checkpoint="After first3 BUILD: all normal equations/frame order/count/counters saved; bitwise CPU restore; new process resumes remaining37",
        expected_native_forwards=400,smoke_native_forwards=30,torch_allocator_fraction=.30,minimum_preflight_free_MiB=20*1024,
        allowed_other_GPU_pids=[13030,15730],
        pretrained_or_learned_prior_claim=False,hidden_GT_used=False,parameter_sweep=False,
        HELD_note="HELD9 is excluded from this field solve; original V2 source proxy already uses all199, so this is not a blinded road benchmark",
        immutable_input_sha256=input_files,prepared_sha256=sha(args.output/"prepared.npz"),
        masks_sha256=sha(args.output/"supervision_masks.npz"))
    write(args.output/"field_contract.json",contract)
    print(json.dumps(dict(contract=str(args.output/"field_contract.json"),extension_rows=len(ids),pixels=2*sum(counts.values()),forwards=400)),flush=True)


def frozen(args):
    contract=read(args.output/"field_contract.json")
    assert sha(args.output/"prepared.npz")==contract["prepared_sha256"]
    assert sha(args.output/"supervision_masks.npz")==contract["masks_sha256"]
    for path,digest in contract["immutable_input_sha256"].items():assert sha(path)==digest,path
    return contract,load(args.output/"prepared.npz"),load(args.output/"supervision_masks.npz")


def gpu_snapshot(contract):
    query=lambda *a:subprocess.check_output(["nvidia-smi",*a,"--format=csv,noheader,nounits"],text=True).strip()
    gpu=query("--query-gpu=memory.free,memory.used,utilization.gpu")
    apps=query("--query-compute-apps=pid,process_name,used_memory")
    unknown=[line for line in apps.splitlines() if line and int(line.split(",")[0]) not in contract["allowed_other_GPU_pids"]+[os.getpid()]]
    if int(gpu.split(",")[0])<contract["minimum_preflight_free_MiB"] or unknown:
        raise RuntimeError("GPU preflight refuses unknown task/insufficient spare memory: "+str((gpu,apps)))
    return dict(gpu=gpu,compute_processes=apps,own_pid=os.getpid())


def save_state(args,state):
    path=args.output/"states"/f"state_{len(state['completed']):03d}.npz";path.parent.mkdir(exist_ok=True)
    if path.exists():raise FileExistsError(path)
    np.savez_compressed(path,**state)
    restored=load(path);checks={k:ahash(restored[k])==ahash(v) for k,v in state.items()}
    assert all(checks.values());return path,checks


def accumulate(args):
    contract,prepared,masks=frozen(args);preflight=gpu_snapshot(contract)
    phase=args.phase;process_dir=args.output/(phase+"_process");process_dir.mkdir(exist_ok=False)
    import torch
    from PIL import Image
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.30)
    sys.path.insert(0,str(RUN/"source"));from corridor_runtime import CorridorRuntime
    start=time.time()
    if phase=="smoke":
        state=dict(normal=np.zeros((99,99),np.float64),rhs=np.zeros((99,3),np.float64),
            residual_squared_sum=np.array(0.,np.float64),pixels=np.array(0,np.int64),
            completed=np.array([],np.int16),frame_order=np.array(BUILD,np.int16),
            frame_pixels=np.zeros(40,np.int64),forwards=np.array(0,np.int64),
            contract_sha256=np.array(sha(args.output/"field_contract.json")),
            prepared_sha256=np.array(contract["prepared_sha256"]))
        raws=BUILD[:3]
    else:
        state=load(args.output/"states/state_003.npz")
        assert np.array_equal(state["completed"],BUILD[:3]) and int(state["forwards"])==30
        assert str(state["contract_sha256"])==sha(args.output/"field_contract.json")
        verification=read(args.output/"smoke_restore.json")
        assert verification["state_sha256"]==sha(args.output/"states/state_003.npz")
        assert all(verification["bitwise_restore_checks"].values());raws=BUILD[3:]
    records=[]
    with torch.no_grad():
        model=CorridorRuntime(args.asset);rt=model.rt
        assert np.array_equal(model.extension.ids,prepared["extension_ids"])
        assert np.array_equal(model.extension.active.cpu().numpy(),prepared["support_active"])
        mode_basis=torch.as_tensor(prepared["basis"],device="cuda")
        for raw in raws:
            data=rt.raw(raw);obs=data["obs"];domain=masks[f"raw{raw:03d}"]
            target=obs["rgb"].astype(np.float32).transpose(2,0,1)
            if target.max()>1:target/=255
            state_records=[]
            for edit_state in STATES:
                scene=model.scene(raw,delete=(edit_state=="delete"),extension=True,retire=False)
                original_features=scene["rows"]["features"]
                n=len(prepared["extension_ids"]);phis=[];first_rgb=None;first_w=None
                cross_batch_rgb=0.;cross_batch_w=0.;all_phi_sum=np.zeros((320,480),np.float64)
                for begin in range(0,99,MODE_BATCH):
                    end=min(99,begin+MODE_BATCH)
                    features=original_features.clone();features[:,2:]=0
                    features[-n:,2:2+end-begin]=mode_basis[:,begin:end]
                    features[-n:,23]=1
                    rows=dict(scene["rows"],features=features)
                    out=rt.raster(scene["cam"],rt.pipe,0,rows,scene["bg"])
                    rgb=(out["rgb"]+(1-out["alpha"])*scene["env"]).cpu().numpy()
                    w=out["extra"][21].cpu().numpy()
                    phi=out["extra"][:end-begin].cpu().numpy()
                    assert np.isfinite(phi).all() and np.isfinite(rgb).all()
                    if first_rgb is None:first_rgb=rgb;first_w=w
                    else:
                        cross_batch_rgb=max(cross_batch_rgb,float(abs(rgb-first_rgb).max()))
                        cross_batch_w=max(cross_batch_w,float(abs(w-first_w).max()))
                    phis.append(phi[:,domain].T.copy());all_phi_sum+=phi.sum(0,dtype=np.float64)
                    state["forwards"]+=1
                    write(process_dir/"forward_progress.json",dict(raw=raw,state=edit_state,mode_start=begin,
                        cumulative_forwards=int(state["forwards"]),completed_frames=state["completed"].tolist()))
                    del out,rows,features
                phi=np.concatenate(phis,1).astype(np.float64)
                conservation=float(abs(all_phi_sum-first_w).max())
                assert conservation<2e-5 and cross_batch_rgb==0 and cross_batch_w==0
                expected=args.reference_results/f"{raw:03d}/extension_{edit_state}.png"
                observed_png=np.asarray(Image.open(expected).convert("RGB"))
                encoded=np.uint8(np.clip(first_rgb.transpose(1,2,0),0,1)*255+.5)
                assert np.array_equal(encoded,observed_png),(raw,edit_state,"feature render changed baseline PNG")
                residual=(target[:,domain]-first_rgb[:,domain]).T.astype(np.float64)
                state["normal"]+=phi.T@phi;state["rhs"]+=phi.T@residual
                state["residual_squared_sum"]+=np.square(residual).sum();state["pixels"]+=int(domain.sum())
                state_records.append(dict(state=edit_state,pixels=int(domain.sum()),
                    conservation_maxabs=conservation,cross_batch_RGB_maxabs=cross_batch_rgb,
                    cross_batch_visibility_maxabs=cross_batch_w,saved_V2_PNG_exact=True))
                del scene,original_features,phi,phis
            state["frame_pixels"][BUILD.index(raw)]=2*int(domain.sum())
            state["completed"]=np.r_[state["completed"],np.int16(raw)]
            path,checks=save_state(args,state)
            record=dict(raw=raw,pixels=2*int(domain.sum()),states=state_records,state_path=str(path),state_sha256=sha(path),
                cumulative_forwards=int(state["forwards"]),cumulative_pixels=int(state["pixels"]))
            records.append(record);write(process_dir/f"{raw:03d}.json",record)
            print(json.dumps(record),flush=True)
            rt.assert_unchanged();assert model.extension.unchanged()
            del data;rt._data.clear();torch.cuda.empty_cache()
        runtime=model.receipt()
        if phase=="smoke":
            write(args.output/"smoke_restore.json",dict(status="PASS_REAL3_BUILD_SMOKE_SAVE_RESTORE",
                state_sha256=sha(path),bitwise_restore_checks=checks,raws=list(BUILD[:3]),
                forwarded=30,next_stage="resume fresh process on remaining37 BUILD",GPU_used=True))
        else:
            assert np.array_equal(state["completed"],BUILD) and int(state["forwards"])==400
            assert int(state["pixels"])==contract["total_supervision_pixels"]
        write(process_dir/"receipt.json",dict(status="PASS_"+phase.upper()+"_FEATURE_ACCUMULATION",phase=phase,
            total_completed=state["completed"].tolist(),forwards=int(state["forwards"]),records=records,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            torch_allocator_fraction=.30,preflight_gpu=preflight,runtime_freeze=runtime,
            neural_training_updates=0,geometry_updates=0,optimizer_updates=0,seconds=time.time()-start,shutdown_performed=False))
    frozen(args)


def solve(args):
    contract,prepared,_=frozen(args);state=load(args.output/"states/state_040.npz")
    assert np.array_equal(state["completed"],BUILD) and int(state["forwards"])==400
    a,y=system(state["normal"],state["rhs"],int(state["pixels"]))
    raw=np.linalg.solve(a,y)
    support=prepared["support_active"]
    coeff,active_colours,limits=constrain(raw,prepared["basis"][support],prepared["source_colours_srgb"][support].astype(float))
    colours=prepared["source_colours_srgb"].copy();colours[support]=active_colours
    assert np.array_equal(colours[~support],prepared["source_colours_srgb"][~support])
    def objective(c):
        sse=float(state["residual_squared_sum"]-2*np.sum(c*state["rhs"])+np.sum(c*(state["normal"]@c)))
        return dict(data=sse/(3*int(state["pixels"])),prior=PRIOR*float(np.square(c).mean()),
            smooth=SMOOTH*float(np.square(edge_matrix()@c).mean()))
    metadata=dict(schema="R0_REAL_BUILD40_MATCHED_STRIP_WORLD_MATERIAL_FIELD_V1",source_asset_sha256=V2_SHA,
        source_role="real original observed BUILD40 road shared GT for keep/delete equal weights; original V2 all199 proxy frozen; HELD9 excluded from residual solve",
        field_contract_sha256=sha(args.output/"field_contract.json"),state_sha256=sha(args.output/"states/state_040.npz"),
        constraints=limits,objective_before=objective(raw*0),objective_raw=objective(raw),objective_scaled=objective(coeff),
        geometry_updates=0,opacity_updates=0,native_updates=0,neural_training_updates=0,hidden_GT_used=False,
        regularizers=dict(prior=PRIOR,smooth=SMOOTH),supervision_pixels=int(state["pixels"]),forward_count=400,
        independent_road_prior_claim=False,heldout_generalization_claim=False)
    metadata["support"]=contract["support"]
    target=args.output/"material_override.npz"
    if target.exists():raise FileExistsError(target)
    np.savez_compressed(target,asset_sha256=np.array(V2_SHA),extension_ids=prepared["extension_ids"],
        colours_srgb=colours,basis=prepared["basis"],field_coefficients=coeff,raw_field_coefficients=raw,
        source_colours_srgb=prepared["source_colours_srgb"],local_xy_m=prepared["local_xy_m"],
        low_m=prepared["low_m"],high_m=prepared["high_m"],grid=prepared["grid"],support_active=support,
        strip_lateral_bounds_m=prepared["strip_lateral_bounds_m"],
        metadata_json=np.array(json.dumps(metadata,sort_keys=True)),frame_order=state["frame_order"])
    write(args.output/"solve_receipt.json",dict(metadata,material_path=str(target),material_sha256=sha(target),
        normal_equation_residual_maxabs=float(abs(a@raw-y).max()),CPU_only=True,optimizer_updates=0,shutdown_performed=False))
    print(json.dumps(dict(material=str(target),sha256=sha(target),constraints=limits,objectives=metadata)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("cpu-check","prepare","smoke","resume","solve"))
    p.add_argument("--asset",type=Path,default=RUN/"surface/frontfill_v2/road_reference_199.npz")
    p.add_argument("--output",type=Path,default=RUN/"surface/material_worldfield11x9_strip_v1")
    p.add_argument("--reference-results",type=Path,default=RUN/"results/full49_corridor")
    args=p.parse_args()
    if args.phase in ("cpu-check","prepare","solve"):os.environ["CUDA_VISIBLE_DEVICES"]=""
    {"cpu-check":cpu_check,"prepare":prepare,"smoke":accumulate,"resume":accumulate,"solve":solve}[args.phase](args)


if __name__=="__main__":main()

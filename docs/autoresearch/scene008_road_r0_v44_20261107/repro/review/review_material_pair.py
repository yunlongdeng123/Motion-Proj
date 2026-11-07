"""Independent CPU review of frozen-geometry material paired evaluations."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(4194304),b""):h.update(b)
    return h.hexdigest()
def count(x):return int(np.count_nonzero(x))
def load(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def pool(records,state,variant,metric,raws=None):
    v=[r["states"][state][variant][metric] for r in records if (raws is None or r["raw"] in raws) and metric in r["states"][state][variant]]
    n=sum(x["pixels"] for x in v);s=sum(x["sum"] for x in v)
    return dict(frames=len(v),pixels=n,sum=s,mean=s/n if n else None)
def summarize(error,mask):
    x=np.asarray(error,np.float64)[mask]
    return dict(pixels=len(x),sum=float(x.sum()),mean=float(x.mean()) if len(x) else None,
                p95=float(np.quantile(x,.95)) if len(x) else None,max=float(x.max()) if len(x) else None)


def audit_field(run):
    folder=run/"surface/material_worldfield11x9_v1"
    fc=read(folder/"field_contract.json");solve=read(folder/"solve_receipt.json")
    material=folder/"material_override.npz"
    expected="4a06539be18722f3fb3eddcfb7cf289d0d032e46878ca8d12d0b856a938e5f73"
    assert sha(material)==solve["material_sha256"]==expected
    a=load(material);prepared=load(folder/"prepared.npz");state=load(folder/"states/state_040.npz")
    masks=load(folder/"supervision_masks.npz")
    held={89,95,101,105,113,121,125,130,137};build=[r for r in range(89,138) if r not in held]
    assert fc["frame_order"]==build and fc["HELD9_excluded_from_field_fit"]==sorted(held)
    assert np.array_equal(state["completed"],build) and int(state["forwards"])==400
    assert np.array_equal(prepared["frame_order"],build)
    input_checks={path:sha(path)==h for path,h in fc["immutable_input_sha256"].items()}
    assert all(input_checks.values())
    assert sha(folder/"prepared.npz")==fc["prepared_sha256"]
    assert sha(folder/"supervision_masks.npz")==fc["masks_sha256"]
    mask_records=[];pcount=0
    for raw in build:
        obs=load(run.parent/f"scene008_instances_20261105_r1/observations/frames/{raw:03d}.npz")
        view=load(run.parent/f"scene008_blue_global_road1k_20261105_r1/inputs/views/{raw:03d}.npz")
        expected_mask=(obs["road"].astype(bool)&~obs["visible"].any(0)&~obs["visible_reviewed"].any(0)
                       &~obs["box_candidate"].any(0)&~view["other_vehicle_guard"])
        assert np.array_equal(masks[f"raw{raw:03d}"],expected_mask)
        assert count(expected_mask)==fc["per_frame_per_state_supervision_pixels"][str(raw)]
        pcount+=count(expected_mask);mask_records.append(dict(raw=raw,pixels=count(expected_mask),exact=True))
    assert 2*pcount==fc["total_supervision_pixels"]==int(state["pixels"])==1429092
    # Rebuild the documented ridge system independently from the accumulated
    # sufficient statistics, rather than invoking the fitter's solve helper.
    edges=[]
    for x in range(11):
        for y in range(9):
            for nx,ny in ((x+1,y),(x,y+1)):
                if nx<11 and ny<9:
                    row=np.zeros(99);row[x*9+y]=1;row[nx*9+ny]=-1;edges.append(row)
    D=np.asarray(edges);assert D.shape==(178,99)
    n=int(state["pixels"]);A=state["normal"]/n+(.01/99)*np.eye(99)+(.05/178)*(D.T@D);Y=state["rhs"]/n
    independently_solved=np.linalg.solve(A,Y)
    solve_diff=float(np.abs(independently_solved-a["raw_field_coefficients"]).max())
    assert solve_diff<1e-12
    assert np.allclose(a["field_coefficients"],independently_solved*solve["constraints"]["global_scale"],atol=1e-12,rtol=0)
    delta=a["basis"].astype(float)@a["field_coefficients"]
    derived=(a["source_colours_srgb"].astype(float)+delta).astype(np.float32)
    assert np.array_equal(derived,a["colours_srgb"])
    assert np.abs(delta).max()<=.15 and np.isfinite(derived).all() and derived.min()>=0 and derived.max()<=1
    with np.load(fc["asset"],allow_pickle=False) as asset:
        ids=np.flatnonzero(asset["active"]&(asset["local_xyz"][:,0]>19.75))
        assert np.array_equal(ids,a["extension_ids"])
        assert np.array_equal(asset["reference_srgb"][ids],a["source_colours_srgb"])
    def objective(c):
        sse=float(state["residual_squared_sum"]-2*np.sum(c*state["rhs"])+np.sum(c*(state["normal"]@c)))
        return dict(data=sse/(3*n),prior=.01*float(np.square(c).mean()),smooth=.05*float(np.square(D@c).mean()))
    obj0=objective(a["field_coefficients"]*0);obj1=objective(a["field_coefficients"])
    assert max(abs(obj1[k]-solve["objective_scaled"][k]) for k in obj1)<1e-12
    return dict(material_path=str(material),material_sha256=expected,asset_sha256=str(a["asset_sha256"]),field_contract_sha256=sha(folder/"field_contract.json"),
        immutable_file_hashes_all_match=True,immutable_files=len(input_checks),all40_masks_recomputed_exact=True,mask_checks=mask_records,
        BUILD40=build,HELD9=sorted(held),supervision_pixels=n,feature_forwards=400,field_scalar_parameters=297,
        independent_ridge_solution_max_difference=solve_diff,normal_equation_residual_maxabs=float(np.abs(A@independently_solved-Y).max()),
        coefficient_global_scale=solve["constraints"]["global_scale"],actual_max_colour_delta=float(np.abs(delta).max()),
        source_plus_basis_field_exact=True,source_colours_exactly_frozen_asset=True,objective_before=obj0,objective_after=obj1,
        field_shared_across_keep_delete=True,geometry_updates=0,opacity_updates=0,native_updates=0,
        supervision="Original RGB at legal observed-road BUILD40 masks. No road_edit_rgb, hidden deleted-actor target, generated image or native residual colour used as GT.",
        caveat="HELD9 excludes only this residual solve; original frozen V2 proxy already used FRONT199. This is not strict world-patch blinding or prior validation.")


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--results",required=True);args=p.parse_args()
    run=args.run;folder=run/"results"/args.results;out=run/"review";start=time.time()
    target=out/(args.results+"_review.json")
    if target.exists():raise FileExistsError("Preserve prior material review")
    receipt=read(folder/"receipt.json");raws=receipt["raws"];variants=receipt["variants"]
    corridor=bool(receipt.get("conservative_world_strip",False))
    assert variants==["baseline","extension","calibrated"]
    assert raws==([105,113,130,137] if receipt["mode"]=="pilot" else list(range(89,138)))
    records=[read(folder/f"{raw:03d}/metrics.json") for raw in raws]
    assert records==receipt["frames"]
    source_checks={name:sha(folder/"source_snapshot"/name)==h for name,h in receipt["executed_source_sha256"].items()}
    assert all(source_checks.values())
    if args.results=="full49_corridor_cellfit":
        from audit_cell_field_cpu import audit_cell_field
        field=audit_cell_field(run)
    else:field=audit_field(run)
    assert field["material_sha256"]==receipt["runtime"]["material_sha256"]
    assert field["asset_sha256"]==receipt["runtime"]["extension"]["sha256"]
    c=read(run/"contract.json");limits=c["preservation_gates"]
    allmetrics=("a6_RGB","a6_depth_m","observed_road_depth_m","observed_road_RGB","protected_RGB","a37_RGB")
    reported={(r["state"],r["variant"],r["metric"]):r for r in receipt["pooled"]}
    pooled=[];pooled_error=0.;tails=[];gates=[];pools={}
    fmax={"a6_RGB":limits["a6_RGB_MAE_max_frame_regression"],
          "a6_depth_m":limits["a6_measured_depth_MAE_max_frame_regression_m"],
          "a37_RGB":limits["a37_keep_RGB_MAE_max_frame_regression"],
          "observed_road_RGB":limits["original_observed_road_RGB_max_frame_regression"],
          "protected_RGB":limits["protected_RGB_max_frame_regression"]}
    for state in ("keep","delete"):
        for variant in variants:
            for metric in allmetrics:
                if metric=="a37_RGB" and state!="keep":continue
                a=pool(records,state,variant,metric);saved=reported[(state,variant,metric)]
                assert a["pixels"]==saved["pixels"]
                pooled_error=max(pooled_error,abs(a["mean"]-saved["mean"]))
                row=dict(state=state,variant=variant,metric=metric,**a);pooled.append(row);pools[(state,variant,metric)]=row
        for candidate in ("extension","calibrated"):
            for metric,limit in fmax.items():
                if metric=="a37_RGB" and state!="keep":continue
                frame_rows=[]
                for r in records:
                    b=r["states"][state]["baseline"][metric];e=r["states"][state][candidate][metric];u=r["states"][state]["extension"][metric]
                    assert b["pixels"]==e["pixels"]==u["pixels"]
                    frame_rows.append(dict(raw=r["raw"],pixels=b["pixels"],baseline=b["mean"],extension=u["mean"],candidate=e["mean"],
                        delta_vs_baseline=e["mean"]-b["mean"],delta_vs_extension=e["mean"]-u["mean"],
                        baseline_pixel_p95=b["p95"],candidate_pixel_p95=e["p95"]))
                ds=np.array([x["delta_vs_baseline"] for x in frame_rows]);bad=[x["raw"] for x in frame_rows if x["delta_vs_baseline"]>limit]
                gates.append(dict(state=state,candidate=candidate,metric=metric,kind="max_frame_regression",threshold=limit,observed=float(ds.max()),pass_gate=not bad,failed_raws=bad))
                pd=pools[(state,candidate,metric)]["mean"]-pools[(state,"baseline",metric)]["mean"]
                if metric in ("observed_road_RGB","protected_RGB"):
                    plimit=limits["original_observed_road_RGB_pooled_MAE_max_regression" if metric=="observed_road_RGB" else "protected_RGB_pooled_MAE_max_regression"]
                    gates.append(dict(state=state,candidate=candidate,metric=metric,kind="pooled_regression",threshold=plimit,observed=pd,pass_gate=pd<=plimit))
                tails.append(dict(state=state,candidate=candidate,metric=metric,pooled_delta_vs_baseline=pd,regressed_frames=count(ds>1e-10),
                    max_delta=float(ds.max()),median_delta=float(np.median(ds)),p95_frame_delta=float(np.quantile(ds,.95)),
                    worst_frames=sorted(frame_rows,key=lambda r:r["delta_vs_baseline"],reverse=True)[:8],all_frames=frame_rows))
    assert pooled_error<1e-12
    float_checks=[];metric_checks=[];png_checks=[];row_checks=[];allframe_invariants=[];full_hash_checks=[]
    obsdir=run.parent/"scene008_instances_20261105_r1/observations/frames"
    oldv44=run.parent/"scene008_v44_rgb_asset_20261107_r1/results/full49_r1/v44"
    oldv2=run/"results/full49_v2"
    invariants=("alpha","depth_m","group_alpha_T","extension_alpha_T")
    for r in records:
        raw=r["raw"];frames=r["states"]
        hashes=[frames[s][v]["native_asset_rows_hash"] for s in ("keep","delete") for v in variants]
        assert len(set(hashes))==1;row_checks.append(dict(raw=raw,all_six_hashes_equal=True,hash=hashes[0]))
        obs=load(obsdir/f"{raw:03d}.npz");gt=obs["rgb"].astype(np.float32).transpose(2,0,1)
        if gt.max()>1:gt/=255
        masks=dict(a6_RGB=obs["visible"][0].astype(bool),a6_depth_m=obs["depth_valid"][0].astype(bool),
            observed_road_depth_m=obs["road"].astype(bool)&(obs["scene_depth_m"]>0),observed_road_RGB=obs["road"].astype(bool),
            protected_RGB=obs["protect"].astype(bool)&~obs["visible"].any(0),a37_RGB=obs["visible"][1].astype(bool))
        for state in ("keep","delete"):
            bp=np.asarray(Image.open(folder/f"{raw:03d}/baseline_{state}.png"));op=np.asarray(Image.open(oldv44/f"{raw:03d}/{state}.png"))
            up=np.asarray(Image.open(folder/f"{raw:03d}/extension_{state}.png"));v2p=np.asarray(Image.open(oldv2/f"{raw:03d}/extension_{state}.png"))
            assert np.array_equal(bp,op)
            if not corridor:assert np.array_equal(up,v2p)
            unalpha=folder/f"{raw:03d}/extension_{state}_road_alphaT.png"
            caalpha=folder/f"{raw:03d}/calibrated_{state}_road_alphaT.png"
            assert sha(unalpha)==sha(caalpha)
            png_checks.append(dict(raw=raw,state=state,baseline_V44_exact=True,uncalibrated_v2_exact=bool(np.array_equal(up,v2p))))
            un_item=frames[state]["extension"];ca_item=frames[state]["calibrated"]
            fields=("alpha","road_alphaT","static_alphaT","asset_alphaT","a37_alphaT","environment_T","extension_alphaT","extension_visibility_ratio","road_only_depth_residual_m","extension_solo_alpha")
            domain_exact=all(un_item["domains"][domain][name]==ca_item["domains"][domain][name] for domain in un_item["domains"] for name in fields)
            depth_exact=all(un_item[name]==ca_item[name] for name in ("a6_depth_m","observed_road_depth_m","a6_alpha"))
            assert domain_exact and depth_exact
            allframe_invariants.append(dict(raw=raw,state=state,all_saved_domain_alpha_depth_summaries_exact=True,all_saved_depth_metrics_exact=True,
                road_alpha_PNG_SHA_exact=True,full_precision_maps_saved=raw in (105,113,130,137)))
            if "full_map_readout_sha256" in ca_item:
                assert ca_item["full_map_readout_sha256"]==un_item["full_map_readout_sha256"]
                assert ca_item["colour_only_full_maps_bitwise_equal_to_extension"]
                full_hash_checks.append(dict(raw=raw,state=state,all_four_full_map_hashes_equal=True,hashes=ca_item["full_map_readout_sha256"]))
            un=folder/f"{raw:03d}/extension_{state}_float.npz";ca=folder/f"{raw:03d}/calibrated_{state}_float.npz"
            if un.exists():
                u=load(un);a=load(ca);old=load(oldv2/f"{raw:03d}/extension_{state}_float.npz");b=load(folder/f"{raw:03d}/baseline_{state}_float.npz")
                oldb=load(oldv44/f"{raw:03d}/{state}_float.npz")
                check=dict(raw=raw,state=state,geometry_alpha_invariants={k:dict(exact=bool(np.array_equal(a[k],u[k])),maxabs=float(np.abs(a[k]-u[k]).max())) for k in invariants},
                    uncalibrated_v2_RGB_exact=bool(np.array_equal(u["rgb"],old["rgb"])),baseline_V44_RGB_exact=bool(np.array_equal(b["rgb"],oldb["rgb"])),
                    calibrated_RGB_max_change=float(np.abs(a["rgb"]-u["rgb"]).max()))
                assert all(x["exact"] for x in check["geometry_alpha_invariants"].values()) and check["baseline_V44_RGB_exact"]
                if not corridor:assert check["uncalibrated_v2_RGB_exact"]
                if "full_map_readout_sha256" in ca_item:
                    for variant,data in (("baseline",b),("extension",u),("calibrated",a)):
                        computed={name:hashlib.sha256(np.ascontiguousarray(data[name]).tobytes()).hexdigest() for name in invariants}
                        assert computed==frames[state][variant]["full_map_readout_sha256"]
                    check["saved_full_map_hashes_recomputed_exact"]=True
                float_checks.append(check)
                for variant,data in (("baseline",b),("extension",u),("calibrated",a)):
                    error=np.abs(data["rgb"]-gt).mean(0);maxdiff=0.
                    for metric in allmetrics:
                        if metric=="a37_RGB" and state!="keep":continue
                        val=np.abs(data["depth_m"]-obs["depth_m"][0]) if metric=="a6_depth_m" else np.abs(data["depth_m"]-obs["scene_depth_m"]) if metric=="observed_road_depth_m" else error
                        x=summarize(val,masks[metric]);saved=frames[state][variant][metric]
                        assert x["pixels"]==saved["pixels"]
                        maxdiff=max(maxdiff,max(abs(x[k]-saved[k]) for k in ("sum","mean","p95","max") if x[k] is not None))
                    assert maxdiff<1e-9;metric_checks.append(dict(raw=raw,state=state,variant=variant,max_abs_difference=maxdiff))
    split=[]
    for label,splitraws in (("BUILD40",set(field["BUILD40"])),("HELD9_field_excluded",set(field["HELD9"]))):
        for state in ("keep","delete"):
            for variant in variants:
                for metric in ("observed_road_RGB","protected_RGB","observed_road_depth_m"):
                    split.append(dict(split=label,state=state,variant=variant,metric=metric,**pool(records,state,variant,metric,splitraws)))
    parity=receipt["baseline_parity"];assert len(parity)==2*len(raws) and max(x["maxabs"] for x in parity)==0
    preserved=all(g["pass_gate"] for g in gates if g["candidate"]=="calibrated")
    corridor_check=None
    if corridor:
        ext=receipt["runtime"]["extension"];support=receipt["runtime"]["persistent_world_support_mask"]
        assert ext==support
        with np.load(field["material_path"],allow_pickle=False) as f:extension_ids=f["extension_ids"]
        with np.load(run/"surface/frontfill_v2/road_reference_199.npz",allow_pickle=False) as f:xy=f["local_xyz"][extension_ids,:2]
        low,high=ext["lateral_envelope_local_m"];strip=(xy[:,1]>=low)&(xy[:,1]<=high)
        assert count(strip)==ext["effective_active_rows"]==10098 and count(~strip)==ext["inactive_outside_strip_rows"]==4563
        assert hashlib.sha256(strip.tobytes()).hexdigest()==ext["support_mask_sha256"]
        corridor_check=dict(fixed_lateral_envelope_m=[low,high],selected_world_rows=len(strip),effective_active_rows=count(strip),inactive_outside_rows=count(~strip),
            support_mask_sha256=ext["support_mask_sha256"],independent_mask_recomputed_exact=True,
            material_sha256_unchanged=field["material_sha256"],material_refit_for_strip=False,
            scope_change="Persistent world active mask only; no image ROI, per-frame routing, geometry movement, sigma/opacity or material coefficient update.")
    failure_path=run/"results/pilot_worldfield/failure_receipt.json";failure=read(failure_path)
    failure_source_checks={name:sha(failure_path.parent/"source_snapshot"/name)==h for name,h in failure["source_snapshot_sha256"].items()}
    assert all(failure_source_checks.values())
    report=dict(schema="R0_MATERIAL_PAIRED_INDEPENDENT_CPU_REVIEW_V1",created_utc=datetime.now(timezone.utc).isoformat(),results=str(folder),mode=receipt["mode"],raws=raws,
        verdict="PASS_CALIBRATED_PRESERVATION" if preserved else "FAIL_CALIBRATED_PRESERVATION_DO_NOT_ADOPT",full49_confirmation_required=receipt["mode"]!="full49",
        input_receipt_sha256=sha(folder/"receipt.json"),source_snapshot_checks=source_checks,executed_source_sha256=receipt["executed_source_sha256"],
        field=field,pooled_sum_count_max_difference=pooled_error,pooled=pooled,tails=tails,preservation_gates=gates,split_pooled=split,
        geometry_alpha_invariance=dict(saved_pairs=len(float_checks),all_saved_pairs_bitwise=True,fields=list(invariants),checks=float_checks,
            all_frame_summary_and_road_alpha_PNG_checks=allframe_invariants,
            all_frame_full_map_hash_checks=full_hash_checks,
            unsaved_full49_scope="For non-keyframes full precision invariance cannot be independently reloaded; compare saved alpha/depth summaries/PNG and runtime declaration separately."),
        native_asset_rows=row_checks,PNG_parity=png_checks,available_float_metric_rechecks=metric_checks,
        conservative_world_strip=corridor,corridor_scope_check=corridor_check,
        baseline_runtime_parity=dict(records=len(parity),maxabs=max(x["maxabs"] for x in parity)),
        new_depth_metric=dict(name="observed_road_depth_m",reference="Original scene_depth_m: nearest actual projected LiDAR return per pixel; evaluated on obs road AND positive scene_depth_m.",
            render_semantics="Native Gaussian expected camera depth = sum_i(T_i*alpha_i*z_i)/sum_i(T_i*alpha_i), divided by meter_to_model. This is not first ray/surface intersection or nearest rendered surface depth.",
            comparison_caveat="Reference is a sparse first-return depth; mixed-layer expected depth can differ substantially. Report absolute residuals and sample counts; do not call this first-hit geometry accuracy.",
            gate="Descriptive metric newly introduced here, no retrospectively invented V1 preservation threshold; old receipts unchanged.",
            calibrated_minus_extension_pooled=[dict(state=s,delta=pools[(s,"calibrated","observed_road_depth_m")]["mean"]-pools[(s,"extension","observed_road_depth_m")]["mean"]) for s in ("keep","delete")]),
        original_failed_evaluation=dict(path=str(failure_path),sha256=sha(failure_path),receipt=failure,source_snapshot_checks=failure_source_checks,
            interpretation="Engineering branch failure before any calibrated forward;2 successful forwards are separate overhead, not a scientific material result."),
        interpretation="Calibration is a real-observed-road world material control with frozen geometry/opacity. Better than uncalibrated alone does not satisfy baseline preservation or prove hidden-road completion/prior ability.",
        hidden_GT_used=False,learned_prior=False,CPU_review_only=True,GPU_used=False,checkpoint_loads=0,seconds=time.time()-start)
    write(target,report)
    cellfit=field.get("audit_kind")=="cellfit1000"
    lines=["# 材质对照独立复核", "", f'{args.results}: **{report["verdict"]}**；{len(raws)}帧成对核查。', "",
        "![Architecture components](corridor_architecture.svg)","", ("固定几何/alpha，仅对2171个有真实renderer监督质量的new-road cell优化6513个RGB标量，恰1000次Adam；matched-strip 11×9 field初始化。" if cellfit else "固定几何/alpha，仅使用原始 BUILD40 可见道路 RGB 拟合11×9世界色场。")+" HELD9未参与材质拟合，但冻结V2proxy使用all199，不构成严格盲测。", "",
        "|状态|指标|baseline|extension|calibrated|", "|---|---|---:|---:|---:|"]
    for s in ("keep","delete"):
        for metric in ("observed_road_RGB","protected_RGB","observed_road_depth_m"):
            lines.append(f'|{s}|{metric}|'+"|".join(f'{pools[(s,v,metric)]["mean"]:.6f}' for v in variants)+"|")
    failed=[g for g in gates if g["candidate"]=="calibrated" and not g["pass_gate"]]
    lines += ["",f'calibrated 对原 V44 的13项保护门失败 {len(failed)} 项。相较extension改善不能代替全部保护门通过。', "",
        f'保存的{len(float_checks)}组 calibrated/extension 全图 alpha、depth、group-alpha、extension-alpha 逐位相同；原车辆行哈希逐帧六状态/变体恒等。baseline与原V44所有可用float RGB精确一致，PNG亦相同。'+(" Corridor只改变固定world active域，不要求与旧wide V2图像相等。" if corridor else " extension与旧V2所有可用float RGB和PNG精确一致。"), "",
        ("独立重算40份监督mask、全部输入SHA、sigmoid导出颜色与资格/冻结区；核1000步记录及smoke/restore独立预算，未重跑训练。" if cellfit else "独立重算40份监督mask与全部输入SHA、297参数ridge解及一次全局scale，结果一致。")+" material SHA `"+field["material_sha256"]+"`。新 observed-road depth 是原生 Gaussian 透射权重期望深度（ΣTαz/ΣTα）对稀疏真实LiDAR最近回波的残差，不是首交点精度；只作本轮实测描述，不写回旧收据或伪造原门限。全部帧的alpha/depth摘要及road-alpha PNG亦一致，未保存的非关键帧float不冒称独立逐像素重读。", "",
        "首跑因 baseline if/else 接线失败，在2次成功forward后终止，未产生 calibrated 分数；失败目录及源快照保留，新目录重跑。未改旧模型/几何/alpha，未使用隐藏GT或生成模型。", ""]
    if receipt["mode"]=="pilot":lines += ["本次仅pilot4，不能替代完整49帧验收。",""]
    if corridor:lines += [f'固定横向world mask独立重算通过：保留{count(strip)}行，inactive{count(~strip)}行。'+(" 本轮为固定1000步per-cell材质对照。" if cellfit else " 本轮仍用原wide field，未重新拟合。")+f'全49×keep/delete共{len(full_hash_checks)}组四张全图读出hash均相同；关键帧保存数组的hash已独立复算。',""]
    (out/(args.results+"_review.md")).write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(dict(verdict=report["verdict"],raws=raws,calibrated_failed_gates=failed,
        pooled=[r for r in pooled if r["metric"] in ("observed_road_RGB","protected_RGB","observed_road_depth_m")],
        alpha_depth_pairs=len(float_checks),field_ridge_diff=field["independent_ridge_solution_max_difference"],seconds=report["seconds"])),flush=True)


if __name__=="__main__":main()

"""Independent CPU acceptance audit of the uncalibrated R0 full49_v2 run.

Read-only inputs; writes only review/. No runtime, torch or checkpoint import.
"""
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
from scipy.ndimage import binary_dilation


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(4194304),b""):h.update(b)
    return h.hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
def count(x):return int(np.count_nonzero(x))


def summary(error,mask):
    a=np.asarray(error,np.float64)[mask]
    return dict(pixels=len(a),sum=float(a.sum()),mean=float(a.mean()) if len(a) else None,
                p95=float(np.quantile(a,.95)) if len(a) else None,max=float(a.max()) if len(a) else None)


def aggregate(records,state,variant,metric):
    values=[r["states"][state][variant][metric] for r in records if metric in r["states"][state][variant]]
    n=sum(v["pixels"] for v in values);total=sum(v["sum"] for v in values)
    return dict(frames=len(values),pixels=n,sum=total,mean=total/n if n else None)


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--refresh-own-review",action="store_true");args=p.parse_args()
    start=time.time();run=args.run;out=run/"review";out.mkdir(exist_ok=True)
    dest=out/"full49_v2_independent_review.json"
    if dest.exists() and not args.refresh_own_review:raise FileExistsError("Preserve prior independent review")
    result=run/"results/full49_v2";receipt=read(result/"receipt.json");contract=read(run/"contract.json")
    lineage=read(run/"contract_v2_lineage.json")
    base=run.parent;v44=base/"scene008_v44_rgb_asset_20261107_r1/results/full49_r1/v44"
    obsdir=base/"scene008_instances_20261105_r1/observations/frames"
    road=Path(receipt["runtime"]["V44"]["binding"]["road_run"])
    records=[read(result/f"{raw:03d}/metrics.json") for raw in range(89,138)]
    assert receipt["raws"]==list(range(89,138)) and receipt["variants"]==["baseline","extension"]
    assert len(receipt["frames"])==49
    file_receipt_exact=all(r==q for r,q in zip(records,receipt["frames"]))
    assert file_receipt_exact
    metrics=("a6_RGB","a6_depth_m","a37_RGB","observed_road_RGB","protected_RGB")
    pooled=[];reported={(r["state"],r["variant"],r["metric"]):r for r in receipt["pooled"]}
    pooled_max_error=0.
    for state in ("keep","delete"):
        for variant in ("baseline","extension"):
            for metric in metrics:
                if metric=="a37_RGB" and state!="keep":continue
                a=aggregate(records,state,variant,metric);r=reported[(state,variant,metric)]
                assert a["pixels"]==r["pixels"]
                pooled_max_error=max(pooled_max_error,abs(a["mean"]-r["mean"]))
                pooled.append(dict(state=state,variant=variant,metric=metric,**a))
    assert pooled_max_error<1e-12
    lookup={(r["state"],r["variant"],r["metric"]):r for r in pooled}
    thresholds=contract["preservation_gates"]
    frame_limits={"a6_RGB":thresholds["a6_RGB_MAE_max_frame_regression"],
                  "a6_depth_m":thresholds["a6_measured_depth_MAE_max_frame_regression_m"],
                  "a37_RGB":thresholds["a37_keep_RGB_MAE_max_frame_regression"],
                  "observed_road_RGB":thresholds["original_observed_road_RGB_max_frame_regression"],
                  "protected_RGB":thresholds["protected_RGB_max_frame_regression"]}
    pooled_limits={"observed_road_RGB":thresholds["original_observed_road_RGB_pooled_MAE_max_regression"],
                   "protected_RGB":thresholds["protected_RGB_pooled_MAE_max_regression"]}
    tails=[];gates=[]
    for state in ("keep","delete"):
        for metric in metrics:
            if metric=="a37_RGB" and state!="keep":continue
            rows=[]
            for r in records:
                b=r["states"][state]["baseline"][metric];e=r["states"][state]["extension"][metric]
                assert b["pixels"]==e["pixels"]
                if b["mean"] is None:continue
                rows.append(dict(raw=r["raw"],pixels=b["pixels"],baseline=b["mean"],extension=e["mean"],delta=e["mean"]-b["mean"],
                    baseline_pixel_p95=b["p95"],extension_pixel_p95=e["p95"],pixel_p95_delta=e["p95"]-b["p95"]))
            delta=np.array([r["delta"] for r in rows]);worst=sorted(rows,key=lambda r:r["delta"],reverse=True)
            bd=lookup[(state,"baseline",metric)];ed=lookup[(state,"extension",metric)]
            pdelta=ed["mean"]-bd["mean"]
            frame_fail=[r["raw"] for r in rows if r["delta"]>frame_limits[metric]]
            gates.append(dict(state=state,metric=metric,kind="max_frame_regression",threshold=frame_limits[metric],
                              observed=float(delta.max()),pass_gate=not frame_fail,failed_raws=frame_fail))
            if metric in pooled_limits:
                gates.append(dict(state=state,metric=metric,kind="pooled_regression",threshold=pooled_limits[metric],observed=pdelta,
                                  pass_gate=pdelta<=pooled_limits[metric]))
            tails.append(dict(state=state,metric=metric,pooled_pixels=bd["pixels"],baseline_pooled=bd["mean"],extension_pooled=ed["mean"],
                pooled_delta=pdelta,frame_count=len(rows),regressed_frames=count(delta>1e-10),improved_frames=count(delta< -1e-10),
                delta_min=float(delta.min()),delta_median=float(np.median(delta)),delta_p95=float(np.quantile(delta,.95)),delta_max=float(delta.max()),
                worst_pixel_p95_delta=max(r["pixel_p95_delta"] for r in rows),worst_frames=worst[:8],all_frame_deltas=rows))

    # Baseline parity: independent PNG and available float re-read, plus complete
    # original runtime assertion records. Do not claim unsaved float re-read.
    parity=receipt["baseline_parity"]
    assert len(parity)==98 and {(r["raw"],r["state"]) for r in parity}=={(raw,s) for raw in range(89,138) for s in ("keep","delete")}
    png_rows=[];float_rows=[];mask_rows=[];metric_rechecks=[];mechanism=[];change_rows=[]
    alpha_error=rgb_error=ratio_max=0.
    for r in records:
        raw=r["raw"];folder=result/f"{raw:03d}"
        with np.load(obsdir/f"{raw:03d}.npz",allow_pickle=False) as f:
            obs={k:f[k] for k in ("rgb","visible","visible_reviewed","box_candidate","depth_valid","depth_m","road","protect")}
        gt=obs["rgb"].astype(np.float32).transpose(2,0,1)
        if gt.max()>1:gt/=255
        masks=dict(a6_RGB=obs["visible"][0].astype(bool),a37_RGB=obs["visible"][1].astype(bool),
                   a6_depth_m=obs["depth_valid"][0].astype(bool),observed_road_RGB=obs["road"].astype(bool),
                   protected_RGB=obs["protect"].astype(bool)&~obs["visible"].any(0))
        outside=obs["protect"].astype(bool)&~obs["visible_reviewed"].any(0)&~obs["box_candidate"].any(0)
        mask_rows.append(dict(raw=raw,counts={k:count(v) for k,v in masks.items()},
                              reviewed_outside_pixels=count(outside),protected_in_a6_reviewed_border=count(masks["protected_RGB"]&obs["visible_reviewed"][0])))
        with np.load(road/f"inputs/views/{raw:03d}.npz",allow_pickle=False) as f:guard=f["other_vehicle_guard"]
        allowed=~binary_dilation(obs["box_candidate"][0],iterations=2)&~guard
        solo=np.asarray(Image.open(folder/"extension_solo_alpha.png"))[:,:,0]
        certain=count((solo>=251)&allowed);possible=count((solo>=250)&allowed)
        declared=r["mechanism_domain_pixels"]
        assert certain<=declared<=possible
        mechanism.append(dict(raw=raw,declared_pixels=declared,PNG_threshold_lower_count=certain,PNG_threshold_upper_count=possible,
                              exact_solo_alpha_saved=False))
        for state in ("keep","delete"):
            baseline_png=np.asarray(Image.open(folder/f"baseline_{state}.png"));original_png=np.asarray(Image.open(v44/f"{raw:03d}/{state}.png"))
            png_rows.append(dict(raw=raw,state=state,decoded_equal=bool(np.array_equal(baseline_png,original_png)),
                file_sha_equal=sha(folder/f"baseline_{state}.png")==sha(v44/f"{raw:03d}/{state}.png")))
            local_float=folder/f"baseline_{state}_float.npz"
            if local_float.exists():
                with np.load(local_float,allow_pickle=False) as a,np.load(v44/f"{raw:03d}/{state}_float.npz",allow_pickle=False) as b:
                    float_rows.append(dict(raw=raw,state=state,fields={k:dict(exact=bool(np.array_equal(a[k],b[k])),
                        maxabs=float(np.abs(a[k]-b[k]).max())) for k in ("rgb","alpha","depth_m","group_alpha_T")}))
            for variant in ("baseline","extension"):
                item=r["states"][state][variant]
                alpha_error=max(alpha_error,item["alpha_conservation_maxabs"]);rgb_error=max(rgb_error,item["RGB_conservation_maxabs"])
                for metric in metrics:
                    if metric in item:assert item[metric]["pixels"]==count(masks[metric])
                for label,d in item["domains"].items():
                    expected=720 if label=="fixed720" else declared if label=="measured_extension_rays" else (1600 if raw==137 else 4992 if raw==130 else 0)
                    assert all(s["pixels"]==expected for s in d.values())
                    if d["extension_visibility_ratio"]["max"] is not None:ratio_max=max(ratio_max,d["extension_visibility_ratio"]["max"])
                fp=folder/f"{variant}_{state}_float.npz"
                if fp.exists():
                    with np.load(fp,allow_pickle=False) as a:
                        error=np.abs(a["rgb"]-gt).mean(0)
                        max_delta=0.
                        for metric in metrics:
                            if metric not in item:continue
                            values=np.abs(a["depth_m"]-obs["depth_m"][0]) if metric=="a6_depth_m" else error
                            check=summary(values,masks[metric]);saved=item[metric]
                            assert check["pixels"]==saved["pixels"]
                            max_delta=max(max_delta,max(abs(check[k]-saved[k]) for k in ("sum","mean","p95","max") if check[k] is not None))
                        metric_rechecks.append(dict(raw=raw,state=state,variant=variant,max_abs_metric_difference=max_delta))
                        if variant=="extension":
                            with np.load(folder/f"baseline_{state}_float.npz",allow_pickle=False) as b:
                                diff=np.abs(a["rgb"]-b["rgb"])
                                change_rows.append(dict(raw=raw,state=state,reviewed_outside_pixels=count(outside),
                                    RGB_mean_change=float(diff.mean(0)[outside].mean()),RGB_max_change=float(diff[:,outside].max()),
                                    RGB_max_channel_gt002=count(diff.max(0)[outside]>.02)))
    assert all(r["decoded_equal"] for r in png_rows)
    assert all(r["fields"]["rgb"]["exact"] for r in float_rows), float_rows
    assert max(r["max_abs_metric_difference"] for r in metric_rechecks)<1e-9
    domain_pooled=[]
    for state in ("keep","delete"):
        for variant in ("baseline","extension"):
            for label in ("fixed720","silver_fixed","measured_extension_rays"):
                for metric in ("road_alphaT","extension_alphaT","environment_T","extension_visibility_ratio","road_only_depth_residual_m"):
                    vals=[r["states"][state][variant]["domains"][label][metric] for r in records]
                    n=sum(v["pixels"] for v in vals);total=sum(v["sum"] for v in vals)
                    domain_pooled.append(dict(state=state,variant=variant,domain=label,metric=metric,pixels=n,mean=total/n if n else None))
    allpass=all(g["pass_gate"] for g in gates)
    report=dict(schema="R0_FULL49_V2_INDEPENDENT_CPU_REVIEW_V1",created_utc=datetime.now(timezone.utc).isoformat(),
        verdict="PASS_PRESERVATION" if allpass else "FAIL_PRESERVATION_DO_NOT_ADOPT",CPU_only=True,
        input_receipt=dict(path=str(result/"receipt.json"),sha256=sha(result/"receipt.json")),
        contract=dict(path=str(run/"contract.json"),sha256=sha(run/"contract.json"),frozen_thresholds=thresholds,
            declared_asset_sha256=contract["measured_surface_sha256"],actual_asset_sha256=receipt["runtime"]["extension"]["sha256"],
            asset_sha_matches=False,v2_lineage_path=str(run/"contract_v2_lineage.json"),v2_lineage_sha256=sha(run/"contract_v2_lineage.json"),
            lineage_timing=lineage["timing"],interpretation="Asset identity differs transparently; original thresholds reused. The later lineage note is not a preregistration."),
        independent_checks=dict(all49_metric_files_match_receipt=file_receipt_exact,pooled_sum_count_max_difference=pooled_max_error,
            all49_observation_mask_counts_match=True,all49_mechanism_denominators_within_saved_PNG_bounds=True,
            available_float_metric_rechecks=metric_rechecks,mask_counts=mask_rows,
            alpha_conservation_maxabs=alpha_error,RGB_conservation_maxabs=rgb_error,
            max_reported_extension_W_over_solo=ratio_max),
        baseline_parity=dict(runtime_records=len(parity),runtime_maxabs=max(r["maxabs"] for r in parity),
            independent_PNG_checks=len(png_rows),all_decoded_PNGs_exact=True,all_PNG_file_SHA_equal=all(r["file_sha_equal"] for r in png_rows),
            independent_float_outputs_reloaded=len(float_rows),float_checks=float_rows,
            saved_RGB_alpha_group_bitwise_exact=all(all(r["fields"][k]["exact"] for k in ("rgb","alpha","group_alpha_T")) for r in float_rows),
            saved_depth_bitwise_exact=all(r["fields"]["depth_m"]["exact"] for r in float_rows),
            saved_depth_maxabs_m=max(r["fields"]["depth_m"]["maxabs"] for r in float_rows),
            depth_note="R0 normalizes depth on Torch/GPU; old V44 normalized saved depth_sum/alpha in NumPy/CPU. RGB/alpha/group are exact; depth has <=7.63e-6m arithmetic rounding and is not bitwise equal.",
            limitation="Only8 full-precision baseline outputs were saved in this run; remaining90 cannot be independently float-reloaded. All98 have runtime assert<2e-6 records and exact saved PNG parity; no GPU replay performed."),
        pooled=pooled,per_frame_tails=tails,preservation_gates=gates,all_preservation_gates_pass=allpass,
        mechanism_domain=dict(rule="solo alpha>=0.98, excluding a6 box_candidate dilated2px and other_vehicle_guard; same per-frame mask across baseline/extension and keep/delete",
            derived_from="measured asset geometry rendered alone, not accuracy/hidden-GT or independently verified unoccluded pixels",
            confidence="World asset centers have >=3 stable real RGB sources and LiDAR-pool support. Per-pixel measured depth/source agreement is not saved in this domain.",
            exact_denominator_saved_in_metrics=True,PNG_only_threshold_reconstruction_is_interval=True,per_frame=mechanism,
            pooled=domain_pooled,warning="Larger ROI road alpha*T only demonstrates a mechanism. It cannot offset failed observed-road/foreground/protection accuracy gates."),
        reviewed_outside_keyframe_changes=change_rows,
        observed_road_reference="Real same-frame observed road RGB. Delete images have no independent hidden clean-road GT under the removed actor.",
        tail_semantics="Frame regression quantiles and worst per-frame pixel-p95 deltas, not a reconstructed pooled pixel-p95.",
        material_field_executed_in_this_run=False,variants=receipt["variants"],learned_prior=False,optimizer_updates=0,
        source_code_note="Current root sources gained optional material/depth instrumentation after this run; review uses immutable saved metric fields, without backfilling new fields into the old receipt.",
        source_projection_audit=str(run/"surface/frontfill_v2/projection_alignment.json"),
        GPU_used=False,checkpoint_loads=0,seconds=time.time()-start)
    write(dest,report)
    lines=["# R0 full49 V2 独立复核", "", "V2 未通过保护门，不采用；当前仅有未标定的 baseline/extension 对照。", "",
           "![Architecture components](../surface/architecture.svg)", "",
           "逐帧49份 metrics 与 receipt 一致；重新由 sum/count 汇总的 pooled 数值精确一致。以下为 sRGB MAE（depth 单独按米审查）：", "",
           "|状态|指标|V44 baseline|V2 extension|变化|", "|---|---|---:|---:|---:|"]
    for r in tails:
        if r["metric"]=="a6_depth_m":continue
        lines.append(f'|{r["state"]}|{r["metric"]}|{r["baseline_pooled"]:.6f}|{r["extension_pooled"]:.6f}|{r["pooled_delta"]:+.6f}|')
    lines += ["",f'保护门共 {len(gates)} 项，失败 {sum(not g["pass_gate"] for g in gates)} 项。最差逐帧退化：',""]
    for r in tails:
        if r["metric"] in ("observed_road_RGB","protected_RGB","a6_RGB","a37_RGB"):
            w=r["worst_frames"][0]
            lines.append(f'- {r["state"]} {r["metric"]}: raw{w["raw"]} {w["delta"]:+.6f}；{r["regressed_frames"]}/{r["frame_count"]} 帧变差。')
    lines += ["",f'98 次运行时 RGB baseline parity 的最大差为 {report["baseline_parity"]["runtime_maxabs"]:.3g}。独立重读全部98张 baseline PNG（与原V44文件SHA和像素均精确一致），以及本轮保存的8个 full-precision baseline：RGB/alpha/group 位等；depth 因 Torch 与 NumPy 端除法舍入，最大差 7.63e-6 米，不称位等。其余90个本轮 float 未保存，未冒称独立重读。另对16个可用 baseline/extension float 重算原始观测误差，全部一致。',
              "", "机制域分母是 extension solo alpha≥0.98，扣除 a6 膨胀候选框和其他车辆 guard；这是由模型几何得到的机制域，未逐像素保存独立实测深度及 source agreement。固定720 ROI 的贡献增加不作为准确率通过证据，隐藏删后地面仍没有独立 GT。", "",
              "原冻结合同仍指向 V1 资产 SHA；V2 是新增7313候选的显式谱系变体。contract_v2_lineage 在 full49 之后记录，未冒称预注册；保护阈值沿用原合同。颜色、法线和三套投影另经 CPU 审核，无编码或坐标接线错误。材质场对照尚未包括在本次审查。", ""]
    (out/"full49_v2_review.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(dict(verdict=report["verdict"],gates=len(gates),failed=sum(not g["pass_gate"] for g in gates),
        pooled=pooled,gate_results=gates,parity=report["baseline_parity"],seconds=report["seconds"])),flush=True)


if __name__=="__main__":main()

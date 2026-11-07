"""CPU diagnostic of asset contribution in unobserved a6 candidate pixels.

No hidden GT or claim that the old inferred surface is physically correct.
Frozen asset rows are distinct from invariant transmittance-weighted visibility.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
import argparse
import json
from pathlib import Path
import numpy as np


def load(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def count(x):return int(np.count_nonzero(x))
def stats(x):
    return dict(points=len(x),mean=float(np.mean(x)) if len(x) else None,p95=float(np.quantile(x,.95)) if len(x) else None,max=float(np.max(x)) if len(x) else None)


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--results",default="full49_corridor");args=p.parse_args()
    run=args.run;result=run/"results"/args.results;target=run/"review"/(args.results+"_inferred_asset_contribution.json")
    if target.exists():raise FileExistsError("Preserve prior unknown-contribution audit")
    rows=[];examples=[];maps={}
    for raw in (105,113,130,137):
        obs=load(run.parent/f"scene008_instances_20261105_r1/observations/frames/{raw:03d}.npz")
        b=load(result/f"{raw:03d}/baseline_delete_float.npz");a=load(result/f"{raw:03d}/calibrated_delete_float.npz")
        candidate=obs["box_candidate"][0].astype(bool);visible=obs["visible"][0].astype(bool);reviewed=obs["visible_reviewed"][0].astype(bool)
        domain=candidate&~visible;baseW=b["group_alpha_T"][0];newW=a["group_alpha_T"][0]
        loss=(baseW-newW).astype(float);loss_positive=np.maximum(loss,0)
        colour=np.abs(a["rgb"]-b["rgb"]);rgbmean=colour.mean(0);rgbmax=colour.max(0)
        subdomains=dict(all_unobserved_candidate=domain,reviewed_margin=domain&reviewed,amodal_outside_reviewed=domain&~reviewed)
        for label,mask0 in subdomains.items():
            for threshold in (0.,.5,.9):
                mask=mask0&(baseW>threshold)
                rec=dict(raw=raw,state="delete",domain=label,baseline_assetW_threshold=threshold,pixels=count(mask),
                    baseline_assetW=stats(baseW[mask]),new_assetW=stats(newW[mask]),
                    asset_alphaT_loss=stats(loss_positive[mask]),asset_alphaT_signed_loss=stats(loss[mask]),
                    loss_sum=float(loss_positive[mask].sum()),counts_loss_gt={str(t):count(mask&(loss>t)) for t in (1e-6,.001,.01,.05,.1,.25,.5)},
                    loss_gt001_and_RGB_max_channel_gt002=count(mask&(loss>.01)&(rgbmax>.02)),
                    RGB_mean_absolute_change=stats(rgbmean[mask]),RGB_max_channel_change=stats(rgbmax[mask]),
                    RGB_changed_gt002=count(mask&(rgbmax>.02)),RGB_changed_gt005=count(mask&(rgbmax>.05)),
                    road_alphaT_gain=stats((a["group_alpha_T"][2]-b["group_alpha_T"][2])[mask]),
                    new_extension_alphaT=stats(a["extension_alpha_T"][mask]),
                    expected_depth_change_m=stats(np.abs(a["depth_m"]-b["depth_m"])[mask]))
                rows.append(rec)
        ids=np.flatnonzero(domain&(baseW>.5));rank=ids[np.argsort(-loss.ravel()[ids],kind="stable")[:12]]
        for idx in rank:
            y,x=divmod(int(idx),480)
            examples.append(dict(raw=raw,pixel_xy=[x,y],within_reviewed=bool(reviewed[y,x]),
                baseline_assetW=float(baseW[y,x]),new_assetW=float(newW[y,x]),assetW_loss=float(loss[y,x]),
                baseline_roadW=float(b["group_alpha_T"][2,y,x]),new_roadW=float(a["group_alpha_T"][2,y,x]),
                new_extensionW=float(a["extension_alpha_T"][y,x]),baseline_RGB=b["rgb"][:,y,x].tolist(),new_RGB=a["rgb"][:,y,x].tolist(),
                RGB_max_channel_change=float(rgbmax[y,x]),baseline_expected_depth_m=float(b["depth_m"][y,x]),new_expected_depth_m=float(a["depth_m"][y,x]),
                measured_scene_first_return_m=float(obs["scene_depth_m"][y,x])))
        maps.update({f"raw{raw}_domain":domain,f"raw{raw}_baseline_assetW":baseW,f"raw{raw}_new_assetW":newW,
                     f"raw{raw}_assetW_loss":loss.astype(np.float32),f"raw{raw}_RGB_max_change":rgbmax})
    totals=[]
    for threshold in (.5,.9):
        chosen=[r for r in rows if r["domain"]=="all_unobserved_candidate" and r["baseline_assetW_threshold"]==threshold]
        n=sum(r["pixels"] for r in chosen)
        totals.append(dict(baseline_assetW_threshold=threshold,pixels=n,counts_loss_gt={str(t):sum(r["counts_loss_gt"][str(t)] for r in chosen) for t in (1e-6,.001,.01,.05,.1,.25,.5)},
            max_assetW_loss=max(r["asset_alphaT_loss"]["max"] or 0 for r in chosen),
            total_assetW_loss=sum(r["loss_sum"] for r in chosen),RGB_changed_gt002=sum(r["RGB_changed_gt002"] for r in chosen),
            RGB_changed_gt005=sum(r["RGB_changed_gt005"] for r in chosen),
            max_RGB_channel_change=max(r["RGB_max_channel_change"]["max"] or 0 for r in chosen)))
    report=dict(schema="R0_UNOBSERVED_A6_ASSET_CONTRIBUTION_DIAGNOSTIC_V1",results=str(result),raws=[105,113,130,137],state="delete",
        diagnostic_domain="Original a6 box_candidate AND NOT original a6 visible; stratified by frozen baseline actual composited assetW>0.5 and >0.9, with reviewed-margin/outside-reviewed subdomains.",
        totals=totals,per_frame=rows,examples_largest_asset_losses=examples,
        interpretation="Asset tensors/row hashes frozen does not imply asset alpha*T invariant. Adding road can change shared-sort transmittance/ownership in inferred body pixels.",
        no_hidden_GT=True,no_accuracy_verdict=True,
        limits=["The old inferred PVD geometry and its unobserved pixels are not accepted hidden ground truth.",
                "A road taking weight in front is a mechanism observation; this diagnostic alone cannot label the old asset or road physically correct.",
                "Expected depth is composited Gaussian expectation, not first ray intersection. Sparse scene first-return values are context, not hidden-body labels."],
        GPU_used=False,optimizer_updates=0)
    target.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    np.savez_compressed(target.with_suffix(".npz"),**maps)
    lines=["# 未观测车体贡献诊断", "", "![Architecture components](corridor_architecture.svg)", "",
        "仅诊断a6 candidate且非visible像素；不作为隐藏车体GT，不默认unknown是road。冻结资产行与冻结实际贡献是不同条件。", "",
        "|raw|baseline assetW阈值|像素|assetW损失>.01|损失>.1|最大损失|RGB最大通道变化|", "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["domain"]!="all_unobserved_candidate" or r["baseline_assetW_threshold"] not in (.5,.9):continue
        lines.append(f'|{r["raw"]}|{r["baseline_assetW_threshold"]}|{r["pixels"]}|{r["counts_loss_gt"]["0.01"]}|{r["counts_loss_gt"]["0.1"]}|{r["asset_alphaT_loss"]["max"] or 0:.6f}|{r["RGB_max_channel_change"]["max"] or 0:.6f}|')
    lines += ["", "如road在实际深度排序中取得权重，只报告这一机制。没有证据可将旧PVD隐藏几何视为正确答案，也不能把已有贡献下降自动判为补全错误。全部像素例及loss map另存JSON/NPZ。", ""]
    target.with_suffix(".md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(dict(totals=totals,per_frame=[r for r in rows if r["domain"]=="all_unobserved_candidate" and r["baseline_assetW_threshold"] in (.5,.9)])),flush=True)


if __name__=="__main__":main()

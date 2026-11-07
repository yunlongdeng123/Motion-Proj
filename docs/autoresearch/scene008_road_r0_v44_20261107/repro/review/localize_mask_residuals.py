"""Locate R0 appearance regression using fixed observation/fit-mask partitions.

Four keyframes use saved float output. Full49 is a separately labelled PNG
proxy; its quantized error is never substituted for exact acceptance metrics.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
import argparse
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image


def load(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def count(x):return int(np.count_nonzero(x))


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--results",default="full49_worldfield");args=p.parse_args()
    run=args.run;result=run/"results"/args.results;out=run/"review";started=time.time()
    target=out/(args.results+"_mask_localization.json")
    if target.exists():raise FileExistsError("Preserve prior mask localization")
    receipt=json.loads((result/"receipt.json").read_text());variants=("baseline","extension","calibrated")
    fc=json.loads((run/"surface/material_worldfield11x9_v1/field_contract.json").read_text())
    actual_masks=load(run/"surface/material_worldfield11x9_v1/supervision_masks.npz")
    accum={};perframe=[]
    groups_order=("road_total","road_fit_allowed","road_amodal_excluded","road_other_excluded",
                  "road_amodal_a6_only","road_amodal_a37_only","road_amodal_both",
                  "protected_total","protected_fit_allowed_road","protected_amodal_excluded_road","protected_other_road","protected_nonroad")
    for raw in receipt["raws"]:
        obs=load(run.parent/f"scene008_instances_20261105_r1/observations/frames/{raw:03d}.npz")
        view=load(run.parent/f"scene008_blue_global_road1k_20261105_r1/inputs/views/{raw:03d}.npz")
        road=obs["road"].astype(bool);protected=obs["protect"].astype(bool)&~obs["visible"].any(0)
        allowed_except_box=road&~obs["visible"].any(0)&~obs["visible_reviewed"].any(0)&~view["other_vehicle_guard"]
        bx=obs["box_candidate"].astype(bool);amodal=allowed_except_box&bx.any(0);allowed=allowed_except_box&~bx.any(0);other=road&~allowed_except_box
        if raw in fc["frame_order"]:assert np.array_equal(allowed,actual_masks[f"raw{raw:03d}"])
        masks=dict(road_total=road,road_fit_allowed=allowed,road_amodal_excluded=amodal,road_other_excluded=other,
            road_amodal_a6_only=amodal&bx[0]&~bx[1],road_amodal_a37_only=amodal&bx[1]&~bx[0],road_amodal_both=amodal&bx.all(0),
            protected_total=protected,protected_fit_allowed_road=protected&allowed,protected_amodal_excluded_road=protected&amodal,
            protected_other_road=protected&other,protected_nonroad=protected&~road)
        assert count(road)==sum(count(masks[x]) for x in ("road_fit_allowed","road_amodal_excluded","road_other_excluded"))
        assert count(protected)==sum(count(masks[x]) for x in ("protected_fit_allowed_road","protected_amodal_excluded_road","protected_other_road","protected_nonroad"))
        gt=obs["rgb"].astype(np.float32).transpose(2,0,1)
        if gt.max()>1:gt/=255
        for state in ("keep","delete"):
            sources={"full49_PNG_proxy":{v:np.asarray(Image.open(result/f"{raw:03d}/{v}_{state}.png"),float).transpose(2,0,1)/255 for v in variants}}
            if (result/f"{raw:03d}/calibrated_{state}_float.npz").exists():
                sources["key4_float_exact"]={v:load(result/f"{raw:03d}/{v}_{state}_float.npz")["rgb"] for v in variants}
            for scope,rgb in sources.items():
                error={v:np.abs(rgb[v]-gt).mean(0) for v in variants}
                d=error["calibrated"]-error["baseline"]
                frame=dict(raw=raw,state=state,scope=scope,groups={})
                for group in groups_order:
                    mask=masks[group];n=count(mask);sums={v:float(np.asarray(error[v],float)[mask].sum()) for v in variants}
                    rec=dict(pixels=n,error_sum=sums,calibrated_minus_baseline_sum=sums["calibrated"]-sums["baseline"],
                        positive_excess_sum=float(np.maximum(d[mask],0).sum()),improvement_sum=float(np.maximum(-d[mask],0).sum()),
                        changed_gt002=count(np.abs(rgb["calibrated"]-rgb["baseline"]).max(0)[mask]>.02))
                    frame["groups"][group]=rec
                    key=(scope,state,group)
                    if key not in accum:accum[key]=dict(frames=0,pixels=0,error_sum={v:0. for v in variants},positive_excess_sum=0.,improvement_sum=0.,changed_gt002=0)
                    a=accum[key];a["frames"]+=1;a["pixels"]+=n
                    for v in variants:a["error_sum"][v]+=sums[v]
                    for k in ("positive_excess_sum","improvement_sum","changed_gt002"):a[k]+=rec[k]
                perframe.append(frame)
    aggregate=[]
    for (scope,state,group),a in accum.items():
        a["MAE"]={v:a["error_sum"][v]/a["pixels"] if a["pixels"] else None for v in variants}
        a["net_excess_sum"]=a["error_sum"]["calibrated"]-a["error_sum"]["baseline"]
        a["net_delta_MAE"]=a["net_excess_sum"]/a["pixels"] if a["pixels"] else None
        aggregate.append(dict(scope=scope,state=state,group=group,**a))
    concentration=[]
    for scope in ("full49_PNG_proxy","key4_float_exact"):
        for state in ("keep","delete"):
            for parent,children in (("road_total",("road_fit_allowed","road_amodal_excluded","road_other_excluded")),
                ("protected_total",("protected_fit_allowed_road","protected_amodal_excluded_road","protected_other_road","protected_nonroad"))):
                total=accum[(scope,state,parent)]
                for group in children:
                    a=accum[(scope,state,group)]
                    concentration.append(dict(scope=scope,state=state,parent=parent,group=group,pixels=a["pixels"],
                        pixel_fraction=a["pixels"]/total["pixels"] if total["pixels"] else None,
                        fraction_of_positive_excess=a["positive_excess_sum"]/total["positive_excess_sum"] if total["positive_excess_sum"] else None,
                        fraction_of_net_excess=a["net_excess_sum"]/total["net_excess_sum"] if total["net_excess_sum"] else None,
                        calibrated_minus_baseline_MAE=a["net_delta_MAE"]))
    report=dict(schema="R0_FIXED_MASK_RESIDUAL_LOCALIZATION_V1",results=str(result),CPU_only=True,
        group_rule="Original obs road partition: fit mask; otherwise eligible road excluded only by conservative a6/a37 amodal box_candidate; other exclusions. Protected mask partitioned by the same original observations.",
        actual_BUILD40_masks_rechecked=True,HELD9_partition_rule="Same fixed observation-mask rule, without fitting those frames.",
        scope_rules=dict(key4_float_exact="Exact saved float at raws105/113/130/137; original GT float32; no output-dependent mask selection.",
            full49_PNG_proxy="All49 quantized/clipped saved PNGs, not replacement acceptance metrics. Each unclipped pixel RGB error can shift by <=0.5/255 per render; delta by <=1/255. Clipped pixels can have larger proxy error."),
        aggregates=aggregate,concentration=concentration,per_frame=perframe,
        causal_limit="Exclusion identifies omitted pixel supervision, not proof a world cell was never supervised in another view. This localization alone cannot isolate field capacity from masks/visibility/regularization.",
        prospective_percell_note="Compare actual percell masks when prepared; this audit refers to exact worldfield masks, matching the stated conservative amodal exclusion rule.",
        GPU_used=False,optimizer_updates=0,seconds=time.time()-started)
    target.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    lines=["# 固定监督mask残差定位", "", "同一原始观测分区，未按预测颜色或误差挑像素。关键4帧用完整float，全49另用PNG代理，后者不替代精确保护门。", "",
        "|范围|状态|区域|像素占比|正退化占比|净退化占比|", "|---|---|---|---:|---:|---:|"]
    for r in concentration:
        lines.append(f'|{r["scope"]}|{r["state"]}|{r["group"]}|{r["pixel_fraction"]:.3%}|{r["fraction_of_positive_excess"]:.3%}|{r["fraction_of_net_excess"]:.3%}|')
    lines += ["", "分区能发现保守amodal排除的像素监督缺口，但不能证明相应世界单元在其他视角没有监督；不能据此只归因于色场容量。所有数值完整见同名JSON。", ""]
    (out/(args.results+"_mask_localization.md")).write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(concentration),flush=True)


if __name__=="__main__":main()

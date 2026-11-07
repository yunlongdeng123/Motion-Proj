"""Final CPU cross-control checks, coverage counts and bounded risk handoff."""
import hashlib
import json
from pathlib import Path
import sys


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
run=Path(sys.argv[1]);review=run/"review";result=run/"results/full49_corridor_cellfit"
path=review/"full49_corridor_cellfit_review.json";report=read(path);receipt=read(result/"receipt.json")
checks=[]
for raw in range(89,138):
    cur=read(result/f"{raw:03d}/metrics.json");old=read(run/f"results/full49_corridor/{raw:03d}/metrics.json")
    for state in ("keep","delete"):
        ch=cur["states"][state];oh=old["states"][state]
        assert ch["extension"]["full_map_readout_sha256"]==oh["extension"]["full_map_readout_sha256"]
        assert ch["calibrated"]["full_map_readout_sha256"]==oh["calibrated"]["full_map_readout_sha256"]
        assert sha(result/f"{raw:03d}/extension_{state}.png")==sha(run/f"results/full49_corridor/{raw:03d}/extension_{state}.png")
        checks.append(dict(raw=raw,state=state,uncalibrated_RGB_PNG_SHA_same_as_prior_strip=True,
            all_four_readout_hashes_same_as_prior_strip_and_prior_wide_field_strip=True))
target137=read(result/"137/metrics.json");coverage=[]
for state in ("keep","delete"):
    for variant in ("extension","calibrated"):
        d=target137["states"][state][variant]["domains"]["fixed720"]
        coverage.append(dict(raw=137,state=state,variant=variant,pixels=d["alpha"]["pixels"],
            solo_alpha_min=d["extension_solo_alpha"]["min"],solo_alpha_below098=d["extension_solo_alpha_below098"],
            solo_pixels_ge098=720-d["extension_solo_alpha_below098"],combined_alpha_min=d["alpha"]["min"],
            environment_T_max=d["environment_T"]["max"],environment_pixels_above002=d["environment_T_above002"],
            actual_extension_alphaT_min=d["extension_alphaT"]["min"],actual_extension_alphaT_mean=d["extension_alphaT"]["mean"]))
coverage_note="Solo coverage and total environment transmittance are mechanism checks, not hidden RGB accuracy. Total alpha includes vehicle/static layers; actual extension alpha*T is listed separately."
report["prior_strip_cross_checks"]=checks
report["fixed720_raw137_exact_threshold_counts"]=coverage
report["coverage_count_interpretation"]=coverage_note
report["corridor_scope_check"]["material_refit_for_strip"]=True
report["corridor_scope_check"]["material_sha256"]=report["corridor_scope_check"].pop("material_sha256_unchanged")
report["unknown_body_interface_risk"]=dict(scene_adoption=False,
    weight_loss_diagnostic="Same full-map alpha/readout hashes as the prior strip control: the previously measured unknown-body contribution changes persist exactly.",
    source= "review/full49_corridor_inferred_asset_contribution.json",
    scientific_views="review/asset_ground_conflict/raw130_a6_crop_and_loss.png",
    bounded_probe="review/asset_ground_conflict/canonical_chart_35ray_first_hit_probe.json",
    interpretation="A/B mixed mechanisms; no hidden-body GT and no claim a complete first-hit renderer was implemented. Passing observed gates would not resolve inferred-body adoption risk.")
path.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
md=review/"full49_corridor_cellfit_review.md";text=md.read_text()
if "最终原13门逐项" not in text:
    lines=["", "最终原13门逐项（calibrated 对原V44；门限未改）：", "", "|状态|指标|门|阈值|退化|结果|", "|---|---|---|---:|---:|---|"]
    for g in report["preservation_gates"]:
        if g["candidate"]!="calibrated":continue
        lines.append(f'|{g["state"]}|{g["metric"]}|{g["kind"]}|{g["threshold"]:.6f}|{g["observed"]:+.9f}|'+("PASS" if g["pass_gate"] else "FAIL")+"|")
    lines += ["", "结论：12/13，唯一delete observed-road pooled +0.001043788仍超过+0.001；不采用、不延长1000步、不改门。保护域keep/delete pooled与原车辆/深度门通过。", "",
        "raw137固定720机制计数：", "", "|状态|solo min|solo≥.98|combined alpha min|env>.02|extension W均值|", "|---|---:|---:|---:|---:|---:|"]
    for r in coverage:
        if r["variant"]!="calibrated":continue
        lines.append(f'|{r["state"]}|{r["solo_alpha_min"]:.6f}|{r["solo_pixels_ge098"]}/720|{r["combined_alpha_min"]:.6f}|{r["environment_pixels_above002"]}/720|{r["actual_extension_alphaT_mean"]:.6f}|')
    lines += ["", "alpha/solo/环境计数只证明表示覆盖与透射机制；总alpha包含车体和static，不能冒充720像素road RGB准确或隐藏GT。与前一strip对照的98组四张全图readout hash一致，8key保存数组独立复算一致；未知推断车体的贡献损失仍保留，不能宣布场景采用。", "",
        "35-ray只读probe：raw130严重35点原RGB均为a37，距a6 reviewed边缘11–14px；同一现有C2 chart三角首交24点chart在前、11点ground在前。最严重18点为7/11。开放chart/matte近似不含封闭完整车体；A型几何贴地/穿地与B型Gaussian足迹/中心排序机制并存，不提供隐藏first-hit GT，也不代表完整MOG已实现。", ""]
    text+="\n".join(lines)
md.write_text(text,encoding="utf-8")
diagnostic=review/"asset_ground_conflict/DIAGNOSTIC_SUMMARY.md"
diagnostic.write_text("""# 未知车地界面诊断

![Architecture components](../corridor_architecture.svg)

raw130的35个baseline assetW>.9、损失>.01像素全部位于原a37可见区域，距原a6 reviewed边缘11–14px，非a6观测轮廓削边：5像素a6 measured-occluded、30像素unknown，0自由road。raw137的W>.5损失>.01共68像素，也全部原a37可见（20 occluded、48 unknown）。原RGB不能作隐藏a6或删后road GT。

![raw130 native crop and exact weight loss](raw130_a6_crop_and_loss.png)

![raw137 native crop and exact weight loss](raw137_a6_crop_and_loss.png)

原C2 S/F与保存du重构实际LOD行数。raw130这35点中，主导Gaussian mean离地中位0.0876m，5点mean低于road超过0.02m；3σ下界最低-0.1136m。现有连续chart有5点低于road超过0.03m，另24点高于road超过0.03m且ground真实ray交点在后，说明不能唯一归因“PVD穿地”。

35-ray canonical activeUV三角片opaque probe在matte阈值.01与.5均得到24点chart先、11点ground先；最严重18点分别7/11。三角与局部连续S交点差最大4.05e-5m。此为开放单面chart近似，不含retained非chart的闭合表面，不是真隐藏GT或完整renderer。

最严重18点的主导roadGaussian投影中心距目标1.08–2.02像素、中心depth相对body中位近0.763m。CPU仅C2+road近似复合对真实GPU new-assetW最大差0.006，支持足迹/中心排序干涉也是候选机制；不是唯一物理根因。独立ground仍是实测池插值，有高度/遮挡边界，不把模型首交当真实车体首交。

所有模型、车体、road几何/alpha均未改；本诊断0 GPU forward。最终材质拟合的全图alpha hash与该strip对照相同，所以未知部分风险持续，需与原13个已观测门分列，不宣布场景采用。
""",encoding="utf-8")
delivery_files=[path,md,review/"asset_ground_conflict/observation_state_strata.json",review/"asset_ground_conflict/geometry_depth_evidence_v2.json",
    review/"asset_ground_conflict/canonical_chart_35ray_first_hit_probe.json",diagnostic,
    review/"asset_ground_conflict/raw130_native_scientific_comparison.png",review/"asset_ground_conflict/raw130_a6_crop_and_loss.png",
    review/"asset_ground_conflict/raw137_native_scientific_comparison.png",review/"asset_ground_conflict/raw137_a6_crop_and_loss.png",
    review/"asset_ground_conflict/raw130_strong09_loss01_mask.png",review/"asset_ground_conflict/raw130_strong09_loss50_mask.png",
    review/"asset_ground_conflict/raw137_moderate05_loss01_mask.png",review/"corridor_architecture.svg"]
manifest=dict(schema="R0_FINAL_INDEPENDENT_REVIEW_DELIVERY_V1",verdict=report["verdict"],scene_adoption=False,
    files={str(p.relative_to(review)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in delivery_files},
    original_evaluation_receipts_modified=False,GPU_review_forwards=0)
(review/"final_independent_review_delivery.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps(dict(coverage=coverage,review_sha256=sha(path),md_sha256=sha(md),files=len(delivery_files))),flush=True)

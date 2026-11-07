"""Finalize corridor review and keep all parent evaluation receipts immutable."""
import hashlib
import json
from pathlib import Path
import sys


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
run=Path(sys.argv[1]);review=run/"review"
report=json.loads((review/"full49_corridor_review.json").read_text());md=review/"full49_corridor_review.md"
text=md.read_text().replace("material_architecture.svg","corridor_architecture.svg")
if "原13项保护门逐项" not in text:
    lines=["", "原13项保护门逐项（calibrated 对原V44）：", "", "|状态|指标|门|阈值|实际退化|结果|", "|---|---|---|---:|---:|---|"]
    for g in report["preservation_gates"]:
        if g["candidate"]!="calibrated":continue
        lines.append(f'|{g["state"]}|{g["metric"]}|{g["kind"]}|{g["threshold"]:.6f}|{g["observed"]:+.8f}|'+("PASS" if g["pass_gate"] else "FAIL")+"|")
    lines += ["", "结论：原13门通过12项，protected keep/delete在全部49帧的pooled门真正通过；delete observed-road pooled仍+0.00126552，超过+0.001阈值，不采用。固定world-strip保留10098行、inactive4563行，沿用原wide field，不混入材质重拟合因素。", "",
              "保真补充诊断：4key delete中，a6 candidate且非visible、baseline assetW>0.9的3482像素里35像素贡献损失>0.01、18像素损失>0.5；最大0.94346，严重变化位于raw130。此为shared-sort贡献机制，旧PVD隐藏几何不是GT，unknown也不能默认road。冻结资产行不等于贡献不变。", "",
              "720目标的solo/实际alpha均值不等于720/720逐像素过阈值；本轮未保存min及98计数，不向旧收据补写。新depth是ΣTαz/ΣTα对稀疏LiDAR最近回波的描述性残差，不是首交精度。", ""]
    text+="\n".join(lines)
md.write_text(text,encoding="utf-8")
names=["full49_corridor_review.json","full49_corridor_review.md","full49_corridor_inferred_asset_contribution.json",
       "full49_corridor_inferred_asset_contribution.md","full49_corridor_inferred_asset_contribution.npz","corridor_architecture.svg"]
delivery=dict(schema="R0_CORRIDOR_INDEPENDENT_REVIEW_DELIVERY_V1",status="COMPLETE_CURRENT_CONTROL_PENDING_MATCHED_REFIT",
    files={name:dict(bytes=(review/name).stat().st_size,sha256=sha(review/name)) for name in names},original_receipts_modified=False,GPU_review_forwards=0)
(review/"corridor_reviews_delivery.json").write_text(json.dumps(delivery,indent=2)+"\n")
print(json.dumps(delivery),flush=True)

"""Finalize review documentation without modifying evaluation receipts."""
import hashlib
import json
from pathlib import Path
import sys


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
run=Path(sys.argv[1]);review=run/"review"
for name in ("pilot_worldfield_r2","full49_worldfield"):
    md=review/(name+"_review.md");text=md.read_text().replace("../surface/architecture.svg","material_architecture.svg")
    if "原13项保护门逐项" not in text:
        report=json.loads((review/(name+"_review.json")).read_text())
        lines=["", "原13项保护门逐项（calibrated 对原V44）：", "", "|状态|指标|门|阈值|实际退化|结果|", "|---|---|---|---:|---:|---|"]
        for g in report["preservation_gates"]:
            if g["candidate"]!="calibrated":continue
            lines.append(f'|{g["state"]}|{g["metric"]}|{g["kind"]}|{g["threshold"]:.6f}|{g["observed"]:+.8f}|'+("PASS" if g["pass_gate"] else "FAIL")+"|")
        lines += ["", "深度定义：原生Gaussian输出为Σ(Tαz)/Σ(Tα)期望相机深度；scene_depth_m是稀疏真实LiDAR最近回波。此比较不是first-hit精度；大绝对残差如实报告，不能用材质对照中的depth恒等冒充几何准确。", ""]
        text+="\n".join(lines)
    if name=="full49_worldfield" and "横向域定位" not in text:
        text+="\n横向域定位：关键4帧exact float中，前方且超出原y[-3.35,2.71]范围的像素占observed-road仅3.59%，却贡献keep66.98%/delete59.44%正退化；protected占4.51%却贡献71.46%/62.45%。主要来自y>2.71，且几乎都有actual extension W>.01。固定world-strip是值得真实重放的领域边界控制；此关联不等于已验证裁剪会通过所有门。\n"
    md.write_text(text,encoding="utf-8")
md=review/"full49_worldfield_mask_localization.md";text=md.read_text()
if "![Architecture" not in text:text=text.replace("# 固定监督mask残差定位\n","# 固定监督mask残差定位\n\n![Architecture components](material_architecture.svg)\n",1)
md.write_text(text,encoding="utf-8")
names=["pilot_worldfield_r2_review.json","pilot_worldfield_r2_review.md","full49_worldfield_review.json","full49_worldfield_review.md",
       "full49_worldfield_mask_localization.json","full49_worldfield_mask_localization.md","full49_worldfield_lateral_localization.json","material_architecture.svg"]
manifest=dict(schema="R0_WORLDFIELD_INDEPENDENT_REVIEWS_DELIVERY_V1",status="COMPLETE_CURRENT_REVIEWS_PENDING_FURTHER_CONTROLS",
    files={name:dict(bytes=(review/name).stat().st_size,sha256=sha(review/name)) for name in names},
    original_receipts_modified=False,GPU_review_forwards=0)
(review/"worldfield_reviews_delivery.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps(manifest),flush=True)

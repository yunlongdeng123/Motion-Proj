"""Build an offline, separately reported seam/contact review into a NEW directory.

Standard library only. Reads immutable result files; does not evaluate models,
change source receipts, update blue/red evidence, or publish anything.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import tarfile

OLD_RUN = "scene008_road_r0_v44_20261107_r1"
RAWS = list(range(89, 138))
METRICS = ("a6_RGB", "a37_RGB", "observed_road_RGB", "protected_RGB", "a6_depth_m", "observed_road_depth_m")
LABELS = {"a6_RGB":"原可见 C2 RGB MAE", "a37_RGB":"原可见 a37 RGB MAE", "observed_road_RGB":"原观测道路 RGB MAE",
          "protected_RGB":"原保护域 RGB MAE", "a6_depth_m":"原可见 C2 expected-depth MAE (m)",
          "observed_road_depth_m":"稀疏 LiDAR 对照 expected-depth MAE (m)"}


def read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else default


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def esc(value):return html.escape(str(value),quote=True)
def fmt(value):return "—" if value is None else f"{value:.6f}"
def delta(value):return "—" if value is None else f"{value:+.6f}"


ARCHITECTURE = '''<svg xmlns="http://www.w3.org/2000/svg" width="1160" height="310" viewBox="0 0 1160 310">
<defs><marker id="a" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7" fill="#60758d"/></marker></defs>
<rect width="1160" height="310" fill="#fff"/>
<g fill="#edf3fa" stroke="#60758d" stroke-width="1.4"><rect x="15" y="95" width="205" height="115" rx="9"/><rect x="265" y="95" width="220" height="115" rx="9"/><rect x="530" y="95" width="225" height="115" rx="9"/></g>
<g fill="#e9f4ff" stroke="#3f78a8"><rect x="810" y="20" width="325" height="75" rx="9"/></g>
<g fill="#fff0ed" stroke="#bc6554"><rect x="810" y="120" width="325" height="75" rx="9"/></g>
<g fill="#f2f3f5" stroke="#8390a0"><rect x="810" y="220" width="325" height="70" rx="9"/></g>
<g font-family="Arial,sans-serif" font-size="17" fill="#142538" text-anchor="middle">
<text x="117" y="127">Frozen C2 + road RGB</text><text x="117" y="155">Real FRONT / ground</text><text x="117" y="183">49 camera poses</text>
<text x="375" y="126">World-space road controls</text><text x="375" y="155">2x2 split / relative Mip</text><text x="375" y="183">Optional boundary material</text>
<text x="642" y="126">Native single-sort rendering</text><text x="642" y="155">keep / delete paired</text><text x="642" y="183">RGB + contribution layers</text>
<text x="972" y="49">Blue: observed seam</text><text x="972" y="77">Fixed masks / world boundary bands</text>
<text x="972" y="149">Red: contact and ordering</text><text x="972" y="177">Fixed opaque chart / ground ray sets</text>
<text x="972" y="248">49-frame observed protection</text><text x="972" y="275">Supporting checks; no combined pass label</text>
</g><g stroke="#60758d" stroke-width="2" fill="none" marker-end="url(#a)"><path d="M220 152H265"/><path d="M485 152H530"/><path d="M755 130L810 58"/><path d="M755 152H810"/><path d="M755 177L810 253"/></g></svg>'''


def load_result(folder, label):
    receipt=read(folder/"receipt.json")
    if receipt is None:return None
    frames=receipt.get("frames",receipt.get("records",[]))
    if not frames or not isinstance(frames[0],dict) or "variants" not in frames[0]:
        frames=[]
        for raw in receipt.get("raws",RAWS):
            data=read(folder/f"{raw:03d}/metrics.json",[])
            frames.extend(data if isinstance(data,list) else [data])
    totals=defaultdict(lambda:[0.,0])
    hashes=[];per_frame=[]
    for row in frames:
        if not row or "variants" not in row:continue
        variants=row["variants"]
        if "parent" not in variants or "candidate" not in variants:continue
        hashes.append(variants["parent"].get("asset_rows_hash")==variants["candidate"].get("asset_rows_hash"))
        for variant in ("parent","candidate"):
            for metric in METRICS:
                value=variants[variant].get(metric)
                if value and value.get("count",0):
                    t=totals[(row["state"],variant,metric)];t[0]+=float(value["sum"]);t[1]+=int(value["count"])
        per_frame.append({"raw":int(row["raw"]),"state":row["state"],"variants":{v:{k:x.get("mean") for k,x in variants[v].items() if k in METRICS and isinstance(x,dict)} for v in ("parent","candidate")}})
    pooled=[]
    for state in ("keep","delete"):
        for metric in METRICS:
            p=totals.get((state,"parent",metric));c=totals.get((state,"candidate",metric))
            if not p or not c:continue
            pm,cm=p[0]/p[1],c[0]/c[1]
            pooled.append(dict(state=state,metric=metric,parent=pm,candidate=cm,delta=cm-pm,parent_count=p[1],candidate_count=c[1]))
    stored={(x["state"],x["variant"],x["metric"]):x for x in receipt.get("pooled",[]) if all(k in x for k in ("state","variant","metric"))}
    mismatch=[]
    for key,(s,n) in totals.items():
        x=stored.get(key)
        if x is None or int(x["count"])!=n or abs(float(x["sum"])-s)>1e-6 or abs(float(x["mean"])-s/n)>1e-12:
            mismatch.append(list(key))
    return dict(label=label,folder=str(folder),raws=sorted({int(x["raw"]) for x in frames if "raw" in x}),
        seconds=receipt.get("seconds"),forwards=receipt.get("forwards"),factor=receipt.get("factor"),relative_mip=receipt.get("relative_mip"),
        material_override=receipt.get("material_override"),pooled=pooled,per_frame=per_frame,
        vehicle_hash_equal_records=sum(hashes),vehicle_hash_compared_records=len(hashes),pooled_receipt_mismatches=mismatch,
        receipt_sha256=sha(folder/"receipt.json"),whole_scene_adoption_claim=False)


def discover(run, explicit, names):
    if explicit:
        p=Path(explicit);p=p if p.is_absolute() else run/p
        if not (p/"receipt.json").exists():raise FileNotFoundError(p/"receipt.json")
        return p
    return next((run/"results"/n for n in names if (run/"results"/n/"receipt.json").exists()),None)


def blue_summaries(run):
    rows=[]
    for p in sorted((run/"blue").glob("*_fixed_blue_comparison.json")):
        a=read(p,{});region=a.get("regions",{}).get("original_road",{})
        if not all(k in region for k in ("parent","candidate")):continue
        pa,ca=region["parent"],region["candidate"]
        if "RGB_abs_error_mean" not in pa or "RGB_abs_error_mean" not in ca:continue
        jumps=a.get("matched_tangent_residual_jumps",{})
        near=next((v for k,v in jumps.items() if k.startswith("near_x")),{})
        name=a.get("name",p.stem)
        reference=a.get("reference_label",a.get("parent_name"))
        if not reference:reference="n2 + relative Mip" if "_vs_n2_mip" in name.lower() or "_vs_aa" in name.lower() else "普通 n2" if "_vs_n2" in name.lower() else "当前 R0 · n1"
        rows.append(dict(name=name,reference=reference,raw=a.get("raw",128),state=a.get("state","delete"),pixels=ca.get("RGB_error_denominator_pixels",ca.get("pixels")),
            parent_MAE=pa["RGB_abs_error_mean"],candidate_MAE=ca["RGB_abs_error_mean"],delta=ca["RGB_abs_error_mean"]-pa["RGB_abs_error_mean"],
            near_jump_parent=near.get("parent",{}).get("equal_bin_abs_residual_jump_mean"),
            near_jump_candidate=near.get("candidate",{}).get("equal_bin_abs_residual_jump_mean"),
            near_paired_bins=near.get("candidate",{}).get("paired_bins"),file="evidence/blue/"+p.name,sha256=sha(p)))
    # The source-locked final summary is authoritative for the main AA row;
    # do not depend on the timing/schema of auxiliary comparison files.
    final=read(run/"blue/blue_final_metrics.json",{})
    final_rows={r["method"]:r for r in final.get("primary_metric_rows",[])}
    authoritative=[]
    for method,reference,name in (("n2","R0","n2"),("AA","R0","AA_vs_R0"),
            ("n2_material","n2","n2_material_vs_n2"),("AA_frozen_material","AA","AA_frozen_material_vs_AA")):
        if method not in final_rows or reference not in final_rows:continue
        r,b=final_rows[method],final_rows[reference]
        assert r["count_pixels"]==b["count_pixels"]
        authoritative.append(dict(name=name,reference=reference,raw=r["raw"],state=r["state"],domain=r["domain"],pixels=r["count_pixels"],
            parent_MAE=b["abs_RGB_error_mean"],candidate_MAE=r["abs_RGB_error_mean"],delta=r["abs_RGB_error_mean"]-b["abs_RGB_error_mean"],
            near_jump_parent=b["near_x_equal_bin_abs_residual_jump"],near_jump_candidate=r["near_x_equal_bin_abs_residual_jump"],
            near_paired_bins=r["near_x_paired_bins"],file="evidence/blue/blue_final_metrics.json",sha256=sha(run/"blue/blue_final_metrics.json"),
            main_candidate=method=="AA",partial_5key=method=="AA_frozen_material"))
    indexed={r["name"]:r for r in rows}
    indexed.update({r["name"]:r for r in authoritative})
    return list(indexed.values())


def fivekey_seam_summaries(run):
    source=read(run/"blue/fivekey_fixed_world_patch_comparison.json",{})
    index=defaultdict(dict)
    for r in source.get("per_view",[]):
        for edge,v in r.get("edges",{}).items():index[(r["raw"],r["state"],edge)][r["method"]]=v
    rows=[]
    for (raw,state,edge),values in sorted(index.items()):
        if "R0" not in values or "AA" not in values:continue
        r,a=values["R0"],values["AA"]
        assert r["paired_bins"]==a["paired_bins"], (raw,state,edge)
        before,after=r["equal_bin_abs_residual_jump_mean"],a["equal_bin_abs_residual_jump_mean"]
        rows.append(dict(raw=raw,state=state,boundary=edge,paired_bins=r["paired_bins"],
            matched_minus_count=sum(b["minus_count"] for b in r.get("bins",[])),matched_plus_count=sum(b["plus_count"] for b in r.get("bins",[])),
            R0=before,n2=values.get("n2",{}).get("equal_bin_abs_residual_jump_mean"),AA=after,
            delta=None if before is None or after is None else after-before))
    return rows


def primary_blue_both_states(run):
    source=read(run/"blue/raw128_fixed_roi_both_states.json",{})
    rows=[]
    for r in source.get("rows",[]):
        road=r.get("regions",{}).get("original_road",{});edge=r.get("edges",{}).get("near_x",{})
        if not road:continue
        rows.append(dict(raw=r["raw"],state=r["state"],method=r["method"],pixels=road["count_pixels"],
            sum=road["abs_RGB_error_sum"],MAE=road["abs_RGB_error_mean"],near_jump=edge.get("equal_bin_abs_residual_jump_mean"),
            paired_bins=edge.get("paired_bins"),matched_minus_count=edge.get("matched_minus_count"),matched_plus_count=edge.get("matched_plus_count")))
    return rows


def red_summaries(run):
    rows=[]
    for p in sorted((run/"red").glob("*_fixed_contact_comparison.json")):
        if "_pilot" in p.name and p.with_name(p.name.replace("_pilot","_full49")).exists():continue
        a=read(p,{})
        for r in a.get("cases",[]):
            if r.get("group") not in ("chart_first","ground_first","near_contact_005m") or not r.get("rays"):continue
            rows.append(dict(control=a.get("control",p.stem),raw=r["raw"],state="delete",group=r["group"],rays=r["rays"],
                parent_assetW=r["parent_assetW"]["mean"],candidate_assetW=r["candidate_assetW"]["mean"],delta=r["assetW_gain"]["mean"],
                gain_gt001=r.get("gained_gt001"),loss_gt001=r.get("lost_gt001"),remaining_severe=r.get("remaining_loss_to_V44_gt05"),
                file="evidence/red/"+p.name,sha256=sha(p)))
    return rows


def table(headers, rows):
    if not rows:return '<p class="muted">尚无该项评价收据。</p>'
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table></div>'


def build_html(data):
    def blue_table(rows):
        return table(("候选对照","raw/state/domain","参照","原road像素","参照MAE","候选MAE","变化","near匹配bins","near残差跳变"),[
            [r["name"]+(" · 主候选" if r.get("main_candidate") else " · 仅5key旁路" if r.get("partial_5key") else ""),
             str(r["raw"])+" / "+r["state"]+" / 原蓝框",r["reference"],r["pixels"],fmt(r["parent_MAE"]),fmt(r["candidate_MAE"]),delta(r["delta"]),r["near_paired_bins"] or "无支持",
             fmt(r["near_jump_parent"])+" → "+fmt(r["near_jump_candidate"])] for r in rows])
    blue=blue_table([r for r in data["blue"] if "material" not in r["name"].lower()])
    material_blue=blue_table([r for r in data["blue"] if "material" in r["name"].lower()])
    seam_views=table(("raw","状态","边界","bins / 配对minus,plus像素","R0 jump","n2 jump","AA jump","AA−R0"),[
        [r["raw"],r["state"],r["boundary"],str(r["paired_bins"])+" / "+str(r["matched_minus_count"])+", "+str(r["matched_plus_count"]),fmt(r["R0"]),fmt(r["n2"]),fmt(r["AA"]),delta(r["delta"])] for r in data["fivekey_seam"]])
    primary_both=table(("原蓝框raw128状态","方法","RGB误差sum / road像素","road MAE","near-x jump","bins / 配对minus,plus像素"),[
        [r["state"],r["method"],fmt(r["sum"])+" / "+str(r["pixels"]),fmt(r["MAE"]),fmt(r["near_jump"]),
         str(r["paired_bins"])+" / "+str(r["matched_minus_count"])+", "+str(r["matched_plus_count"])] for r in data["primary_blue_both_states"]])
    red=table(("对照","raw/state/固定组","点数","父R0 assetW","候选assetW","变化","gain>.01 / loss>.01","仍严重缺失"),[
        [r["control"],str(r["raw"])+" / "+r["state"]+" / "+r["group"],r["rays"],fmt(r["parent_assetW"]),fmt(r["candidate_assetW"]),delta(r["delta"]),
         str(r["gain_gt001"])+" / "+str(r["loss_gt001"]),r["remaining_severe"]] for r in data["red"]])
    protection=[]
    for result in data["results"]:
        for p in result["pooled"]:
            protection.append([result["label"],str(len(result["raws"]))+"帧",p["state"],LABELS[p["metric"]],fmt(p["parent"]),fmt(p["candidate"]),delta(p["delta"]),str(p["parent_count"])+" / "+str(p["candidate_count"])])
    protection=table(("对照","范围","状态","指标","父R0","候选","变化","父/候选像素数"),protection)
    v44_rows=[]
    for result in data["results"]:
        for r in result.get("relative_V44",[]):
            v44_rows.append([result["label"],r["state"],LABELS[r["metric"]],fmt(r["V44"]),fmt(r["candidate"]),delta(r["delta"]),r["count"]])
    v44_table=table(("全49对照","状态","附属指标","原V44","候选","相对V44变化","同分母像素数"),v44_rows)
    options=''.join('<option value="'+esc(v["id"])+'">'+esc(v["label"]+" · "+str(len(v["raws"]))+"帧")+'</option>' for v in data["variants"] if v["id"] not in ("v44","r0"))
    evidence=''.join('<li><a href="'+esc(p)+'">'+esc(Path(p).name)+'</a></li>' for p in data["evidence_links"])
    downloads=''.join('<li><a href="'+esc(d["file"])+'">'+esc(d["label"])+'</a> · '+esc(d["sha256"][:16])+'…</li>' for d in data.get("downloads",[]))
    ledger=data.get("round_decision",{})
    ledger_note=('本轮原生forward合计 '+str(ledger.get("total_native_forwards","—"))+'；原模型/车辆更新为0，无backward。canonical fresh replay结果由根收据另列，source与ledger可下载。') if ledger else '本轮账本待根收据冻结。'
    checks=''.join('<li>'+esc(r["label"])+': 车辆行hash '+str(r["vehicle_hash_equal_records"])+'/'+str(r["vehicle_hash_compared_records"])+
        ' 相同；pooled sum/count 收据不一致 '+str(len(r["pooled_receipt_mismatches"]))+' 项；'+str(r["forwards"])+
        ' forwards / '+str(round(r["seconds"],2) if r["seconds"] is not None else "—")+'s。</li>' for r in data["results"])
    embedded=json.dumps(data,ensure_ascii=False,allow_nan=False).replace("<","\\u003c")
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scene008 · 接缝与车地接触分项审查</title><style>
:root{font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;color:#17283a;background:#f4f6f9}*{box-sizing:border-box}body{margin:0}main{max-width:1500px;margin:auto;padding:30px 24px 60px}h1{font-size:30px;line-height:1.25;margin:0 0 12px}h2{font-size:23px;margin:0 0 10px}h3{font-size:18px;margin:15px 0 8px}p{margin:8px 0 13px}.muted,small{color:#536579}.eyebrow{font-size:13px;letter-spacing:.12em;color:#536579;text-transform:uppercase}.lead{max-width:1050px}.notice{border-left:4px solid #778ba5;background:#eaf0f6;padding:12px 16px;border-radius:4px}.cards{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:24px 0}.card,section{background:white;border:1px solid #dce3ec;border-radius:12px;padding:22px;margin:20px 0}.cards .card{margin:0;min-width:0}.blue{border-top:5px solid #3980ba}.red{border-top:5px solid #bc6454}.tag{display:inline-block;font-size:13px;border:1px solid #c5d0dd;border-radius:20px;padding:2px 10px;margin:0 4px 8px 0}.blue .tag{background:#edf7ff}.red .tag{background:#fff3ef}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}th,td{text-align:left;padding:9px 10px;border-bottom:1px solid #e5eaf1;white-space:nowrap}th{background:#f4f7fa;color:#40536b}tr:last-child td{border-bottom:0}a{color:#1c6299}details{margin:15px 0}summary{cursor:pointer;font-weight:600}.architecture{display:block;width:100%;max-width:1160px;margin:12px auto}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin:12px 0}button,select{font:inherit;padding:7px 12px;border:1px solid #bcc9d9;border-radius:7px;background:white;color:#17304b;cursor:pointer}button.active{background:#243f61;color:white;border-color:#243f61}input[type=range]{flex:1;min-width:220px}output{min-width:100px;font-weight:650}.frames{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:15px}.frames figure{margin:0;min-width:0}.frames h3{margin:5px 0 8px}.imagebox{aspect-ratio:3/2;background:#e7ecf3;display:flex;align-items:center;justify-content:center;overflow:hidden;border-radius:6px;position:relative}.imagebox img{width:100%;height:auto;display:block;cursor:zoom-in}.imagebox img[hidden]{display:none}.placeholder{padding:25px;text-align:center;font-size:14px;color:#5c6c80}.frames figcaption{font-size:13px;color:#536579;margin-top:7px;min-height:42px}.references{display:flex;gap:16px;flex-wrap:wrap}.references img{max-width:100%;width:600px}dialog{border:0;border-radius:12px;padding:16px;max-width:96vw;max-height:96vh}dialog::backdrop{background:#000b}dialog img{display:block;width:min(1440px,90vw);height:auto;image-rendering:auto}dialog p{margin:0 0 10px}.evidence{columns:2;font-size:14px}footer{font-size:13px;color:#536579;margin-top:28px}ul{padding-left:22px}@media(max-width:1000px){.cards{grid-template-columns:1fr}.frames{grid-template-columns:1fr}.frames figure{max-width:720px}.evidence{columns:1}}@media(prefers-reduced-motion:no-preference){button{transition:background .12s}}[hidden]{display:none!important}
.cards>*,.frames>*{min-width:0}a,code,pre,li,p,figcaption{overflow-wrap:anywhere}.controls>label{min-width:0;max-width:100%}select{max-width:100%}.table-wrap{max-width:100%}
</style><main><div class="eyebrow">SCENE008 · SAME-SCENE DEVELOPMENT CONTROLS</div>
<h1>路面接缝与轮胎遮挡，分别看证据</h1>
<p class="lead">车辆全程冻结。本轮主候选为 <strong>n2 + AA，保留父 cellfit1000 颜色</strong>，用于改善道路支撑与车地成像。蓝圈看原观测道路接缝；红圈看车地表面先后与原生成像贡献。18个边界颜色系数只作旁路对照，未选入主候选。</p>
<p class="notice">本轮仅为道路支撑研究候选（GEOMETRY_SUPPORT_RESEARCH_CANDIDATE）。蓝接缝跨帧不一致、仍开放；红侧支持改善，尚有严重缺失。未作整道路场景升级或全场景采用。当前R0是上一轮候选，原V44作为道路扩展前参照；49帧保护指标不合并成总通过标签。</p>
<img class="architecture" src="architecture.svg" alt="冻结输入经过世界道路控制和原生单排序渲染，分别输出蓝圈、红圈及保护检查">
<div class="cards"><article class="card blue"><span class="tag">蓝通道 · 接缝开放</span><h2>raw128局部改善，跨帧不一致</h2>
<p>raw128 蓝圈按用户截图近似登记为 x=[165,208), y=[188,234)，不是精确圆形分割。主分母为框内原始 observed-road 的1385像素；保守车辆guard另列。世界边界与采样规则在读取新控制之前冻结。</p>
'''+blue+'''<details><summary>18系数材质旁路（未选入主候选；AA组合仅5关键帧）</summary>'''+material_blue+'''</details><p><strong>原raw128蓝框 / delete / near-x / 6 bins</strong>残差跳变：R0 .012529→AA .009794；原两侧42/34px，实际配对35/33px。另一固定世界patch副视图中，<strong>raw105 / delete / lateral / 8 bins</strong> .011334→.019934退化；同raw105的<strong>keep / lateral / 8 bins</strong>则.008962→.007797改善，不能跨状态概括。18系数材质控制未选入主候选。</p><details><summary>原蓝框主分母：raw128 KEEP / DELETE各自表列</summary>'''+primary_both+'''</details><p class="muted">主蓝框与副世界patch是不同mask/分母：副视图raw128/delete/near-x虽也为6 bins，数值为.012250→.009522。raw128 far边无支持；lateral无满足每侧≥3像素的匹配bin。颜色补偿不能独自证明排序正确。</p><details><summary>副世界patch：5关键帧 keep/delete 分列（不替代上方蓝框分母）</summary>'''+seam_views+'''</details>
<p><a href="evidence/blue/roi_registration.json">固定分母/边界登记</a> · <a href="evidence/blue/blue_parent_audit.json">父R0贡献来源审查</a></p></article>
<article class="card red"><span class="tag">红通道 · 支持改善，未彻底修复</span><h2>车表面先到与地面先到分开</h2>
<p>以下红侧贡献统计均为<strong>delete状态</strong>。raw128近似红区 x=[232,284), y=[190,218)，共1456像素。原V44 assetW&gt;.9、加入父R0后loss&gt;.01的固定36点：<strong>23车表面先到、12地面先到、1接触</strong>。raw130沿用原35射线：24/11。各控制不重选射线。</p>
'''+red+'''<p>AA保留父颜色，在固定chart-first组比普通n2有明确增量；ground-first组保持零车辆贡献。相对原V44，raw128仍有6点、raw130仍有7点chart-first严重缺失（assetW损失&gt;.5）。</p><p class="muted">全部raw128固定36点原始可见内容为a37，a6为unknown或occluded，没有隐藏轮胎GT。表面参考是双面、不封闭canonical chart三角片与同场景LiDAR测量池近似地面；地面是另一CPU读出，与R0共享BUILD测量，并非独立验证数据。23/12/1仅为当前C2/地面模型关系；缺少chart命中时保留不确定。普通n2有正有负，不能称轮胎已修复。</p>
<p><a href="evidence/red/CONTACT_FINDINGS.md">红侧独立诊断</a> · <a href="evidence/red/CONTACT_CONTROL_DESIGN.md">细分及relative Mip适用边界</a></p></article></div>
<section><h2>49帧原生图对照</h2><p class="muted">所有图像均为原生480×320 PNG。a37保留/删除对所有方法同步；原始真实RGB始终含原场景内容。点击图像放大。仅关键帧的控制在其余帧显示缺测，不补图。</p>
<div class="controls"><button id="prev" aria-label="上一帧">←</button><input id="frame" type="range" min="89" max="137" step="1" value="128" aria-label="raw帧"><button id="next" aria-label="下一帧">→</button><output id="rawLabel">raw128 / 49帧</output></div>
<div class="controls"><button data-state="keep">a37 保留</button><button data-state="delete" class="active">a37 删除</button><label>中图 <select id="variant">'''+options+'''</select></label><label>旁路 <select id="secondary"><option value="none">不显示旁路</option>'''+options+'''</select></label><button id="toggleReference">显示原V44 / 真实RGB</button></div>
<div class="frames"><figure><h3>当前 R0 · n1</h3><div class="imagebox"><img id="r0Image" hidden alt=""><div id="r0Missing" class="placeholder"></div></div><figcaption>上一轮 cellfit1000 候选；冻结车辆、父道路与材质。</figcaption></figure>
<figure><h3 id="candidateTitle">普通 n2</h3><div class="imagebox"><img id="candidateImage" hidden alt=""><div id="candidateMissing" class="placeholder"></div></div><figcaption id="candidateCaption"></figcaption></figure>
<figure id="secondaryPanel"><h3 id="secondaryTitle">旁路对照</h3><div class="imagebox"><img id="secondaryImage" hidden alt=""><div id="secondaryMissing" class="placeholder"></div></div><figcaption id="secondaryCaption"></figcaption></figure>
<figure id="referencePanel" hidden><h3>原 V44 · 扩展前参照</h3><div class="imagebox"><img id="v44Image" hidden alt=""><div id="v44Missing" class="placeholder"></div></div><figcaption>原V44车辆与道路成像参照；标签随当前状态更新。</figcaption></figure>
<figure id="observedPanel" hidden><h3>原始真实 FRONT RGB</h3><div class="imagebox"><img id="observedImage" hidden alt=""><div id="observedMissing" class="placeholder"></div></div><figcaption>这是原观察，不是删除a37后的隐藏车体/路面真值。</figcaption></figure></div></section>
<section><h2>原观测保护与覆盖 · 附属指标</h2><p>数值由每帧sum/count重新汇总，RGB为编码sRGB三通道绝对误差的逐像素均值。每行父R0与候选使用同一帧范围和分母；5关键帧不能当作全49帧结果。</p>
'''+protection+'''<details><summary>相对原V44的同分母全49附属指标</summary><p>仍保留相对道路扩展前V44的退化量；旧单项RGB门的微小余量不构成整场景成功，也不代表接缝已解决。下表仅含全49收据。</p>'''+v44_table+'''</details><details><summary>贡献、深度与冻结口径</summary><p>asset参数冻结不保证成像贡献冻结。Talpha、group alpha与expected centre depth都受原生Gaussian足迹和中心排序影响，不是不透明首交深度。LiDAR nearest-return与Gaussian expected-depth口径不同，深度MAE仅描述该比较。relative Mip是parent-relative滤波能量控制，不是完整Mip/2DGS/MOG；高alpha、重叠、截断与early-stop下不保证逐像素覆盖守恒。</p><ul>'''+checks+'''</ul></details></section>
<section><h2>证据与范围</h2><p>本轮未扩展R1。前轮FRONT真实支持基准与其分母仍单独保留；本轮没有新增学习道路先验，不能据此宣称跨镜头完整恢复。隐藏车体与未观测路面不因unknown状态变成可用真值。</p>
<h3>Canonical道路、source与计算账本</h3><p>'''+esc(ledger_note)+'''</p><ul>'''+downloads+'''</ul><details><summary>查看来源和独立审查文件</summary><ul class="evidence">'''+evidence+'''</ul><p><a href="report_data.json">完整页面数据与输入SHA</a> · <a href="file_manifest.json">离线文件SHA清单</a></p></details></section>
<footer>仅离线审查稿；没有发布动作。构建时间 '''+esc(data["built_at_UTC"])+'''。模型采用状态由独立分项评价决定，此页面不生成总pass。</footer>
</main><dialog id="zoom"><p><span id="zoomLabel"></span> <button id="closeZoom">关闭</button></p><img id="zoomImage" alt=""></dialog>
<script id="report-data" type="application/json">'''+embedded+'''</script><script>
'use strict';const D=JSON.parse(document.querySelector('#report-data').textContent);let state='delete',generation=0;
const byId=id=>document.getElementById(id), variants=Object.fromEntries(D.variants.map(v=>[v.id,v]));
function showImage(id,missing,path,label,ticket){const img=byId(id),box=byId(missing);img.hidden=true;img.removeAttribute('src');img.alt=label;box.hidden=false;box.textContent=path?'正在读取图像…':'此帧尚无该对照结果';if(!path)return;const pending=new Image();pending.onload=()=>{if(ticket!==generation)return;img.src=path;img.hidden=false;box.hidden=true};pending.onerror=()=>{if(ticket!==generation)return;box.textContent='图像读取失败，请核对离线文件完整性';};pending.src=path;}
function refresh(){const ticket=++generation,raw=Number(byId('frame').value),key=String(raw)+'_'+state,v=variants[byId('variant').value],status=state==='keep'?'a37保留':'a37删除';
byId('rawLabel').textContent='raw'+raw+' / 49帧';document.querySelectorAll('[data-state]').forEach(b=>b.classList.toggle('active',b.dataset.state===state));
byId('candidateTitle').textContent=v.label;byId('candidateCaption').textContent=(v.id==='aa'?'本轮主候选；':'旁路对照；')+v.caption+' · raw'+raw+' · '+status;
showImage('r0Image','r0Missing',variants.r0.images[key],'当前R0 raw'+raw+' '+status,ticket);
showImage('candidateImage','candidateMissing',v.images[key],v.label+' raw'+raw+' '+status,ticket);
const secondary=variants[byId('secondary').value];byId('secondaryPanel').hidden=!secondary;if(secondary){byId('secondaryTitle').textContent=secondary.label;byId('secondaryCaption').textContent='旁路对照，未选入主候选 · '+secondary.caption+' · raw'+raw+' · '+status;showImage('secondaryImage','secondaryMissing',secondary.images[key],secondary.label+' raw'+raw+' '+status,ticket);}
showImage('v44Image','v44Missing',variants.v44.images[key],'原V44 raw'+raw+' '+status,ticket);
showImage('observedImage','observedMissing',D.observed[String(raw)],'真实原观察 raw'+raw,ticket);}
byId('frame').addEventListener('input',refresh);byId('variant').addEventListener('change',refresh);byId('secondary').addEventListener('change',refresh);
byId('prev').onclick=()=>{byId('frame').value=Math.max(89,Number(byId('frame').value)-1);refresh()};byId('next').onclick=()=>{byId('frame').value=Math.min(137,Number(byId('frame').value)+1);refresh()};
document.querySelectorAll('[data-state]').forEach(b=>b.onclick=()=>{state=b.dataset.state;refresh()});
byId('toggleReference').onclick=()=>{const hidden=!byId('referencePanel').hidden;byId('referencePanel').hidden=hidden;byId('observedPanel').hidden=hidden;byId('toggleReference').textContent=hidden?'显示原V44 / 真实RGB':'收起参考图';};
document.querySelectorAll('.imagebox img').forEach(img=>img.onclick=()=>{if(img.hidden)return;byId('zoomImage').src=img.src;byId('zoomImage').alt=img.alt;byId('zoomLabel').textContent=img.alt;byId('zoom').showModal()});byId('closeZoom').onclick=()=>byId('zoom').close();
byId('variant').value=variants.aa?'aa':'n2';byId('secondary').value=variants.material?'material':'n2';refresh();
</script></html>'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,default=Path(__file__).resolve().parent)
    p.add_argument("--output",type=Path)
    p.add_argument("--parent-run",type=Path)
    p.add_argument("--aa-result",type=Path)
    p.add_argument("--material-result",type=Path)
    p.add_argument("--intermediate-result",type=Path,action="append",default=[],help="Repeat for isolated intermediate result folders; keeps their own table and viewer label")
    args=p.parse_args();run=args.run.resolve();output=(args.output or run/"delivery_draft_v1").resolve()
    if output.exists():raise FileExistsError("Use a new --output directory: "+str(output))
    if not (run/"results/n2_full49/receipt.json").exists():raise FileNotFoundError(run/"results/n2_full49/receipt.json")
    old=(args.parent_run or run.parent/OLD_RUN).resolve()
    aa=discover(run,args.aa_result,("n2_mip_full49","n2_mip_pilot","n2_mip_pilot_v1"))
    material=discover(run,args.material_result,("boundary_material_full49","n2_boundary_material_full49","boundary_material_pilot","n2_boundary_material_pilot"))
    result_specs=[("n2",run/"results/n2_full49","普通 n2 · 2×2同面细分")]
    if aa:result_specs.append(("aa",aa,"n2 + relative Mip（AA）"))
    for i,given in enumerate(args.intermediate_result):
        folder=discover(run,given,())
        label="普通 n2 + 边界材质（隔离，未选入）" if "mip" not in folder.name else "AA + 边界材质（旁路，未选入）"
        result_specs.append((f"intermediate_{i+1}",folder,label))
    if material:
        mr=read(material/"receipt.json",{})
        scope="5关键帧" if len(mr.get("raws",[]))==5 else str(len(mr.get("raws",[])))+"帧"
        label="n2 + AA + 边界材质（"+scope+"旁路，未选入）" if mr.get("relative_mip") else "普通 n2 + 边界材质（未选入）"
        result_specs.append(("material",material,label))
    output.mkdir(parents=True);manifest={};sources={}
    def copy(source,relative):
        if not source.exists():return None
        dest=output/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        manifest[relative]=dict(bytes=dest.stat().st_size,sha256=sha(dest));sources[str(source)]=manifest[relative]["sha256"]
        return relative
    variants=[]
    for identifier,label,folder,prefix in (("v44","原 V44",old/"results/full49_corridor_cellfit","baseline"),
            ("r0","当前 R0 · n1",run/"results/n2_full49","parent")):
        images={}
        for raw in RAWS:
            for state in ("keep","delete"):
                rel=copy(folder/f"{raw:03d}/{prefix}_{state}.png",f"frames/{identifier}/{raw}_{state}.png")
                if rel:images[f"{raw}_{state}"]=rel
        variants.append(dict(id=identifier,label=label,images=images,raws=sorted({int(k.split('_')[0]) for k in images})))
    results=[]
    old_receipt=read(old/"results/full49_corridor_cellfit/receipt.json",{})
    previous_receipt_link=copy(old/"results/full49_corridor_cellfit/receipt.json","evidence/results/previous_R0_receipt.json")
    old_baseline={(x["state"],x["metric"]):x for x in old_receipt.get("pooled",[]) if x["variant"]=="baseline"}
    for identifier,folder,label in result_specs:
        result=load_result(folder,label);result["relative_V44"]=[]
        if result["raws"]==RAWS:
            for row in result["pooled"]:
                ref=old_baseline.get((row["state"],row["metric"]))
                if ref and row["metric"] in ("observed_road_RGB","protected_RGB") and ref.get("pixels")==row["candidate_count"]:
                    result["relative_V44"].append(dict(state=row["state"],metric=row["metric"],count=row["candidate_count"],V44=ref["mean"],candidate=row["candidate"],delta=row["candidate"]-ref["mean"]))
        results.append(result);images={}
        for raw in RAWS:
            for state in ("keep","delete"):
                rel=copy(folder/f"{raw:03d}/candidate_{state}.png",f"frames/{identifier}/{raw}_{state}.png")
                if rel:images[f"{raw}_{state}"]=rel
        variants.append(dict(id=identifier,label=label,images=images,raws=sorted({int(k.split('_')[0]) for k in images}),
            caption="父RGB材质冻结；仅细分支撑" if identifier=="n2" else "per-camera滤波幅度控制；不是首交渲染" if identifier=="aa" else "独立材质对照；车辆与支撑冻结"))
        copy(folder/"receipt.json",f"evidence/results/{identifier}_receipt.json")
    observed={}
    for raw in RAWS:
        src=run/f"results/n2_full49/{raw:03d}/observed.png"
        rel=copy(src,f"frames/observed/{raw}.png")
        if rel:observed[str(raw)]=rel
    evidence=[]
    if previous_receipt_link:evidence.append(previous_receipt_link)
    for side in ("red","blue"):
        for src in sorted((run/side).glob("*")):
            if not src.is_file() or src.suffix.lower() not in (".json",".md",".svg",".png",".py",".npz"):continue
            # Source/array evidence is copied only when explicitly linked by a
            # public MD below, keeping private probe arrays out of the page.
            if src.suffix.lower() in (".py",".npz"):continue
            rel=copy(src,f"evidence/{side}/{src.name}")
            if src.suffix.lower() in (".json",".md"):evidence.append(rel)
    # Follow small, explicitly linked local evidence dependencies in MD, e.g.
    # formal material contracts. Never recursively collect observation banks.
    pending=list((output/"evidence").rglob("*.md"));seen=set()
    while pending:
        doc=pending.pop()
        if doc in seen:continue
        seen.add(doc);source_doc=run/doc.relative_to(output/"evidence")
        for ref in re.findall(r'\[[^\]]*\]\(([^)]+)\)',doc.read_text(encoding="utf-8-sig")):
            ref=ref.strip("<>").split("#")[0]
            if not ref or re.match(r"^(?:https?:|data:|mailto:)",ref):continue
            source=(source_doc.parent/ref).resolve()
            try:relative=source.relative_to(run)
            except ValueError:continue
            if not source.is_file() or source.stat().st_size>10*1024*1024:continue
            if source.suffix.lower() not in (".json",".md",".svg",".png",".py",".npz"):continue
            dest_relative="evidence/"+relative.as_posix()
            if not (output/dest_relative).exists():copy(source,dest_relative)
            if source.suffix.lower()==".md":pending.append(output/dest_relative)
    contract=copy(run/"contract/round_scope.json","evidence/round_scope.json")
    if contract:evidence.append(contract)
    decision_link=copy(run/"round_decision.json","evidence/round_decision.json")
    if decision_link:evidence.append(decision_link)
    downloads=[]
    def archive(label,relative,entries,base):
        entries=sorted(set(p for p in entries if p.is_file()))
        if not entries:return
        dest=output/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        with tarfile.open(dest,"w:gz") as tf:
            for source in entries:
                try:member=source.relative_to(base)
                except ValueError:member=Path("builder")/source.name
                tf.add(source,arcname=str(member),recursive=False);sources[str(source)]=sha(source)
        manifest[relative]=dict(bytes=dest.stat().st_size,sha256=sha(dest));downloads.append(dict(label=label,file=relative,**manifest[relative]))
    canonical=run/"engineer/canonical_n2_mip_candidate_v2"
    if canonical.exists():
        entries=[p for p in canonical.iterdir() if p.is_file() and p.suffix in (".npz",".py",".json")]
        entries+=list((canonical/"parity").glob("*.json"))
        entries+=list((canonical/"parity").glob("*/*comparison.json"))
        archive("主候选 canonical newroad + runtime/source + replay receipt（不含完整场景checkpoint）","downloads/canonical_n2_mip_candidate_v2.tgz",entries,canonical)
        for name in ("asset_interface.json","scene_manifest.json","cpu_export_verification.json","parity/receipt.json"):
            rel=copy(canonical/name,"evidence/canonical/"+name)
            if rel:evidence.append(rel)
    source_entries=list((run/"source").rglob("*.py"))+list((run/"engineer").glob("*.py"))+list((run/"engineer").glob("*.json"))
    source_entries+=list((run/"red").glob("*.py"))+list((run/"blue").glob("*.py"))
    source_entries+=list((run/"contract").glob("*.json"))+[run/"round_decision.json",Path(__file__)]
    # Include the actual executed source snapshots and compact fit ledgers.
    for _,folder,_ in result_specs:
        source_entries+=list((folder/"source_snapshot").rglob("*.py"))+list((folder/"source_snapshot").rglob("*.json"))
    source_entries+=[p for p in (run/"blue/boundary_material_v1").rglob("*.json") if p.is_file()]
    archive("本轮 source snapshots、合同与forward/fit账本","downloads/sources_and_ledgers.tgz",source_entries,run)
    data=dict(schema="SEPARATE_BLUE_RED_OFFLINE_REVIEW_V1",built_at_UTC=datetime.now(timezone.utc).isoformat(),
        run=str(run),parent_run=str(old),raws=RAWS,variants=variants,observed=observed,results=results,
        blue=blue_summaries(run),primary_blue_both_states=primary_blue_both_states(run),fivekey_seam=fivekey_seam_summaries(run),red=red_summaries(run),evidence_links=evidence,input_sha256=sources,downloads=downloads,
        round_decision=read(run/"round_decision.json",{}),
        main_candidate="aa",main_candidate_status="GEOMETRY_SUPPORT_RESEARCH_CANDIDATE",main_candidate_material="frozen parent cellfit1000",boundary_material_selected=False,
        adoption_claim=False,combined_pass_label=False,R1_expanded_this_round=False,learned_road_prior_added=False,
        model_updates_by_builder=0,GPU_used_by_builder=False,published=False,
        builder_sha256=sha(Path(__file__)))
    (output/"architecture.svg").write_text(ARCHITECTURE,encoding="utf-8")
    (output/"report_data.json").write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (output/"index.html").write_text(build_html(data),encoding="utf-8")
    # Every file copied has a source SHA; generated artifacts are also hashed.
    for name in ("architecture.svg","report_data.json","index.html"):
        f=output/name;manifest[name]=dict(bytes=f.stat().st_size,sha256=sha(f))
    (output/"file_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    # Check local HTML links and links carried by copied evidence MD.
    missing=[]
    for ref in re.findall(r'(?:href|src)="([^"#]+)"',(output/"index.html").read_text(encoding="utf-8")):
        if not re.match(r"^(?:https?:|data:)",ref) and not (output/ref).exists():missing.append(ref)
    for document in output.rglob("*.md"):
        for ref in re.findall(r'\[[^\]]*\]\(([^)]+)\)',document.read_text(encoding="utf-8-sig")):
            ref=ref.strip("<>").split("#")[0]
            if not ref or re.match(r"^(?:https?:|data:|mailto:)",ref):continue
            if not (document.parent/ref).exists():missing.append(str(document.relative_to(output))+" -> "+ref)
    required={"r0":98,"v44":98,"n2":98}
    incomplete={v["id"]:len(v["images"]) for v in variants if v["id"] in required and len(v["images"])!=required[v["id"]]}
    qa=dict(schema="OFFLINE_REPORT_BUILD_STATIC_QA_V1",output=str(output),files=len(manifest),missing_static_refs=missing,
        incomplete_required_variants=incomplete,optional_variants={v["id"]:len(v["images"]) for v in variants if v["id"] not in required},
        observed_frames=len(observed),result_pooled_mismatches={r["label"]:r["pooled_receipt_mismatches"] for r in results},
        browser_QA_performed=False,published=False)
    (output/"build_qa.json").write_text(json.dumps(qa,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(qa,ensure_ascii=False),flush=True)
    if missing or incomplete:raise RuntimeError("Incomplete offline output; inspect build_qa.json")


if __name__=="__main__":main()

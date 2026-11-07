# -*- coding: utf-8 -*-
"""Build a research-candidate report from saved PNG/JSON only.

No renderer, checkpoint, GPU or model import. Never overwrites an existing
delivery. Root supplies delivery_summary.json and performs the final build/QA.
Example: python build_report_v2.py --run RUN --output RUN/delivery_candidate
Use --validate-only to inspect complete-run selection without creating output.
"""
from pathlib import Path
import argparse
import hashlib
import html
import json
import math
import shutil
import struct

RAW_IDS=list(range(89,138))
VARIANTS=("baseline","extension","calibrated")
LABELS={"baseline":"冻结 V44", "extension":"同一实测补面 / 原材质", "calibrated":"同一补面 / 材质控制"}
PREFERENCE=("full49_corridor_cellfit","full49_corridor_matched","full49_corridor","full49_worldfield")
STAGES={
 "pilot_v1":("R0入口 / 原候选域","4帧入口；后续发现候选域与halo不完整，保留原现场。"),
 "pilot_v2":("V2完整矩形 / 入口","补齐几何候选及halo；不能用入口成功代替49帧保护验收。"),
 "full49_v2":("V2完整矩形 / 原材质","恢复大缺口，但补面也在非road区域接管，保护失败；未采用。"),
 "pilot_clip49":("材质clip控制 / 入口","单列失败或中间控制，不覆盖原材质结果。"),
 "pilot_worldfield_r2":("低cap世界色场 / 入口","11×9世界色场的独立入口，后续全49仍需检查。"),
 "full49_worldfield":("低cap世界色场 / 全49","calibrated仍有4个预注册可见保护门失败；未采用。"),
 "pilot_corridor":("固定world strip / 入口","按原road y-world包络固定active，不使用逐帧图像ROI。"),
 "full49_corridor":("固定world strip / 全49","active仅10098/14661；保护域回到门内，delete原观测road仍+0.0012655，高于原+0.001门；未采用。"),
 "full49_corridor_matched":("matched strip重拟合 / 全49","同scope材质重拟合；delete原观测road约+0.0012693，13门仅12通过；未采用。"),
 "full49_corridor_cellfit":("newroad-only高cap cellfit / 全49","新增材质容量控制。只按实际收据与完整PNG读结果，不预写成功；可见门与未知下体兼容性分开。")}
ORDER=list(STAGES)


def load(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def esc(value):return html.escape(str(value),quote=True)
def fmt(value,digits=7):return "—" if value is None else f"{value:.{digits}f}"
def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def frames_of(receipt):
    frames=receipt.get("frames",[])
    if isinstance(frames,dict):frames=list(frames.values())
    return sorted(frames,key=lambda x:x["raw"])


def pooled(receipt,state,variant,metric):
    # Authoritative sum/count comes from every original frame statistic, not
    # the receipt's rounded pooled mean or an unweighted mean of frame means.
    stats=[f["states"][state][variant][metric] for f in frames_of(receipt)
           if metric in f.get("states",{}).get(state,{}).get(variant,{})]
    if not stats:return None
    count=sum(int(s.get("pixels",s.get("count",0))) for s in stats)
    total=math.fsum(float(s["sum"]) for s in stats)
    return dict(pixels=count,sum=total,mean=total/count if count else None,frames=len(stats))


def gate_rows(receipt,contract,candidate):
    g=contract["preservation_gates"];rows=[]
    for state in ("keep","delete"):
        definitions=[("a6_RGB","max_frame_regression",g["a6_RGB_MAE_max_frame_regression"]),
          ("a6_depth_m","max_frame_regression",g["a6_measured_depth_MAE_max_frame_regression_m"])]
        if state=="keep":definitions.append(("a37_RGB","max_frame_regression",g["a37_keep_RGB_MAE_max_frame_regression"]))
        for metric,prefix in (("observed_road_RGB","original_observed_road_RGB"),("protected_RGB","protected_RGB")):
            definitions.extend([(metric,"max_frame_regression",g[prefix+"_max_frame_regression"]),
                (metric,"pooled_regression",g[prefix+"_pooled_MAE_max_regression"])])
        for metric,kind,threshold in definitions:
            differences=[]
            for f in frames_of(receipt):
                v=f.get("states",{}).get(state,{})
                if metric in v.get(candidate,{}) and metric in v.get("baseline",{}):
                    differences.append((f["raw"],v[candidate][metric]["mean"]-v["baseline"][metric]["mean"]))
            if not differences:continue
            if kind=="max_frame_regression":observed=max(v for _,v in differences)
            else:observed=pooled(receipt,state,candidate,metric)["mean"]-pooled(receipt,state,"baseline",metric)["mean"]
            rows.append(dict(state=state,candidate=candidate,metric=metric,kind=kind,threshold=threshold,
                observed=observed,pass_gate=observed<=threshold+1e-12,
                failed_raws=[r for r,d in differences if d>threshold+1e-12] if kind=="max_frame_regression" else []))
    return rows


def complete_run(run,name,require_three=True):
    path=run/"results"/name/"receipt.json"
    if not path.exists():return False
    a=load(path)
    if a.get("raws")!=RAW_IDS or [f["raw"] for f in frames_of(a)]!=RAW_IDS:return False
    variants=VARIANTS if require_three else tuple(a.get("variants",[]))
    if any(v not in a.get("variants",[]) for v in variants):return False
    for raw in RAW_IDS:
        for state in ("keep","delete"):
            for variant in variants:
                p=path.parent/f"{raw:03d}"/(variant+"_"+state+".png")
                if not p.is_file():return False
                with p.open("rb") as stream:header=stream.read(24)
                if len(header)<24 or header[:8]!=b"\x89PNG\r\n\x1a\n" or struct.unpack(">II",header[16:24])!=(480,320):return False
    return True


def select_primary(run,override=None):
    names=(override,) if override else PREFERENCE
    for name in names:
        if complete_run(run,name):return name
    raise ValueError("No complete49/2-state baseline-extension-calibrated PNG result; preserve pending runs.")


def benchmark_path(run):return run/"benchmark/FRONT_geometry_v2"


def compact_metric_cell(x):
    if not x:return "<td>—</td>"
    return f'<td><strong>{fmt(x["mean"])}</strong><small>sum {x["sum"]:.12g} / {x["pixels"]:,}</small></td>'


def architecture():
    return '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 350" role="img" aria-label="R0 measured road integration and separately isolated R1 benchmark"><defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#45677b"/></marker></defs><style>text{font:17px 'Microsoft YaHei',sans-serif;fill:#17394e}.s{font-size:14px}.b{fill:#edf5ef;stroke:#789585;stroke-width:1.4}.n{fill:#f1f4f9;stroke:#8fa3b2;stroke-width:1.4}.a{fill:none;stroke:#45677b;stroke-width:2;marker-end:url(#arr)}</style><rect class="n" x="15" y="25" width="190" height="85" rx="8"/><text x="110" y="58" text-anchor="middle">FRONT RGB / LiDAR</text><text class="s" x="110" y="86" text-anchor="middle">199时刻 · 来源与未知分开</text><rect class="b" x="250" y="25" width="235" height="85" rx="8"/><text x="367" y="58" text-anchor="middle">R0 实测世界表面</text><text class="s" x="367" y="86" text-anchor="middle">补齐候选域 → 固定strip</text><rect class="b" x="530" y="25" width="235" height="85" rx="8"/><text x="647" y="58" text-anchor="middle">共享道路材质</text><text class="s" x="647" y="86" text-anchor="middle">原色 / 世界场 / cellfit</text><rect class="b" x="810" y="25" width="295" height="85" rx="8"/><text x="957" y="58" text-anchor="middle">冻结 C2/SF · V44联合排序</text><text class="s" x="957" y="86" text-anchor="middle">keep/delete只移除a37 · 真实PNG</text><rect class="n" x="250" y="215" width="235" height="85" rx="8"/><text x="367" y="248" text-anchor="middle">R1 世界patch + guard</text><text class="s" x="367" y="276" text-anchor="middle">外观盲 / 几何＋RGB盲</text><rect class="n" x="530" y="215" width="235" height="85" rx="8"/><text x="647" y="248" text-anchor="middle">合法CPU基线</text><text class="s" x="647" y="276" text-anchor="middle">独立进程 · 51/199帧预算</text><rect class="n" x="810" y="215" width="295" height="85" rx="8"/><text x="957" y="248" text-anchor="middle">预测冻结 → 外部评价</text><text class="s" x="957" y="276" text-anchor="middle">封存参考 / 分母 / 开封界限</text><path class="a" d="M205 67H250M485 67H530M765 67H810M205 90H226V257H250M485 257H530M765 257H810"/><text class="s" x="560" y="337" text-anchor="middle">R0的旧场景、道路、环境与checkpoints不进入R1估计器；透明Gaussian合成不是完整MOG不透明首交。</text></svg>'''


def make_plots(out,patches,budget):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    figures=out/"figures";figures.mkdir(exist_ok=True)
    names=[]
    if patches:
        fig,ax=plt.subplots(figsize=(10.5,4.2))
        for p in patches["patches"]:
            x0,y0,x1,y1=p["xy_bounds_local_m"];g=p["guard_xy_bounds_local_m"]
            color="#36648b" if p["split"]=="validation" else "#b16238"
            ax.add_patch(Rectangle((g[0],g[1]),g[2]-g[0],g[3]-g[1],fill=False,linestyle="--",edgecolor=color,alpha=.55))
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor=color,alpha=.65))
            ax.annotate(f'{p["size_m"]:g}m',xy=((x0+x1)/2,y1),xytext=(0,7),textcoords="offset points",ha="center",va="bottom",fontsize=8,color=color)
        ax.autoscale_view();ax.set_aspect("equal");ax.margins(.12);ax.grid(alpha=.2)
        ax.set(xlabel="Anchor-local x (m), fixed world gauge",ylabel="Anchor-local y (m)",title="Geometry-only fixed patches: blue validation / orange test; dashed 0.75m guard")
        fig.tight_layout();fig.savefig(figures/"r1_patch_layout.png",dpi=160);fig.savefig(figures/"r1_patch_layout.svg");plt.close(fig);names.append("r1_patch_layout")
    if budget:
        rows=[x for x in budget["comparisons"] if x["track"]=="appearance_blind"]
        fig,axes=plt.subplots(1,2,figsize=(10.5,4.2))
        for i,r in enumerate(rows):
            a,c=r["budget51"],r["budget199"];label=r["split"].replace("sealed_test","opened test")
            axes[0].bar(i-.16,a["RGB_supported_target_cells"],width=.30,color="#9caebb",label="51 frames" if i==0 else None)
            axes[0].bar(i+.16,c["RGB_supported_target_cells"],width=.30,color="#36648b",label="199 frames" if i==0 else None)
            axes[0].text(i,max(a["RGB_supported_target_cells"],c["RGB_supported_target_cells"])+18,f'/ {a["target_cells"]}',ha="center",fontsize=9)
            axes[1].plot([51,199],[r["common_denominator_RGB_MAE_51"],r["common_denominator_RGB_MAE_199"]],marker="o",label=f'{label}, same {r["common_supported_stable_cells"]} cells')
        axes[0].set(xticks=range(len(rows)),xticklabels=[r["split"].replace("sealed_test","opened test") for r in rows],ylabel="RGB-supported target cells",title="Coverage, same in both tracks");axes[0].legend()
        axes[1].set(xticks=[51,199],xlabel="Fixed input-frame budget",ylabel="Linear-RGB MAE",title="Common support denominator only");axes[1].legend(fontsize=8)
        for ax in axes:ax.grid(alpha=.2,axis="y")
        fig.suptitle("Full199 is a same-scene development control after test opening",fontsize=11)
        fig.tight_layout();fig.savefig(figures/"r1_budget_control.png",dpi=160);fig.savefig(figures/"r1_budget_control.svg");plt.close(fig);names.append("r1_budget_control")
    return names


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--primary",default=None);ap.add_argument("--validate-only",action="store_true");ap.add_argument("--skip-plots",action="store_true")
    args=ap.parse_args();run=args.run.resolve();out=args.output.resolve()
    primary=select_primary(run,args.primary);contract=load(run/"contract.json")
    receipts={p.parent.name:load(p) for p in sorted((run/"results").glob("*/receipt.json"))}
    final=receipts[primary];main_gates=gate_rows(final,contract,"calibrated")
    wide="full49_v2" if complete_run(run,"full49_v2",False) else None
    benchmark=benchmark_path(run)
    budget_path=benchmark/"baseline_runs/full199_plane_boundary_median_development_v1/public_summary.json"
    if not budget_path.exists():budget_path=run/"benchmark/source/public_summary_full199.json"
    budget=load(budget_path) if budget_path.exists() else None
    patch_path=benchmark/"public/patch_manifest.json";patches=load(patch_path) if patch_path.exists() else None
    summary_path=run/"delivery_summary.json";narrative=load(summary_path) if summary_path.exists() else {}
    body_paths=sorted((run/"review").glob("*inferred_asset_contribution.json"))
    body_path=next((p for p in body_paths if p.name.startswith(primary)),body_paths[-1] if body_paths else None)
    body=load(body_path) if body_path else None
    plan=dict(primary=primary,raws=RAW_IDS,states=["keep","delete"],variants=list(VARIANTS),primary_saved_PNGs=294,
      wide_failedcase=wide,observed_protection_pass=sum(x["pass_gate"] for x in main_gates),observed_protection_total=len(main_gates),
      research_candidate=True,scene_adopted_by_builder=False,root_summary_present=summary_path.exists(),R1_full199_summary_present=budget is not None,
      lower_body_diagnostic=str(body_path) if body_path else None,lower_body_extra_acceptance_required=True,
      output=str(out),old_delivery_overwritten=False,renderer_or_checkpoint_loaded=False)
    plan["pooled_denominators"]={state:{metric:pooled(final,state,"baseline",metric)["pixels"] for metric in ("a6_RGB","a6_depth_m","observed_road_RGB","protected_RGB")} for state in ("keep","delete")}
    if args.validate_only:
        print(json.dumps(plan,ensure_ascii=False,indent=2));return
    if budget is None:raise ValueError("R1 full199 public summary is required; do not publish fixed claims without evidence")
    out.mkdir(parents=True,exist_ok=False)
    copy_records=[]
    def copy(source,relative):
        source=Path(source);target=out/relative
        if source.is_symlink():raise ValueError("Do not package symlink dependencies")
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        digest=sha(source)
        if sha(target)!=digest:raise ValueError("Report copy byte parity failure")
        copy_records.append(dict(source=str(source),path=target.relative_to(out).as_posix(),sha256=digest,bytes=target.stat().st_size))
        return target.relative_to(out).as_posix()
    evidence=out/"evidence";evidence.mkdir()
    for name in ("contract.json","contract_v2_lineage.json","delivery_summary.json"):
        if (run/name).exists():copy(run/name,Path("evidence")/name)
    for name in receipts:copy(run/"results"/name/"receipt.json",Path("evidence")/(name+".json"))
    # Explicit report allowlist: no NPZ, checkpoints, raw RGB/LiDAR, isolated
    # estimator sources, sealed GT, or original source features/atlases.
    allowed={".py",".md",".svg",".json"}
    for directory in ("source","surface","observations","coverage","review"):
        base=run/directory
        if not base.exists():continue
        for p in sorted(base.rglob("*")):
            if not p.is_file() or p.is_symlink():continue
            relative=p.relative_to(run)
            if p.suffix in allowed or (directory=="review" and p.suffix.lower() in (".png",".jpg",".jpeg")):
                copy(p,Path("repro")/relative)
    bsrc=run/"benchmark/source"
    if bsrc.exists():
        for p in sorted(bsrc.iterdir()):
            if p.is_file() and p.suffix in allowed:copy(p,Path("repro/benchmark/source")/p.name)
    for p in sorted((benchmark/"audit").glob("*.json")) if (benchmark/"audit").exists() else []:
        if p.name in ("preparation_receipt.json","selection_receipt.json","isolated_export.json","full199_budget_extension_verification.json","baseline_input_preparation.json","baseline_input_preparation_full199.json"):
            copy(p,Path("repro/benchmark/audit")/p.name)
    if patch_path.exists():copy(patch_path,Path("repro/benchmark/patch_manifest.json"))
    for p in sorted((benchmark/"baseline_runs").glob("*/*.json")) if (benchmark/"baseline_runs").exists() else []:
        copy(p,Path("repro/benchmark/baseline_runs")/p.relative_to(benchmark/"baseline_runs"))
    index={};wide_index={}
    for raw in RAW_IDS:
        index[str(raw)]={};wide_index[str(raw)]={}
        src=run/"results"/primary/f"{raw:03d}"
        for state in ("keep","delete"):
            index[str(raw)][state]={}
            for v in VARIANTS:
                name=v+"_"+state+".png";index[str(raw)][state][v]=copy(src/name,Path("frames")/f"{raw:03d}"/name)
            if wide:
                name="extension_"+state+".png"
                wide_index[str(raw)][state]=copy(run/"results"/wide/f"{raw:03d}"/name,Path("wide_failedcase")/f"{raw:03d}"/name)
        for name in ("observed.png","extension_solo_alpha.png","extension_solo_rgb.png","metrics.json"):
            if (src/name).exists():copy(src/name,Path("frames")/f"{raw:03d}"/name)
    write_json(out/"frame_index.json",index);write_json(out/"wide_frame_index.json",wide_index)
    stage_data=[];stage_html=[]
    for name in sorted(receipts,key=lambda n:(ORDER.index(n) if n in ORDER else len(ORDER),n)):
        a=receipts[name];candidate="calibrated" if "calibrated" in a.get("variants",[]) else "extension"
        gates=gate_rows(a,contract,candidate);delta=None
        if pooled(a,"delete",candidate,"observed_road_RGB"):
            delta=pooled(a,"delete",candidate,"observed_road_RGB")["mean"]-pooled(a,"delete","baseline","observed_road_RGB")["mean"]
        title,note=STAGES.get(name,(name,"额外保存对照；按原收据显示。"))
        stage_data.append(dict(run=name,frames=len(a.get("raws",[])),candidate=candidate,delete_observed_road_RGB_delta=delta,
                              gates=gates,title=title,note=note))
        completed=complete_run(run,name,require_three=candidate=="calibrated") if len(a.get("raws",[]))==49 else False
        stage_html.append(f'<tr><td><a href="evidence/{esc(name)}.json">{esc(title)}</a><small>{esc(name)}</small></td><td>{len(a.get("raws",[]))}帧'+(" · PNG齐全" if completed else " · pilot/未齐")+f'</td><td>{esc(candidate)}</td><td>{fmt(delta)}</td><td>{sum(g["pass_gate"] for g in gates)}/{len(gates)}<small>已观测门，非采用</small></td><td>{esc(note)}</td></tr>')
    if "full49_corridor_cellfit" not in receipts:
        stage_html.append('<tr><td>newroad-only高cap cellfit</td><td colspan="4">尚无完整49帧结果；不预写GPU质量结论</td><td>CPU/训练/恢复receipt单独保留。最终运行时若完整PNG与收据齐备会自动选入。</td></tr>')
    write_json(out/"data/stage_summary.json",stage_data);write_json(out/"data/observed_protection_gates.json",main_gates)
    metric_rows=[]
    for state in ("keep","delete"):
        metrics=[("a6_RGB","a6原可见 RGB"),("a6_depth_m","a6实测深度 / m"),("observed_road_RGB","原观测道路 RGB"),("protected_RGB","原保护域 RGB")]
        if state=="keep":metrics.append(("a37_RGB","a37原可见 RGB"))
        for metric,label in metrics:
            metric_rows.append(f'<tr><td>{state}</td><td>{label}</td>'+"".join(compact_metric_cell(pooled(final,state,v,metric)) for v in VARIANTS)+"</tr>")
    gate_html="".join(f'<tr><td>{g["state"]}</td><td>{esc(g["metric"])}</td><td>{esc(g["kind"])}</td><td>{fmt(g["observed"])}</td><td>≤{fmt(g["threshold"])}</td><td class="{"pass" if g["pass_gate"] else "fail"}">{"PASS" if g["pass_gate"] else "FAIL"}</td><td>{esc(g.get("failed_raws",[]))}</td></tr>' for g in main_gates)
    f137=next(f for f in frames_of(final) if f["raw"]==137)
    mechanism=[]
    for v in VARIANTS:
        q=f137["states"]["delete"][v].get("domains",{}).get("fixed720",{})
        mechanism.append(f'<tr><td>{esc(LABELS[v])}</td>'+"".join(compact_metric_cell(q.get(m)) for m in ("road_alphaT","environment_T","extension_alphaT"))+"</tr>")
    body_html='<p class="warning">未观测车身 / 地面兼容性尚未据此通过。资产及行hash冻结不保证alpha×T贡献不变，也不证明隐藏车身或道路正确。</p>'
    if body:
        body_html+=f'<p>诊断来源：{esc(Path(body.get("results","")).name)}；实际raw {esc(body.get("raws",[]))}。它不是全49隐藏表面的GT评价。</p>'
        body_html+='<table><tr><th>冻结baseline assetW阈值</th><th>像素分母</th><th>最大贡献损失</th><th>损失&gt;.1</th><th>RGB改变量&gt;.02</th></tr>'
        for r in body.get("totals",[]):
            body_html+=f'<tr><td>&gt;{r["baseline_assetW_threshold"]}</td><td>{r["pixels"]:,}</td><td>{fmt(r.get("max_assetW_loss"))}</td><td>{r.get("counts_loss_gt",{}).get("0.1","—")}</td><td>{r.get("RGB_changed_gt002","—")}</td></tr>'
        body_html+='</table>'
    body_html+='<p>raw130的35个原assetW&gt;.9、贡献损失&gt;.01像素（其中18个损失&gt;.5）全在原a37可见域，距原a6 reviewed边缘约11–14px；已有5个确测遮挡、30个unknown，0个free/原观测road。不能把原RGB当作隐车或道路GT。几何物理高度归因另列，不由图像改变量单独决定。</p>'
    body_images=[]
    for raw in (130,137):
        p=run/"review/asset_ground_conflict"/f"raw{raw}_a6_crop_and_loss.png"
        if p.exists():body_images.append(f'<figure><figcaption>raw{raw} · 推断下体与道路贡献诊断</figcaption><a href="repro/review/asset_ground_conflict/{p.name}"><img src="repro/review/asset_ground_conflict/{p.name}" alt="真实保存的下体贡献诊断"></a></figure>')
    body_html+='<div class="bodyboard">'+"".join(body_images)+'</div>'
    extra_reviews=sorted((run/"review/asset_ground_conflict").glob("*.json")) if (run/"review/asset_ground_conflict").exists() else []
    body_html+='<p>'+" · ".join(f'<a href="repro/{p.relative_to(run).as_posix()}">{esc(p.name)}</a>' for p in extra_reviews)+'</p>'
    r1_rows=[]
    if budget:
        for r in budget["comparisons"]:
            a,c=r["budget51"],r["budget199"]
            r1_rows.append(f'<tr><td>{esc(r["track"])}</td><td>{esc(r["split"])}</td><td>{a["RGB_supported_target_cells"]}→{c["RGB_supported_target_cells"]} / {c["target_cells"]}</td><td>{c["RGB_scored_cells"]}/{c["stable_reference_cells"]}</td><td>{fmt(c["conditional_linear_RGB_MAE"])}</td><td>{fmt(c["measured_height_MAE_m"])}, {c["measured_height_scored_returns"]}/{c["measured_reference_returns"]} returns</td><td>{fmt(r["common_denominator_RGB_MAE_51"])}→{fmt(r["common_denominator_RGB_MAE_199"])}<small>同{r["common_supported_stable_cells"]}格</small></td></tr>')
    plots=[]
    if not args.skip_plots:plots=make_plots(out,patches,budget)
    (out/"architecture.svg").write_text(architecture(),encoding="utf-8")
    r1_figures="".join(f'<figure><a href="figures/{n}.svg"><img src="figures/{n}.png" alt="R1固定patch与预算科学图"></a></figure>' for n in plots)
    panels="".join(f'<figure id="panel_{v}"><figcaption>{esc(LABELS[v])}</figcaption><a id="link_{v}" href="{index["137"]["delete"][v]}"><img id="img_{v}" src="{index["137"]["delete"][v]}" alt="真实保存PNG：{esc(v)}"></a></figure>' for v in VARIANTS)
    if wide:panels+=f'<figure id="panel_wide" hidden><figcaption>失败scope对照 · V2完整矩形 / 原材质</figcaption><a id="link_wide" href="{wide_index["137"]["delete"]}"><img id="img_wide" src="{wide_index["137"]["delete"]}" alt="wide failedcase真实PNG"></a></figure>'
    panels+='<figure id="panel_observed" hidden><figcaption>原始实拍 · 含a37，不是删后隐藏GT</figcaption><a id="link_observed" href="frames/137/observed.png"><img id="img_observed" src="frames/137/observed.png" alt="原始实拍"></a></figure>'
    details="".join('<p>'+esc(x)+'</p>' for x in narrative.get("details",[]) if isinstance(x,(str,int,float)))
    headline=esc(narrative.get("headline","道路观测补面已恢复覆盖；材质保护与推断下体兼容性分别验收"))
    decision=esc(narrative.get("decision","研究候选，未自动替换或宣称整场景采用。最终结论由根的delivery_summary.json补入。"))
    root_summary_notice='' if narrative else '<p class="warning">根的最终delivery_summary.json尚未生成；本页保留研究候选状态，不编造最终判定。</p>'
    denominator_text="；".join(f'{state}: '+", ".join(f'{k}={v:,}' for k,v in values.items()) for state,values in plan["pooled_denominators"].items())
    page=TEMPLATE
    replacements={"HEADLINE":headline,"DECISION":decision,"SUMMARY_NOTICE":root_summary_notice,"PRIMARY":esc(primary),
       "PANELS":panels,"METRIC_ROWS":"".join(metric_rows),"GATES":gate_html,"PASSCOUNT":str(sum(x["pass_gate"] for x in main_gates)),
       "GATECOUNT":str(len(main_gates)),"STAGES":"".join(stage_html),"MECHANISM":"".join(mechanism),"BODY":body_html,
       "DETAILS":details,"R1_ROWS":"".join(r1_rows),"R1_FIGURES":r1_figures,"DENOMINATORS":esc(denominator_text),
       "INDEX":json.dumps(index).replace("<","\\u003c"),"WIDE_INDEX":json.dumps(wide_index).replace("<","\\u003c"),"HAS_WIDE":"true" if wide else "false"}
    for key,value in replacements.items():page=page.replace("@@"+key+"@@",value)
    if "@@" in page:raise ValueError("Unfilled report template token")
    (out/"index.html").write_text(page,encoding="utf-8")
    write_json(out/"build_receipt.json",dict(**plan,stage_count=len(receipts),plots=plots,primary_receipt_sha256=sha(run/"results"/primary/"receipt.json"),
        root_summary_sha256=sha(summary_path) if summary_path.exists() else None,sum_count_recomputed_from_frame_statistics=True,
        synthetic_or_generated_scene_images=False,raw_or_GT_inputs_packaged=False,learned_road_prior_success_claim=False,
        full_MOG_first_hit_claim=False,generalization_claim=False,lower_body_diagnostic_is_hidden_GT=False))
    write_json(out/"copy_manifest.json",copy_records)
    write_json(out/"manifest.json",{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob("*")) if p.is_file()})
    print(json.dumps(dict(output=str(out),primary=primary,files=len(copy_records),visible_gate_pass=sum(x["pass_gate"] for x in main_gates),scene_adopted=False),ensure_ascii=False))


TEMPLATE='''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scene008 · R0/R1 research candidate</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f2f5f8;color:#213a4e;font:16px/1.8 'Microsoft YaHei',system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:28px}h1{font-size:30px;line-height:1.5}h2{font-size:23px;line-height:1.5}section{background:white;border:1px solid #dce5ec;border-radius:10px;margin:20px 0;padding:22px}a{color:#12648d}p{margin:10px 0}small,.meta{display:block;font-size:13px;color:#62798b}.lead{font-size:19px}.badge{display:inline-block;background:#fff0c9;color:#72541b;padding:3px 12px;border-radius:20px}.warning{background:#fff4dd;border-left:4px solid #ae7f31;padding:13px}.tablewrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.65}th,td{border-bottom:1px solid #dfe6ec;padding:9px;vertical-align:top;text-align:left}th{background:#edf3f8;white-space:nowrap}td strong{font-variant-numeric:tabular-nums}td small{white-space:nowrap}.pass{color:#35714b}.fail{color:#a74435;font-weight:700}.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:15px 0}button,select{font:inherit;padding:4px 10px}input[type=range]{width:260px}.board{display:grid;grid-template-columns:repeat(var(--cols,3),minmax(0,1fr));gap:12px}figure{margin:0}figure[hidden]{display:none}figure img{display:block;width:100%;height:auto}figcaption{font-size:14px;font-weight:650;margin:4px 0}.bodyboard{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}.plotboard{display:grid;gap:20px}#architecture{width:100%;height:auto}details{margin:14px 0}summary{cursor:pointer;font-weight:650}#loadStatus{font-size:13px;color:#62798b}@media(max-width:850px){main{padding:12px}section{padding:15px}.board,.bodyboard{grid-template-columns:1fr!important}h1{font-size:25px}table{font-size:12px}th,td{padding:6px}}</style>
</head>
<body>
<main>
<p class="meta">Scene008 · 2026-11-07 · R0实测恢复 / R1条件控制 · PNG与JSON重放</p>
<span class="badge">Research candidate · 可见门通过不等于整场景采用</span>
<h1>@@HEADLINE@@</h1>
<p class="lead">@@DECISION@@</p>@@SUMMARY_NOTICE@@<section>
<h2>输入、组件与数据流</h2>
<img id="architecture" src="architecture.svg" alt="Architecture components: real sensor input, measured road, material controls, native sorting and isolated benchmark">
</section>
<section id="viewer">
<h2>完整49帧 · 双状态真实PNG</h2>
<p>主运行 <strong>@@PRIMARY@@</strong>。baseline / extension / calibrated代码字段保留。所有列共用相机、轨迹和冻结C2/SurfaceLOD；delete只移除a37，道路不按编辑状态临时挖洞。点击图像查看无损PNG。</p>
<div class="controls">
<label>raw <input id="raw" type="range" min="89" max="137" value="137">
</label>
<strong id="rawLabel">137</strong>
<select id="state">
<option value="delete">删除a37</option>
<option value="keep">保留a37</option>
</select>
<button data-r="105">105</button>
<button data-r="113">113</button>
<button data-r="130">130 下体</button>
<button data-r="137">137 大缺口</button>
<label>
<input id="showWide" type="checkbox">wide失败scope</label>
<label>
<input id="showObserved" type="checkbox">原始实拍</label>
<span id="loadStatus">已保存raw137真实PNG</span>
</div>
<div class="board" id="board">@@PANELS@@</div>
<p class="meta">原实拍含a37；它既不是删除后的隐藏道路GT，也不是被遮挡a6下体的GT。当前是Gaussian期望深度/透明度排序，非不透明首交。</p>
</section>
<section>
<h2>raw137大缺口：覆盖机制已恢复</h2>
<p>固定720像素、delete状态：原道路贡献为0，补面后约0.974；环境透射约0.210降至0.00036。下面逐项保留sum/count。高覆盖证明成像接管，不能单独证明隐藏几何或材质正确。</p>
<div class="tablewrap">
<table>
<tr>
<th>方案</th>
<th>joint road alpha×T ↑</th>
<th>环境T ↓</th>
<th>新补面alpha×T ↑</th>
</tr>@@MECHANISM@@</table>
</div>
</section>
<section>
<h2>所有阶段与失败scope控制</h2>
<p>V1→V2是候选域/halo完整性修复；世界色场、strip、matched与cellfit是单列控制。预注册保护阈值没有为了结果放宽。pilot与full49分别标记。</p>
<div class="tablewrap">
<table>
<tr>
<th>阶段 / 收据</th>
<th>实际帧数</th>
<th>被验方案</th>
<th>delete原观测road pooled Δ</th>
<th>可见保护门</th>
<th>范围与保留结论</th>
</tr>@@STAGES@@</table>
</div>
</section>
<section>
<h2>主运行：固定观测分母与sum/count</h2>
<p>对每帧原始误差sum与像素/点count重新合并，未平均49个帧均值。本运行从实际sum/count统计得到的精确分母：@@DENOMINATORS@@。原观测road含共同参考；删后proxy与始终未知区不冒充独立GT。</p>
<div class="tablewrap">
<table>
<tr>
<th>状态</th>
<th>指标 ↓</th>
<th>冻结V44</th>
<th>实测补面 / 原材质</th>
<th>同补面 / 材质控制</th>
</tr>@@METRIC_ROWS@@</table>
</div>
<details open>
<summary>预注册已观测保护门：@@PASSCOUNT@@ / @@GATECOUNT@@</summary>
<p class="warning">即使13/13通过，也只覆盖这里声明的已观测域；不涵盖未观测推断车身与地面的物理兼容性，不能据此采用整个场景。</p>
<div class="tablewrap">
<table>
<tr>
<th>状态</th>
<th>指标</th>
<th>门类型</th>
<th>实际退化</th>
<th>原阈值</th>
<th>判定</th>
<th>失败raw</th>
</tr>@@GATES@@</table>
</div>
</details>
</section>
<section id="body-risk">
<h2>额外验收：推断a6下体贡献发生改变</h2>@@BODY@@</section>
<section id="R1">
<h2>R1：FRONT-only世界patch盲化与合法CPU基线</h2>
<p>6个固定0.5/1/2m世界patch，validation/test各3个，guard 0.75m。选择只用重新计算的几何支持和空间散列；不读旧source_valid（包含全图sharpness）、目标颜色或拟合误差。所有199个FRONT源原生JPEG在resize/特征提取前屏蔽，原始5 LiDAR×2return在融合前按棱柱过滤。其他四个RGB相机未审、未入条件。</p>
<p>appearance-blind允许目标实测XYZ、移除RGB/强度；geometry＋RGB-blind另移除目标＋guard内XYZ。mask使用context高度区间，非目标GT高程轮廓。旧R0/V44资产、road/env/checkpoints与sealed参考不进入估计器。每patch/condition是独立Python -S进程，精确文件allowlist＋hash＋真实读取日志；拒绝读取sealed manifest的探针通过，但<strong>未认证OS强隔离或完整rolling-shutter物理边界</strong>。</p>
<p>最小51帧来自代理为CPU入口设定的stride4预算（0,4,…196及198），不是用户科学合同或合法上下文上限。全199控制复用完全相同估计器、方法、阈值、patch和guard，510项重叠输入核验一致；未依据颜色或已开封误差挑选新增source。全199是<strong>51帧测试开封后的same-scene开发控制，不是独立确认</strong>。</p>
<div class="plotboard">@@R1_FIGURES@@</div>
<div class="tablewrap">
<table>
<tr>
<th>条件</th>
<th>split</th>
<th>51→199 RGB支持 / 全目标</th>
<th>199可评分 / 稳定参考</th>
<th>199条件linear-RGB MAE</th>
<th>199实测高度MAE / m</th>
<th>共同分母RGB：51→199</th>
</tr>@@R1_ROWS@@</table>
</div>
<p>全199仅727/1647目标格有局部RGB支持。2m test与1m validation仍unsupported，范围仅限当前FRONT、4m局部算法、guard及资格门；不能说所有199或其他相机没有合法上下文。test条件RGB MAE只计125/673稳定参考格，不能当整块或整条道路准确率。0.5m坡度proxy误差仍约4.2–4.56°，高度均值不能掩盖局部结构不足。</p>
<p>封存参照是实际合格radiance及稀疏LiDAR；不是独立隐藏albedo/normal真值。标线GT缺失，不报标线准确率。未跑RBF/spline或预训练道路生成先验；未来prior收益须使用匹配全199预算及支持分母，独立确认须新untouched patches。</p>
<p>
<a href="repro/benchmark/baseline_runs/full199_plane_boundary_median_development_v1/public_summary.json">51/199公共预算摘要</a> · <a href="repro/benchmark/source/benchmark_contract.json">完整基准合同</a> · <a href="repro/benchmark/patch_manifest.json">固定patch manifest</a> · <a href="repro/benchmark/audit/preparation_receipt.json">199输入准备验收</a>
</p>
</section>
<section>
<h2>根的最终整理与科学边界</h2>@@DETAILS@@<p class="warning">R0是同场景真实观测恢复及材质控制，不是生成road-prior成功；当前合成不是完整MOG不透明首交，也没有跨场景泛化结论。保存资产冻结、道路alpha提高或13可见门通过，均不能替代未知车身/地面的额外验收。</p>
<p>
<a href="evidence/delivery_summary.json">根最终delivery_summary</a> · <a href="evidence/contract.json">原预注册门</a> · <a href="data/stage_summary.json">所有阶段精确数据</a> · <a href="data/observed_protection_gates.json">主运行13门复算</a> · <a href="frame_index.json">49×2×3 PNG映射</a> · <a href="copy_manifest.json">来源/哈希</a> · <a href="build_receipt.json">构建scope</a> · <a href="repro/source/r0_runtime.py">道路运行时</a>
</p>
<p class="meta">本页只重放保存结果，不加载模型、不触发训练、渲染或电源操作。原已交付包不覆盖。</p>
</section>
</main>
<script>const frameIndex=@@INDEX@@;const wideIndex=@@WIDE_INDEX@@;const hasWide=@@HAS_WIDE@@;const variants=['baseline','extension','calibrated'];let drawGeneration=0;const wideToggle=document.getElementById('showWide');wideToggle.disabled=!hasWide;async function draw(){const generation=++drawGeneration;const raw=document.getElementById('raw').value,state=document.getElementById('state').value;const wide=hasWide&&wideToggle.checked,observed=document.getElementById('showObserved').checked;document.getElementById('rawLabel').textContent=raw;document.getElementById('loadStatus').textContent='加载raw'+raw+' '+state;const paths=variants.map(v=>[v,frameIndex[raw][state][v]]);if(wide)paths.push(['wide',wideIndex[raw][state]]);if(observed)paths.push(['observed','frames/'+String(raw).padStart(3,'0')+'/observed.png']);try{await Promise.all(paths.map(([v,p])=>new Promise((resolve,reject)=>{const image=new Image();image.onload=resolve;image.onerror=()=>reject(new Error(p));image.src=p})));if(generation!==drawGeneration)return;paths.forEach(([v,p])=>{document.getElementById('img_'+v).src=p;document.getElementById('link_'+v).href=p});if(hasWide)document.getElementById('panel_wide').hidden=!wide;document.getElementById('panel_observed').hidden=!observed;document.getElementById('board').style.setProperty('--cols',3+Number(wide)+Number(observed));document.getElementById('loadStatus').textContent='真实PNG · raw'+raw+' · '+state}catch(error){if(generation===drawGeneration)document.getElementById('loadStatus').textContent='PNG加载失败：'+error.message}}document.getElementById('raw').oninput=draw;document.getElementById('state').onchange=draw;wideToggle.onchange=draw;document.getElementById('showObserved').onchange=draw;document.querySelectorAll('button[data-r]').forEach(b=>b.onclick=()=>{document.getElementById('raw').value=b.dataset.r;draw()});</script>
</body>
</html>'''


if __name__=="__main__":main()

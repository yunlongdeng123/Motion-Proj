from pathlib import Path
import json,html
r=Path("work/v43_sim_mechanism_20261030/source_control_r1");s=json.loads((r/"fresh_comparison_summary.json").read_text())
def f(x):return f"{x:.5f}"
labels={"old_native":"旧 native delete","new_native":"新 native delete","old_M1":"旧同规则 M1 delete","new_M1":"新同规则 M1 delete"}
rows=[]
for key,label in labels.items():
 vals=[]
 for frame in s["frames"]:
  x=frame["regions"][key];vals.append(f'<td>{f(x["raw_mae"])} / {f(x["clamped_mae"])}</td>')
 rows.append("<tr><th>"+label+"</th>"+"".join(vals)+"</tr>")
sel=[]
for frame in s["frames"]:
 x=frame["new_M1"];a=x["original_body_selection_audit"]
 sel.append(f'<tr><th>{frame["frame"]}</th><td>{x["selected_rows"]:,}</td><td>{a["selected_static_body_coverage_fraction"]*100:.4f}%</td><td>{a["static_original_body_alphaT"]:.3f}</td><td>{x["revealed_road_retained_pre_alphaT"]:.5f} → {x["revealed_road_retained_post_alphaT"]:.3f}</td></tr>')
gallery=[]
for frame in s["frames"]:
 fid=frame["frame"];cards=[]
 for key,label in labels.items():
  final=f"evidence/source/{key}/{fid}_final.png";white=f"evidence/source/{key}/{fid}_white.png";cards.append(f'<figure><figcaption>{label}</figcaption><a href="{final}"><img loading="lazy" src="{final}" data-final="{final}" data-white="{white}" alt="{fid} {label}"></a></figure>')
 cards.append(f'<figure><figcaption>独立无车参考</figcaption><a href="evidence/source/independent_reference/{fid}.png"><img loading="lazy" src="evidence/source/independent_reference/{fid}.png" alt="独立无车参考"></a></figure>')
 gallery.append(f'<section class="anchor" id="frame-{fid}"><h3>共同锚点 {fid}</h3><div class="gallery">{"".join(cards)}</div></section>')
bincolors=["#8c2d04","#d94801","#f16913","#fdae6b","#d9d9d9"];bars=[]
for key,label in (("old","旧 M1 / 旧16view"),("new","新 M1 / 新16view")):
 x=0;pieces=[]
 for i,b in enumerate(s["visibility"][key]["bins"]):
  width=b["postdelete_body_alphaT_fraction"]*100;pieces.append(f'<span style="width:{width}%;background:{bincolors[i]}" title="{html.escape(b["bin"])}: {width:.2f}%"></span>')
 bars.append(f'<div class="barrow"><b>{label}</b><div class="bar">{"".join(pieces)}</div><strong>低于 .1：{s["visibility"][key]["alphaT_weak_lt01"]*100:.2f}%</strong></div>')
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fresh 10K：同口径来源比较</title><style>
:root{{font-family:system-ui,"Microsoft YaHei",sans-serif;color:#19242c;background:#f4f5f6}}body{{margin:0}}main{{max-width:1580px;margin:auto;padding:30px}}h1{{font-size:29px}}h2{{font-size:21px;margin-top:34px}}h3{{font-size:17px}}p{{max-width:1120px;line-height:1.75}}.lead{{font-size:18px}}.card,.anchor{{background:white;border:1px solid #dfe4e8;border-radius:10px;padding:20px;margin:18px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:11px;border-bottom:1px solid #dce2e6;text-align:left}}thead{{background:#eef2f4}}.gallery{{display:grid;grid-template-columns:repeat(5,1fr);gap:9px}}figure{{margin:0;min-width:0}}figcaption{{font-size:12px;font-weight:600;padding-bottom:8px}}img{{width:100%;display:block;border:1px solid #dde2e6}}a{{color:#155a85}}.muted{{color:#5d6872;font-size:13px}}button{{border:1px solid #8e9ca7;padding:7px 13px;background:white;cursor:pointer;margin-right:8px}}.barrow{{display:grid;grid-template-columns:180px 1fr 180px;gap:12px;align-items:center;margin:15px 0;font-size:13px}}.bar{{height:24px;display:flex}}.legend span{{display:inline-block;width:12px;height:12px;margin-left:16px;margin-right:4px}}svg{{width:100%;height:auto;max-height:150px}}@media(max-width:1000px){{main{{padding:15px}}.gallery{{grid-template-columns:repeat(2,1fr)}}.barrow{{grid-template-columns:1fr}}table{{font-size:12px}}}}
</style><main><p class="muted">V43 · Fresh 10K 共同锚点来源复查 · 仅冻结诊断，0额外训练更新</p><h1>更充分观测后，原分支删除仍失败；同规则 M1 删除 ROI 误差改善但残影仍在</h1>
<p class="lead">原 native delete 在中、末锚点的独立道路误差变差。按同一真实 αT 阈值选择静态行的 M1 对照，其删除操作在三个锚点的 revealed-road ROI 误差均低于旧 M1，但新中帧图像仍有明显红色车形残影和路面块状、条纹伪影。新 M1 的 shift / rotate 未在本小节验证，不据此推广为其他编辑操作改善。原生分支路由与 M1 是两种协议，分别比较。</p>
<div class="card"><svg viewBox="0 0 1160 120" role="img" aria-label="共同口径组件图"><defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#607887"/></marker></defs><g font-size="16" text-anchor="middle" fill="#163a50"><rect x="10" y="25" width="200" height="65" rx="8" fill="#e9f1f5"/><text x="110" y="52">旧 / 新16帧输入</text><text x="110" y="75">独立 fresh 10K 重建</text><rect x="285" y="25" width="220" height="65" rx="8" fill="#e9f1f5"/><text x="395" y="52">冻结全部参数</text><text x="395" y="75">共同相机 0 / 8 / 15</text><rect x="580" y="10" width="250" height="95" rx="8" fill="#fff0de"/><text x="705" y="39">native：删除原动态分支</text><text x="705" y="66">M1：同阈值静态选择</text><text x="705" y="89">＋原动态分支删除</text><rect x="905" y="25" width="245" height="65" rx="8" fill="#e9f1f5"/><text x="1027" y="52">独立无车参考评价</text><text x="1027" y="75">raw / clamped ＋来源回放</text></g><g stroke="#607887" stroke-width="2" marker-end="url(#arrow)"><path d="M210 58H280"/><path d="M505 58H575"/><path d="M830 58H900"/></g></svg></div>
<p>两次重建各从零正式 10,000 更新，保留原 30,000 步学习率周期；13/16 个输入相机改变，共同锚点不变。静态行数由 667,559 变为 1,157,894，属于同算法下自然产生的容量差异，不能把结果解释为固定容量的单因素实验。M1 是允许独立像素真值的诊断对照，既不是真实 owner GT，也不是部署方法或数学上界。</p>
<h2>1 · 同一独立 revealed-road ROI：raw / clamped MAE</h2><p class="muted">ROI = 原共同视图 first-hit 车体 ∩ 独立删车 first-hit 道路；四列协议逐帧像素数相同。越低越好。所有分数由原始 FP32 final 评价，未用 PNG 反算。</p><div class="card"><table><thead><tr><th>协议</th><th>000000</th><th>000008</th><th>000015</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
<h2>2 · 新 M1 选前覆盖与删后重新显露分开核验</h2><p>M1 每个 checkpoint 自行按原车体像素内真实统一深度排序的 ΣαT &gt; 10⁻⁴ 选择静态行，删除这些行及全部原动态行。无近深度门限，不移用旧 row IDs。选前覆盖率的分母是原 body 内所有静态贡献；最后一列则是 revealed-road ROI 内保留静态行的删前 → 删后贡献，两个区域不可混作同一个统计。</p><div class="card"><table><thead><tr><th>帧</th><th>选中静态行</th><th>原 body 静态贡献覆盖</th><th>原 body 静态 ΣαT</th><th>保留行 ROI ΣαT：pre → post</th></tr></thead><tbody>{"".join(sel)}</tbody></table></div>
<h2>3 · 全行16view回放：高选前覆盖不等于删后来源已尽</h2><p>每个模型均对自己的16个合法观测视图、正确动态时刻作冻结回放，累计所有静态行的原 vehicle / road / other 像素贡献。以下以各自中帧 M1 删后 revealed-road ROI 的实际 αT 为分母；“累计可见量低于 .1”不是身份标签，也不是训练全过程无梯度的证明。</p><div class="card">{"".join(bars)}<p class="legend muted">{"".join(f'<span style="background:{color}"></span>{html.escape(label)}' for color,label in zip(bincolors,["<1e-4","[1e-4,.01)","[.01,.1)","[.1,1)","≥1"]))}</p><p>相同分箱以共享像素误差加权 αT 作权重，低于 .1 的占比为旧 {s["visibility"]["old"]["errorweight_weak_lt01"]*100:.2f}%、新 {s["visibility"]["new"]["errorweight_weak_lt01"]*100:.2f}%。该统计用于来源排序和分布，不是每颗高斯的单独因果误差。新模型仍有大量原观测低贡献行在删除后显露。</p></div>
<h2>4 · 三个共同锚点原始对照</h2><p><button onclick="mode(false)">final：原环境合成</button><button onclick="mode(true)">white：替换环境为白色</button> <span class="muted">点击图片查看原始 PNG；参考图保持独立无车 RGB。</span></p>{"".join(gallery)}
<h2>证据与边界</h2><p><a href="evidence/source/comparison_summary.json">共同口径总表 JSON</a> · <a href="evidence/source/new_native_metrics.json">新 native 独立评分</a> · <a href="evidence/source/old_native_metrics.json">旧 native 独立评分</a> · <a href="evidence/source/fresh_source_r1/source_000008/source_receipt.json">新中帧来源、row IDs 与守恒收据</a> · <a href="evidence/source/fresh_source_r1/source_000008/top_rows.csv">新中帧 top2000 CSV</a> · <a href="evidence/source/fresh_weighted_visibility_r1/weighted_visibility_bins.json">新全行分箱</a> · <a href="evidence/source/fresh_weighted_visibility_r1/allrow_visibility_aggregate.npz">新全行聚合 NPZ</a> · <a href="evidence/source/delivery_manifest.json">原始文件 SHA256 清单</a></p>
<p class="muted">共同外参 / K、native none 恒等、特征梯度 ΣαT 守恒及 frozen 参数检查均保存；活动训练 Scene 的 transform 内容与 mtime 未变。图像残影判断为代理视觉复核，human_verdict=null，未宣布质量验收、真实场景泛化或新 V43 方法成功。旧模型换新观测的桥接 probe 与这里的 fresh10K 结果分开保存。</p></main><script>function mode(white){{document.querySelectorAll("img[data-white]").forEach(im=>{{im.src=white?im.dataset.white:im.dataset.final;im.parentElement.href=im.src}})}}</script></html>'''
(r/"FRESH_COMPARISON.html").write_text(page,encoding="utf-8")
print(r/"FRESH_COMPARISON.html")

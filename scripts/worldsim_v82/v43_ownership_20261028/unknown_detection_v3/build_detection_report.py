"""生成可核验的检测阶段报告；原图、原生渲染和候选图分别保留。"""
from pathlib import Path
import json,shutil,html
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
OUT=BASE/'outputs/V43_Ownership_20261027/gpu_v43/target_detection_v3'
CTX='14705303724557273004_3105_000_3125_000'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    runs=['native_strict_smoke_r1','native_photo_smoke_r1','native_shadow_only_smoke_r1','native_target_contact_smoke_r1']
    if (HERE/'native_witness_smoke_r1/summary.json').exists():runs.append('native_witness_smoke_r1')
    summaries={}
    for run in runs:
        dest=OUT/run;dest.mkdir(exist_ok=True)
        summaries[run]=json.loads((HERE/run/'summary.json').read_text())
        for name in ['summary.json','contract.json','status.json']:
            shutil.copy2(HERE/run/name,dest/name)
        for case in ['baseline','remove_car','remove_u','remove_car_u']:
            shutil.copytree(HERE/run/case/'native_rgb',dest/case,dirs_exist_ok=True)
    shadow=OUT/'shadow';shadow.mkdir(exist_ok=True)
    for raw in range(77,126):
        for source,name in [(HERE/'shadow_evidence_r2'/f'{raw:03d}_original.png',f'{raw:03d}_original.png'),(HERE/'shadow_evidence_r3'/f'{raw:03d}_propagation_overlay.png',f'{raw:03d}_evidence.png')]:shutil.copy2(source,shadow/name)
    for name in ['contract.json','summary.json','verification.json']:
        shutil.copy2(HERE/'shadow_evidence_r3'/name,shadow/name)
    review=OUT/'review';review.mkdir(exist_ok=True)
    for source in (HERE/'review').glob('*.json'):shutil.copy2(source,review/source.name)
    if (HERE/'row_background_evidence_r1').exists():
        dest=OUT/'background_witness';dest.mkdir(exist_ok=True)
        for source in (HERE/'row_background_evidence_r1').glob('*.json'):shutil.copy2(source,dest/source.name)
    contact=OUT/'contact_effect';contact.mkdir(exist_ok=True)
    for source in (HERE/'contact_effect_evidence_r1').glob('*.json'):shutil.copy2(source,contact/source.name)
    shutil.copy2(HERE/'native_control_audit.json',OUT/'native_control_audit.json')
    coverage=OUT/'source_coverage';coverage.mkdir(exist_ok=True)
    for source in (HERE/'row_surface_coverage_r1').glob('*.json'):shutil.copy2(source,coverage/source.name)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    titles=['冻结基线','只去 car','只去目标 unknown','去 car＋unknown']
    for run in runs:
        sheet=Image.new('RGB',(4*480,3*354),(240,244,248));d=ImageDraw.Draw(sheet)
        for row,raw in enumerate([98,113,114]):
            for col,case in enumerate(['baseline','remove_car','remove_u','remove_car_u']):
                d.text((col*480+8,row*354+3),f'raw{raw} · {titles[col]}',font=font,fill=(25,45,65))
                sheet.paste(Image.open(OUT/run/case/f'{raw:03d}.png'),(col*480,row*354+34))
        sheet.save(OUT/run/'comparison.png')
    rows=[]
    names={'native_strict_smoke_r1':'A 严格阴影＋车身贡献','native_photo_smoke_r1':'B 光度阴影＋车身贡献','native_shadow_only_smoke_r1':'C 只纳入阴影 effect 候选（隔离车身投影规则）','native_target_contact_smoke_r1':'D 车身＋阴影＋新增接地 effect 候选','native_witness_smoke_r1':'独立背景见证保护消融'}
    for run,summary in summaries.items():
        for rec in summary['records']:
            if rec['mode']!='remove_car_u':continue
            rows.append(f"<tr><td>{names[run]}</td><td>{rec['raw']}</td><td>{rec['removed_static_rows']:,}</td><td>{rec['outside_domain_changed_pixels']:,}</td><td>{rec['outside_domain_alpha_drop']:.6f}</td></tr>")
    diagram='''<svg viewBox="0 0 1320 390" role="img" aria-label="原始观测到候选证据到归属暂缓和独立验证，无反馈标签循环" xmlns="http://www.w3.org/2000/svg"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8Z" fill="#547188"/></marker></defs><g font-family="Microsoft YaHei,sans-serif" fill="#17384b" text-anchor="middle"><g fill="#e9f1f8" stroke="#8daabe"><rect x="10" y="20" width="270" height="340" rx="12"/><rect x="340" y="20" width="300" height="340" rx="12" fill="#e6f4ec"/><rect x="700" y="20" width="270" height="340" rx="12" fill="#f2eafb"/><rect x="1030" y="20" width="280" height="340" rx="12" fill="#fff3dc"/></g><g font-size="23" font-weight="700"><text x="145" y="58">① 原始观测</text><text x="490" y="58">② 独立证据</text><text x="835" y="58">③ 高斯归属暂缓</text><text x="1170" y="58">④ 原生验收</text></g><g stroke="#9bb3c2" fill="white"><rect x="30" y="88" width="230" height="64" rx="6"/><rect x="30" y="186" width="230" height="64" rx="6"/><rect x="360" y="88" width="260" height="64" rx="6"/><rect x="360" y="186" width="260" height="64" rx="6"/><rect x="720" y="88" width="230" height="64" rx="6"/><rect x="720" y="186" width="230" height="64" rx="6"/><rect x="1050" y="88" width="240" height="64" rx="6"/><rect x="1050" y="186" width="240" height="64" rx="6"/></g><g font-size="18"><text x="145" y="115">RGB / LiDAR / 相机</text><text x="145" y="141">目标轨迹 / 伪掩码</text><text x="145" y="213">原世界表面跨时刻对应</text><text x="145" y="239">曝光与遮挡核对</text><text x="490" y="115">车身 / 近地阴影证据</text><text x="490" y="141">原始暗化＋随动见证</text><text x="490" y="213">背景正见证 / 冲突</text><text x="490" y="239">缺测保持缺测</text><text x="835" y="115">冻结高斯 alpha × T</text><text x="835" y="141">语义 / 身份 / effect 分开</text><text x="835" y="213">目标 U / 阴影 U / 混合 U</text><text x="835" y="239">逐行原因与来源</text><text x="1170" y="115">只去 car / U / car＋U</text><text x="1170" y="141">同帧原生 RGB 与 alpha</text><text x="1170" y="213">残影＋背景损伤核验</text><text x="1170" y="239">AI 初审 ≠ 人工真值</text></g><g stroke="#547188" stroke-width="3" marker-end="url(#a)"><path d="M145 153V184"/><path d="M490 153V184"/><path d="M835 153V184"/><path d="M1170 153V184"/><path d="M280 170H338"/><path d="M640 170H698"/><path d="M970 170H1028"/></g><g font-size="16"><text x="145" y="300">不使用补景图作证据</text><text x="490" y="300">阴影保留道路语义</text><text x="835" y="300">为后续分类器保存证据</text><text x="1170" y="300">删除结果不回流造标签</text></g></g></svg>'''
    arrowheads=''.join(f'<polygon points="{x-10},{y-6} {x},{y} {x-10},{y+6}"/>' for x,y in [(338,170),(698,170),(1028,170)])
    arrowheads+=''.join(f'<polygon points="{x-6},{y-10} {x},{y} {x+6},{y-10}"/>' for x,y in [(145,184),(490,184),(835,184),(1170,184)])
    diagram=diagram.replace('</svg>','<g fill="#547188">'+arrowheads+'</g></svg>')
    (OUT/'architecture.svg').write_text(diagram,encoding='utf-8')
    bg='<p>独立背景见证正在验收；尚无保护结果。</p>'
    if (OUT/'background_witness/summary.json').exists():
        b=json.loads((OUT/'background_witness/summary.json').read_text())
        bg='<table><tr><th>raw</th><th>背景几何见证行</th><th>保护候选</th><th>颜色不符 mixed</th><th>缺证</th></tr>'+''.join(f"<tr><td>{q['raw']}</td><td>{q['background_geometry_witness_rows']}</td><td>{q['background_protection_candidate_rows']}</td><td>{q['mixed_color_mismatch_rows']}</td><td>{q['unknown_rows']}</td></tr>" for q in b['queries'])+'</table><p>三帧保护候选均未与当前U相交，删除集合不变，跳过重复GPU。源帧没有完整的“无阴影”真值，因此 certified background=0；没有见证不能反推属于车辆。</p>'
    gal=''.join(f'<h3>{names[r]}</h3><p><a href="{r}/summary.json">实际渲染记录</a> · <a href="{r}/contract.json">固定参数与输入合同</a></p><a href="{r}/comparison.png"><img src="{r}/comparison.png" alt="三帧四种原生剥离对照" /></a>' for r in runs)
    cov=json.loads((HERE/'row_surface_coverage_r1/summary.json').read_text())
    coverage_table='<table><tr><th>raw</th><th>49源稳定表面见证</th><th>加入片段外源后（87时刻）</th><th>新增命中原U</th></tr>'+''.join(f"<tr><td>{q['raw']}</td><td>{q['stages']['local49_all']['stable_raw_surface_witness_rows']}</td><td>{q['stages']['union87']['stable_raw_surface_witness_rows']}</td><td>{q['union_added_witness_intersection_u']}</td></tr>" for q in cov['queries'])+'</table>'
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V43 车辆残影与阴影检测 · 原生验证</title><style>body{{background:#f5f8fb;color:#19384a;font:16px/1.7 'Microsoft YaHei',sans-serif;margin:0}}main{{max-width:1400px;margin:28px auto;padding:0 24px}}h1{{font-size:30px}}h2{{margin-top:34px}}.card{{background:white;padding:24px;border:1px solid #cbd9e3;border-radius:12px;margin:18px 0}}.warn{{background:#fff2dd;border-left:6px solid #b87914}}img,svg{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:9px;border-bottom:1px solid #d7e0e8}}a{{color:#206b97}}small{{color:#597182}}input{{width:65%}}.legend span{{display:inline-block;margin:0 15px 10px 0}}.dot{{width:15px;height:15px;display:inline-block;border-radius:50%;vertical-align:middle}}code{{background:#edf2f6;padding:2px 5px}}</style><main>
    <h1>V43：车辆残影检测与归属暂缓</h1><p>第一场景 · 原始观测驱动 · 2026-10-28 · 冻结 checkpoint，新增训练 0 步</p>
    <p><a href="../../../GS_Scene_Diagnostics_20261027/index.html">共享入口</a> · <a href="../../../GS_Scene_Diagnostics_20261027/{CTX}/index.html#targetDetection">同帧 2D RGB＋3D 高斯</a></p>
    <div class="card warn"><b>阴影已纳入算法与显示；“完整去车影且不伤背景”尚未通过。</b><p>49 帧完成独立阴影证据和完整高斯贡献检查。三帧原生对照发现：仅凭重复车身投影仍会纳入后方背景；删除整颗混合高斯造成亮色空洞，车底暗残留也未完全清除。停止扩大同一失败方案，转查独立背景见证。候选数量增加不是召回率提升，暂无人工 GT 或跨场景精确率/召回率。</p></div>
    <div class="card">{diagram}</div>
    <h2>算法改动与可迁移边界</h2><table><tr><th>组件</th><th>实际改动</th><th>泛化与证据边界</th></tr>
    <tr><td>目标专属 unknown</td><td>只处理目标实例相关证据；全场景低语义置信度不自动并入。</td><td>目标 ID 是输入；不依赖场景像素坐标或手画删点框。</td></tr>
    <tr><td>独立阴影通道</td><td>原始 LiDAR 拟合路面；同世界点跨时刻曝光对齐；暗化、颜色衰减和目标随动分别记录。</td><td>预测道路不是无阴影真值；源帧缺证保留 unknown。</td></tr>
    <tr><td>有约束传播</td><td>从严格种子沿路面与光度支持传播，遇车身、其他车、前景深度和背景反证停止。</td><td>每个新增点仍需原始光度证据；不按固定 5px 一圈泛化。不连种子的低置信候选另存。</td></tr>
    <tr><td>完整贡献归因</td><td>用原生 alpha×T 追踪真实行贡献；保存车身、阴影、背景和混合原因。</td><td>中心在框内外不能代替像素贡献；贡献也不等于正确身份。</td></tr>
    <tr><td>背景见证</td><td>原始 RGB 与实际 LiDAR 可见表面、多源颜色稳定性及 SH 颜色比对。</td><td>深度比车辆远不作为排除规则。颜色不符或缺证不能自动归车。</td></tr>
    <tr><td>运行参数</td><td>曝光、深度、时间跨度、颜色差等阈值完整写入合同；未逐帧手调。</td><td>这些仍需校准；未声称阈值已跨场景验证。</td></tr></table>
    <h2>阴影证据：49 帧可拖动检查</h2><p>严格候选 1861 像素；受约束传播后 3881 像素。另存 6397 个光度支持的 unknown，其中未连接种子 4377 个。两者存在包含关系，不相加。19 帧没有严格种子，不强行填满车底。</p>
    <div class="legend"><span><i class="dot" style="background:#00dc8c"></i>车身身份基线</span><span><i class="dot" style="background:#d2d6dc"></i>目标相关 unknown</span><span><i class="dot" style="background:#b864f5"></i>阴影 effect 候选</span><span><i class="dot" style="background:#e9a554"></i>混合支撑；紫边表示兼有阴影证据</span></div>
    <div class="card"><label>raw <b id="raw">98</b> <input id="slider" type="range" min="77" max="125" value="98"></label><p>左：严格种子；中：受约束传播；右：仅有光度支持、尚不够确认的候选。青线是原车身伪掩码轮廓。</p><img id="shadow" src="shadow/098_evidence.png" alt="阴影候选与待定证据"/><details><summary>同帧原始 RGB</summary><img id="original" src="shadow/098_original.png" alt="原始RGB"/></details></div>
    <h2>接地 effect 补充</h2><p>raw98/113/114 新增独立接地候选总量为 67/52/59 像素；其中 5/1/7 像素位于原车身伪掩码内。每点要求原始最前 LiDAR 回波距路面平面不超过 5cm，并在真实回波世界位置重新验证跨时刻原始 RGB 暗化。它仍可能混入轮胎接点，仅作为弱 effect；右半车底仍大幅缺证。</p>
    <h2>原生剥离验收：3 帧 × 每方案 4 个控制</h2><p>以下是原生渲染，未补景。相同 checkpoint、相机、时间和环境，只抑制对应高斯；灰色/紫色候选可同时承担真实道路，当前整行删除会暴露这一问题。图中亮洞不能算“已成功清除”。shadow-only 保留邻近道路较好，但车底暗残留未清除。A/B/D的车身＋阴影组合损伤更大；B不含新增contact也已有白洞，不能把损伤单独归因contact。</p>{gal}
    <h2>背景损伤代理量</h2><p>车身＋effect候选域外，未裁剪RGB改变超过1/255的像素数与平均alpha降幅。它们检查副作用，不等于背景真值错误或检测精确率，不能直接跨候选域排序。另有<a href="native_control_audit.json">216项CPU区域审计</a>：按可见PNG分别检查body、core、effect、effect外和弱道路反证区域，避免shadow-only的大body统计域遮住副作用。</p><table><tr><th>方案</th><th>raw</th><th>抑制 static 行</th><th>区域外改变像素</th><th>区域外平均 alpha 降幅</th></tr>{''.join(rows)}</table>
    <h2>独立背景见证</h2>{bg}<h3>完整199帧原始输入的覆盖对照</h3>{coverage_table}<p>固定每query高斯的世界点，49个原始时刻加片段外每4帧采样，合计87个源时刻；未外推模型运动、未执行源SH匹配。15962个原始颜色观测回放通过。此处是表面观测覆盖，不能替代上表严格背景保护，也不能把508等数值增加归因于片段扩展：两者合同不同。<a href="source_coverage/summary.json">完整分母及匹配采样对照</a>。</p>
    <h2>核验与下一步</h2><p>已用 gpt-6-luna 做三帧视觉初审，<a href="review/native_removal_smoke_visual_review.json">记录</a>保留 human_verdict=null。检查点未写回；19 类语义编码未改变。车底阴影是道路上的关联 effect，不把道路语义改成 car。</p><ol><li>补齐漏检车底的独立可见性/光度证据；无证据区域明确缺证，不能以暗色和邻近硬贴阴影。</li><li>用独立背景见证核验目标候选；目标与背景同时成立的高斯标混合，研究可分离支撑，不能静默当成纯车辆点。</li><li>在独立人工标注的车身残影、阴影、背景区域分别统计贡献加权召回与误伤；冻结规则后再测第二场景。</li><li>将逐行证据、来源、缺测与冲突交给后续可学习分类器；删除渲染只作验收，不能回流生成标签。</li></ol>
    <h2>相关工作与本轮迁移</h2><p><a href="https://arxiv.org/abs/2508.03227">Trace3D（ICCV 2025）</a>将跨视角实例权重与高斯歧义联系起来，并研究训练时拆分歧义高斯。本轮借鉴保存逐行跨视角证据；没有复现其训练或把论文结果当成本项目收益。<a href="https://ijcai.org/proceedings/2025/111">OSDR-GS（IJCAI 2025）</a>显式处理光照与可见性阴影；本轮据此把阴影 effect、路面语义和车辆身份分开，未引入完整逆渲染训练。</p>
    <p><small>task WS-V43-TARGET-DETECTION-20261028 · failure_ledger_refs=[V82-F01] · failure_ledger_delta=none · quality_accepted=false · 当前只第一场景，第三场景仍保持停止。</small></p></main><script>slider.oninput=()=>{{let n=String(slider.value).padStart(3,'0');document.getElementById('raw').textContent=slider.value;shadow.src='shadow/'+n+'_evidence.png';original.src='shadow/'+n+'_original.png';}};</script></html>'''
    (OUT/'index.html').write_text(body,encoding='utf-8')
    print(OUT/'index.html')
if __name__=='__main__':main()

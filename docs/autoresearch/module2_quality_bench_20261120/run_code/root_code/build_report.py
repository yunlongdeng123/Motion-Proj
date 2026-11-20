from pathlib import Path
import json,html,shutil,hashlib

W=Path(__file__).resolve().parent
O=W.parents[1]/'outputs/road_quality_bench_20261120_r1'

def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def prune_row(r):
    out={k:r[k] for k in ('raw','region','case','space','fixed_score_pixels','context_pixels','context_scored_pixels','score_outside_context_pixels','score_excluded_by_alpha95') if k in r}
    out['resolution']=r['resolution'];out['teacher_role']=r['teacher_role'];out['scales']=[]
    for s in r['scales']:
        out['scales'].append(dict(sigma=s['sigma_pixels'],core=s['core'],boundary=s['boundary'],
            boundary_pixels=s['boundary_pixels'],flat=s['flat'],tile_macro=s['tile_macro'],tiles=s['tiles'],
            worst_tiles=[{k:v for k,v in x.items() if k in ('tile_id','bounds_y0_y1_x0_x1','metrics')} for x in s['worst_tiles']]))
    out['edges']=dict(core=r['edges']['context_core_pixels'],boundary=r['edges']['boundary_pixels'],
        reference_role=r['edges']['reference_role'],curves=[])
    for e in r['edges']['curves']:
        out['edges']['curves'].append({k:v for k,v in e.items() if k not in ('tile_aggregate','tiles')})
    return out

def main():
    big=json.loads((O/'report_data.json').read_text(encoding='utf-8'))
    aux=json.loads((O/'material_input_aux/summary.json').read_text(encoding='utf-8'))
    marking=json.loads((O/'marking_reference/results.json').read_text(encoding='utf-8'))
    data=dict(schema='road_quality_bench_display.v2',resolutions={})
    for res,ds in big['resolutions'].items():
        data['resolutions'][res]=[prune_row(r) for r in ds['rows'] if 'scales' in r]
    data.update(aux=aux,marking=marking,gradient=big['gradient'],handoff=big['native_handoff'],
        ROIs=big['ROIs'],previews=big['previews'],native_GPU_receipt=big['native_GPU_receipt'],
        frequency_diagnostic=big['frequency_diagnostic'],source_sha256=big['source_summary_sha256'])
    dump(O/'bench_data.json',data)
    shutil.copyfile(W/'protocol.json',O/'protocol.json');shutil.copyfile(W/'metric_freeze.json',O/'metric_freeze.json')
    svg='''<svg xmlns="http://www.w3.org/2000/svg" width="1320" height="256" viewBox="0 0 1320 256"><defs><marker id="a" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0L8,4L0,8" fill="#7bd5c4"/></marker></defs><rect width="1320" height="256" rx="12" fill="#111e2d"/><g fill="#182f44" stroke="#52728b"><rect x="20" y="30" width="225" height="106" rx="10"/><rect x="280" y="30" width="225" height="106" rx="10"/><rect x="540" y="30" width="225" height="106" rx="10"/><rect x="800" y="30" width="225" height="106" rx="10"/><rect x="1060" y="30" width="240" height="106" rx="10"/></g><g fill="none" stroke="#7bd5c4" stroke-width="2" marker-end="url(#a)"><path d="M245 83H280"/><path d="M505 83H540"/><path d="M765 83H800"/><path d="M1025 83H1060"/></g><g fill="#e3edf7" font-family="Microsoft YaHei,sans-serif" text-anchor="middle" font-size="20"><text x="132" y="65">冻结二维教师</text><text x="392" y="65">世界材质 A₀ + L + H</text><text x="652" y="65">道路高斯 / 足迹读取</text><text x="912" y="65">静态 / 动态 / 环境</text><text x="1180" y="65">最终三维街景</text></g><g fill="#aac1d5" font-family="Microsoft YaHei,sans-serif" text-anchor="middle" font-size="16"><text x="132" y="100">原生1920 → 训练480</text><text x="392" y="100">混合前已见错误细纹</text><text x="652" y="100">覆盖缺口单独验收</text><text x="912" y="100">同次深度排序归因</text><text x="1180" y="100">材质与残影分别检查</text></g><g fill="#133a3c" stroke="#4c9a91"><rect x="40" y="172" width="380" height="60" rx="8"/><rect x="470" y="172" width="380" height="60" rx="8"/><rect x="900" y="172" width="380" height="60" rx="8"/></g><g fill="#d5f3ed" font-family="Microsoft YaHei,sans-serif" text-anchor="middle" font-size="18"><text x="230" y="209">细节对应 · 伪纹理 · 边缘位置</text><text x="660" y="209">完整合法域 · 覆盖 · 有效分母</text><text x="1090" y="209">原生1:1裁片 · 静态残影</text></g></svg>'''
    (O/'architecture.svg').write_text(svg,encoding='utf-8')
    note='''# 道路资产画质复评：细节、覆盖与静态交接

当前道路资产未通过画质验收。最大材质差距已在世界材质读取结果中出现，不主要由后续高斯混合新生；覆盖缺口与静态残影另行失败。MAE仅作颜色拟合辅助。

![组件与数据流](architecture.svg)

## 先查细节匹配

1. 原生1920×1280真实新相机/basis评估：M11孔洞σ1像素HF对应度raw110/86为0.0268/0.00065，能量比分别24.64/40.88。它们是新分辨率、完整合法参考域上稳定道路颜色的条件指标，不能与旧480的0.31/2.42直接比较成方法进退。有效core分别94,288/44,892，所有排除/边界保留。
2. 材质输入层排查：在共同来源有效域，raw110阴影区M11世界输入能量比30.658、对应度0.01814，道路R/α为30.680、0.01849；道路可写权重几乎完整。因此不匹配在混合前已经存在。H置零仍有错误高频；增加H进一步放大不匹配，不能把全部问题归给H或把能量比当因果份额。
3. RGB目标与细节梯度弱对齐：M11平衡双视角余弦约0.015–0.050；HF存在非零局部下降方向，逐帧RGB方向也局部降低HF。它支持测试细节目标，不证明少HF loss是唯一根因，也不证明加loss后必然改善。
4. 4×4教师缩图是已确认的监督损失。当前mask-conditioned box4→线性重建控制在孔洞原生σ1尺度只剩约9–10%教师HF能量。控制不是3DGS，不是表示上限；边界质量归一化与历史全幅均值不完全等价。原历史RGB4×均值另有逐值一致证据。
5. 即使限制到原480训练区域在原生图中的对应位置，M11孔洞σ1能量仍为26.18/44.02倍；新合法域扩展不足以解释这些碎纹。该分域是在首轮原生结果后新增的只读诊断，主bench未改。

## 再查其他问题

- 原生raw110/86目标孔洞合法参考域168,932/90,943像素；其中20,643/17,165像素没有固定道路mesh交点。道路α≥.95覆盖率约72.90%/66.67%。这些区域不能靠颜色微调恢复；较薄覆盖与真正空洞、几何缺失分开记录。完整可见A超出有限道路资产范围的部分不直接判成本任务失败。
- 同次原生深度排序的静态分支取证：道路本身α≥.95，却被静态分支遮挡至联合道路权重<.05的目标孔洞像素为3,392/2,280。贡献裁片仍显示车身/尾灯状残留；这是模块4的独立问题。分支权重不是语义残影检测器，source RGB也不是删除静态层后的反事实渲染。
- 真实标线完整几何尚未验收。raw86只验证了几处实拍白漆局部ROI；raw110没有足够合格连续白漆样本。局部教师外观边缘不能替代全标线中心线、线宽、连续性的真值。

## 更新后的bench

主评价是同分辨率稳定道路R/α对原教师T，公共α≥.95且所有方法alpha exact；完整pre-alpha合法域保留覆盖/只读/边界分母。σ1/2/4多尺度HF对应、误差、能量与教师平坦区伪HF并列；边缘位置、方向、精度/召回并列；固定tile宏均值、尾部和最坏tile注明有效core数（1个core像素不代表整块32²）。原P对αT作为独立辅助空间，新指标没有总分或事后画质通过阈值。

原生GPU12次前向、0参数更新；CPU指标13项合成负控、2项HF数值/伴随检查通过，另有独立78项聚合核验。真实1920调用51.65秒，峰分配约4.93GB。GPU保持开机，未改生产资产。

下一步先做原始分辨率、同世界区域的受约束材质蒸馏对照：原生RGB与可信HF/边缘目标分开消融，并将高频可学习带宽与教师实际足迹对应；同时先补足合法目标道路的表面支持，再依据真实贡献清退静态残影。不建议用增加步数或更换生成大模型代替这三项修复。

远端证据：`/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1/gpu_20261119/quality_bench_20261120_r1`。完整float、原始tile JSON、nativebasis和source参考保留远端，局部无损图与紧凑结果已取回。
'''
    (O/'report.md').write_text(note,encoding='utf-8')
    result=dict(schema='road_quality_bench_key_results.v2',verdict='not_accepted',
        native_primary=[r for r in data['resolutions']['native1920'] if r['region']in('C','S')and r['case']in('M01','M11','M10','M11_noH','teacher_box4_linear')],
        material_input=aux,native_handoff=data['handoff'],GPU=data['native_GPU_receipt'],
        scope='frozen existing assets; no optimization; pseudo hidden-road teacher;199BUILD not blind',
        full_bench_data='bench_data.json',remote_full_evidence=big['protocol']['paths']['remote_output'])
    dump(O/'results.json',result)
    assert (O/'results.json').stat().st_size<2_000_000
    payload=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>道路资产画质复评</title><style>:root{color-scheme:dark}*{box-sizing:border-box}body{background:#0a111b;color:#e1ebf6;margin:0;font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1500px;margin:auto;padding:28px}h1{font-size:28px;margin:0 0 12px}h2{font-size:21px;margin:28px 0 10px}h3{font-size:16px;margin:0 0 7px}p{margin:8px 0;color:#afc0d3}.notice{padding:14px 18px;border:1px solid #b57552;background:#38271f;border-radius:8px;color:#ffd0ab}.architecture{width:100%;margin:18px 0}.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.card,details{padding:14px;background:#112031;border:1px solid #30465f;border-radius:8px}.card strong{color:#91d6ca}.controls{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0;align-items:center}select{padding:8px;background:#172c43;color:#e5eef7;border:1px solid #496580;border-radius:5px}.scroll{overflow:auto}.strip{display:flex;gap:12px;overflow:auto;padding-bottom:10px}.panel{flex:0 0 auto;padding:8px;border:1px solid #30465d;border-radius:7px;background:#111e2d}.panel img{display:block;image-rendering:pixelated;max-width:none}.overview{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.overview img{width:100%;display:block}.overview .panel{min-width:0}.overview .panel img{image-rendering:auto}table{border-collapse:collapse;white-space:nowrap;width:100%;font-size:13px}th,td{padding:9px;border-bottom:1px solid #2a4059;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#182b40;color:#b9d4ee}.small{font-size:12px}a{color:#7ccaff}summary{cursor:pointer;color:#bdd9f3}details{margin:14px 0}pre{white-space:pre-wrap;font-size:12px;color:#b5c9db}@media(max-width:760px){main{padding:14px}.cards,.overview{grid-template-columns:1fr}h1{font-size:24px}}</style><main>
    <h1>道路资产画质复评</h1><div class="notice">画质未通过。世界材质读取结果已出现错误细纹；覆盖缺口与静态车残影另行失败。颜色MAE不作为恢复完成的证据。</div><img class="architecture" src="architecture.svg" alt="二维教师到世界材质、道路高斯、场景交接与画质验收">
    <div class="cards"><div class="card"><h3>① 主要材质卡点</h3><p>原生raw110阴影区，混合前世界材质的高频能量为教师<strong>30.66倍</strong>，对应度<strong>0.018</strong>；道路输出几乎不变。错误细纹已在材质输入层。</p></div><div class="card"><h3>② 监督与目标</h3><p>教师4×4缩到480；RGB与细节梯度余弦仅<strong>0.015–0.050</strong>。需要原生分辨率细节约束，不能只加训练步数；缺HF loss也不是已证唯一根因。</p></div><div class="card"><h3>③ 覆盖与静态交接</h3><p>完整合法孔洞域的道路α≥.95仅<strong>72.9% / 66.7%</strong>。两帧另有<strong>3,392 / 2,280</strong>个高覆盖道路像素被静态分支强遮挡。</p></div></div>
    <h2>直接看原生结果</h2><div class="controls"><label>帧 <select id="raw"><option selected>110</option><option>86</option></select></label><label>区域 <select id="region"><option value="C">目标孔洞</option><option value="S">目标阴影</option></select></label><label>道路对照 <select id="method"><option>M11</option><option>M01</option><option>M10</option><option>M11_noH</option></select></label></div><div class="overview"><div class="panel"><h3>原二维教师</h3><img id="overviewT" alt="教师概览"></div><div class="panel"><h3>三维道路层</h3><img id="overviewR" alt="道路概览"></div><div class="panel"><h3>M11最终联合街景</h3><img id="overviewJ" alt="联合场景概览"></div></div><p class="small">概览缩到480仅便于定位，不参与评分。下方为真实1920渲染的原生1:1裁片，横向滚动查看。教师从未乘道路α；道路层黑区含覆盖缺口。</p><div class="strip" id="crops"></div><p id="roiNote" class="small"></p>
    <h2>细节对应、伪纹理与边缘位置</h2><div class="controls"><label>评价像素网格 <select id="resolution"><option value="native1920" selected>原生1920×1280</option><option value="coarse480">训练480×320</option></select></label><label>HF尺度 <select id="sigma"><option>1</option><option>2</option><option>4</option></select></label></div><div class="scroll"><table id="metrics"></table></div><p id="denom" class="small"></p><p class="small">对应度理想为1；能量比需与对应度同时看，接近1也可纹理错位。边缘为同域教师外观边缘，阈值2/255、容差1个当前网格像素；不冒称完整标线几何真值。原生与480结果的分辨率、分母和尺度不同，不能直接比较成方法进退。</p>
    <details><summary>固定tile尾部与最差病例</summary><div class="scroll"><table id="tails"></table></div><p class="small">tile固定全图原点，480为32²、1920为128²。宏均值保留每个非空tile；最坏tile可能只有1个core像素，必须结合n解读，不能当整块质量。最差样本是事后指标定位，不是盲测。</p></details>
    <h2>材质输入与高斯混合：同域拆分</h2><div class="scroll"><table id="layers"></table></div><p class="small">这是来源可用且α≥.95的辅助子域，和完整主bench分母不同。阴影区可写道路权重接近1，混合前后的接近支持“材质本身已不匹配”；孔洞有只读混合，不能把能量比当因果百分比。H关闭后仍有错误纹理，说明基底+L也未恢复教师质感。</p>
    <h2>完整目标域的覆盖与残影</h2><div class="scroll"><table id="coverage"></table></div><p class="small">完整可见A远大于有限资产范围，资产chart之外的A不直接判本次修复失败。C/S保留合法目标完整分母。静态贡献来自同一次深度排序，不是独立删除静态层的反事实；具体残影形状仍需结合上方裁片。</p>
    <details open><summary>真实标线与清晰度的验收边界</summary><p>raw86仅有三处实拍白漆局部ROI，当前可评局部外观边缘与细纹；raw110连续白漆样本不合格。没有完整中心线、线宽和连续性真值，完整标线形状保持<strong>未验收</strong>，C/S生成边缘不作真实标线答案。</p><p>最大raw86白漆ROI内，边缘位置P/R约M11 0.689/0.850、M10 0.733/0.847；M11细纹对应度仍仅约0.046。邻近边缘存在不等于路面质感或完整线形正确。</p></details>
    <details><summary>监督缩图控制、梯度与复现</summary><p>教师box4→线性重建控制在原生孔洞σ1尺度约保留9–10%教师HF能量。它是mask-conditioned降采样控制，不是GS预测，也不是表示上限；partial-context不宣称完整历史链路逐像素等价。</p><p>CPU HF求导与冻结指标RMSE最大差1.06×10⁻⁹。原RGB与HF方向弱对齐，逐帧原RGB下降仍局部改善HF；仅凭“没有HF loss”不能断言唯一根因或保证新loss有效。原生stream在旧480的256行/帧spot对compiled颜色与质量max0，对保存RGB≤1.19×10⁻⁷。</p><p>原生GPU12次前向 / 0更新，51.65秒，峰分配约4.93GB；两帧新K权重及K RGB变化均0。CPU指标13项负控、2项HF数值/伴随检查通过，另有78项独立聚合核验。既有199帧参与BUILD，非盲测。</p><p><a href="results.json">关键结果</a> · <a href="bench_data.json">细节与分母</a> · <a href="protocol.json">bench协议</a> · <a href="report.md">完整分析</a> · <a href="architecture.svg">组件图</a></p><pre>/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1/gpu_20261119/quality_bench_20261120_r1
    metrics/ · lowres_r1/ · native_detail_r1/ · gradient_probe/
    highres/material_input_aux_r1/ · secondary_audit/native_r1/</pre></details>
    <h2>下一步修复顺序</h2><p>先在原始分辨率、相同世界区域验证受约束材质蒸馏：分别消融RGB与可信HF/边缘目标，并使可学习高频带宽匹配教师足迹。再补齐目标区域道路表面支持，依据真实成像贡献处理静态残影。当前证据不支持用增加步数或更换生成大模型代替这些修复。</p><p class="small">本轮只重评冻结资产；未追加训练或修改生产资产。GPU保持开机。</p></main>
    <script>const D=PAYLOAD;const $=x=>document.getElementById(x),fmt=(x,n=4)=>x==null?'N/A':Number(x).toFixed(n),esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const assets=D.previews.assets;
    function asset(raw,role,kind,roi){return assets.find(a=>a.raw===raw&&a.role===role&&a.kind===kind&&(!roi||a.ROI===roi))}function imagePanel(label,a){let p=document.createElement('div');p.className='panel';let h=document.createElement('h3');h.textContent=label;p.append(h);let im=document.createElement('img');im.alt=label;im.src='previews/'+a.file;p.append(im);return p}
    function images(){let raw=+$('raw').value,reg=$('region').value,m=$('method').value,roi=reg+'_native_source';for(let [id,role]of[['overviewT','T_'+reg],['overviewR',m+'_road_P'],['overviewJ','M11_joint_J']])$(id).src='previews/'+asset(raw,role,'overview_only').file;$('crops').replaceChildren();for(let [label,role]of[['原始二维伪教师','T_'+reg],[m+'道路层P',m+'_road_P'],['M11最终联合街景','M11_joint_J'],['静态分支RGB贡献','static_contribution']])$('crops').append(imagePanel(label,asset(raw,role,'native_1to1_crop',roi)));let a=D.ROIs.ROIs.find(a=>a.raw===raw&&a.id===roi);$('roiNote').textContent='源mask边界框+32像素：'+a.bbox_xyxy.join(', ')+'；1:1无锐化。ROI晚于GPU产生、仅按源mask选择，未按预测选择。'}
    function table(id,heads,rows){$(id).innerHTML='<tr>'+heads.map(x=>'<th>'+esc(x)+'</th>').join('')+'</tr>'+rows.map(r=>'<tr>'+r.map(x=>'<td>'+esc(x)+'</td>').join('')+'</tr>').join('')}
    function metrics(){let raw=+$('raw').value,reg=$('region').value,res=$('resolution').value,sigma=+$('sigma').value;let cases=res==='native1920'?['M01','M11','M10','M11_noH','teacher_box4_linear']:['M00','M10','M01','M11','M12','teacher_blur_sigma1'];let rows=D.resolutions[res].filter(r=>r.raw===raw&&r.region===reg&&cases.includes(r.case)&&(res!=='coarse480'||r.space==='straight_road_srgb_alpha95'));let vals=[],tail=[];for(let r of rows){let s=r.scales.find(s=>s.sigma===sigma),e=r.edges.curves.find(e=>Math.abs(e.strength_threshold-2/255)<1e-9),pr=e.curves.find(c=>c.tolerance_pixels===1);vals.push([r.case,s.core.pixels,fmt(s.core.HF_cosine),fmt(s.core.HF_energy_ratio,2),fmt(s.core.HF_relative_RMSE,3),fmt(pr.precision,3),fmt(pr.recall,3)]);let mac=s.tile_macro.HF_MAE,w=s.worst_tiles[0];tail.push([r.case,fmt(mac.mean,5),fmt(mac.p95,5),w?w.tile_id.join(','):'N/A',w?w.metrics.pixels:0,w?fmt(w.metrics.HF_RMSE,5):'N/A'])}table('metrics',['方法 / 控制','HF core n','HF对应度 ↑','HF能量 / 教师','相对HF误差 ↓','边缘精度 ↑','边缘召回 ↑'],vals);table('tails',['方法','tile宏MAE','tile p95 MAE','最坏tile','该tile core n','该tile HF RMSE'],tail);let r=rows.find(r=>r.case==='M11');if(r){let s=r.scales.find(s=>s.sigma===sigma);$('denom').textContent='M11：完整score '+r.fixed_score_pixels+'；稳定context内score '+r.context_scored_pixels+'；σ'+sigma+'严格core '+s.core.pixels+'；context边界 '+s.boundary_pixels+'。core外像素未被当作已通过。'}
    let a=D.aux.filter(a=>a.raw===raw&&a.region===reg&&a.sigma===sigma);let layer=[];for(let x of a){layer.push([x.case+'材质输入',x.core,fmt(x.world_cosine),fmt(x.world_energy,2),fmt(x.world_HF_MAE,5)]);layer.push([x.case+'道路R/α',x.core,fmt(x.straight_cosine),fmt(x.straight_energy,2),fmt(x.straight_HF_MAE,5)])}table('layers',['读取 / 成像层','共同core n','HF对应度','能量比','HF误差'],layer);
    let cov=[];for(let r of D.handoff.rows)for(let k of['C','S']){let x=r.domains[k],c=x.coverage.find(v=>v.alpha_threshold===.95);cov.push([r.raw+' / '+k,x.expected_pre_GS_alpha_pixels,x.no_surface_hit_pixels,fmt(100*c.pixels/x.expected_pre_GS_alpha_pixels,2)+'%',x.high_road_alpha_but_joint_road_weight_below05,x.stable_road_static_ge50_joint_road_below05])}table('coverage',['帧 / 目标域','完整合法像素','无mesh交点','α≥.95覆盖','有路但联合道路<.05','其中静态主导'],cov)}function update(){images();metrics()}for(let id of['raw','region','method'])$(id).onchange=update;for(let id of['resolution','sigma'])$(id).onchange=metrics;update();</script></html>'''
    page=page.replace('<h2>细节对应、伪纹理与边缘位置</h2>','<h2>细节对应、伪纹理与边缘位置</h2><p class="small">M01：像素着色，仅实测高频许可；M11：像素着色，加入教师高频许可；M10：高斯足迹平均，教师高频许可；M11_noH：固定最终低频关闭H。480对照另含M00（高斯平均、仅实测许可）和M12（来源足迹滤波分支）。教师缩图控制不是三维预测。</p>')
    page=page.replace("vals.push([r.case,s.core.pixels,fmt(s.core.HF_cosine)","let flat=s.flat.find(f=>Math.abs(f.teacher_HF_RMS_cut-2/255)<1e-9);vals.push([r.case,s.core.pixels,fmt(s.core.HF_cosine)")
    page=page.replace("fmt(pr.precision,3),fmt(pr.recall,3)]);let mac", "fmt(pr.precision,3),fmt(pr.recall,3),flat?fmt(flat.predicted_HF_RMS,5)+' / '+fmt(flat.teacher_HF_RMS,5)+' (n='+flat.pixels+')':'N/A']);let mac")
    page=page.replace("'边缘精度 ↑','边缘召回 ↑'],vals", "'边缘精度 ↑','边缘召回 ↑','平坦区 预测/教师HF RMS'],vals")
    page=page.replace('边缘为同域教师外观边缘，','平坦区按教师HF RMS≤2/255定义，另保留其样本数。边缘为同域教师外观边缘，')
    (O/'index.html').write_text(page.replace('PAYLOAD',payload),encoding='utf-8')
    dump(O/'report_build.json',dict(index_bytes=(O/'index.html').stat().st_size,results_bytes=(O/'results.json').stat().st_size,
        bench_data_bytes=(O/'bench_data.json').stat().st_size,sources={'report_data.json':hashlib.sha256((O/'report_data.json').read_bytes()).hexdigest()},
        model_quality_pass=False,training_updates=0))
    print(json.dumps(dict(output=str(O/'index.html'),results_bytes=(O/'results.json').stat().st_size,bench_data_bytes=(O/'bench_data.json').stat().st_size)))

if __name__=='__main__':main()

"""Build the local dark engineering report from frozen CPU metrics and lossless views."""
from pathlib import Path
import argparse,json,shutil,html,hashlib,datetime
import numpy as np

ARMS=['M00','M10','M01','M11','T11']
LABELS={'M00':'高斯平均 · 仅实测高频','M10':'高斯平均 · 教师高频','M01':'像素着色 · 仅实测高频','M11':'像素着色 · 教师高频','T11':'像素着色 · 教师高频 + TRELLIS种子',
        'M12':'像素着色 · 教师高频 + 来源足迹约束',
        'N_parent':'历史：旧材质 + 修复覆盖','pixel_initial':'像素着色初始基底','pixel_common_A200':'像素着色共同200步','avg_initial':'高斯平均初始基底','avg_common_A200':'高斯平均共同200步',
        'teacher_A':'原始观测（车下无GT）','teacher_C':'固定孔洞教师（伪真值）','teacher_S':'固定去影教师（伪真值）'}
ARCH='''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 270"><defs><marker id="a" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7" fill="#7f95b5"/></marker></defs><rect width="1120" height="270" rx="14" fill="#101925"/><g fill="#18283c" stroke="#45607d"><rect x="22" y="28" width="174" height="83" rx="8"/><rect x="22" y="154" width="174" height="83" rx="8"/><rect x="238" y="72" width="205" height="127" rx="8"/><rect x="488" y="72" width="178" height="127" rx="8"/><rect x="712" y="28" width="173" height="83" rx="8"/><rect x="712" y="154" width="173" height="83" rx="8"/><rect x="930" y="72" width="168" height="127" rx="8"/></g><g fill="#e6eef8" text-anchor="middle" font-family="sans-serif" font-size="16"><text x="109" y="59">实拍 A / 教师 C、S</text><text x="109" y="84" font-size="13">固定86、110帧</text><text x="109" y="184">修复后的道路几何</text><text x="109" y="210" font-size="13">位置 / 尺度 / 不透明度冻结</text><text x="340" y="106">独立材质证据与保护</text><text x="340" y="133" font-size="13">实测 / 教师高频许可</text><text x="340" y="158" font-size="13">共同写域 89,887 / K保护</text><text x="577" y="106">持久世界材质场</text><text x="577" y="133" font-size="13">冻结基底 + 12cm低频</text><text x="577" y="158" font-size="13">+ 2cm高频（界 .34）</text><text x="798" y="59">高斯平均读取</text><text x="798" y="84" font-size="13">物理Σ积分 → 每行颜色</text><text x="798" y="184">像素世界着色</text><text x="798" y="210" font-size="13">表面射线 / 像素足迹积分</text><text x="1014" y="106">原生 GS 贡献</text><text x="1014" y="133" font-size="13">固定真实 αT / 只读分支</text><text x="1014" y="158" font-size="13">五帧图像 / 独立评价</text></g><g fill="none" stroke="#7f95b5" stroke-width="2" marker-end="url(#a)"><path d="M196 69L238 103"/><path d="M196 195L238 164"/><path d="M443 135H488"/><path d="M666 109L712 69"/><path d="M666 163L712 195"/><path d="M885 69L930 108"/><path d="M885 195L930 164"/></g></svg>'''

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def fmt(v,d=5):return '—'if v is None else f'{v:.{d}f}'
def make(inputs,out):
    inputs=Path(inputs);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    existing=[out/n for n in ['index.html','results.json','report.md','report_data.json','architecture.svg']if(out/n).exists()]
    if existing:
        backup=inputs.parent/'report_backups'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');backup.mkdir(parents=True)
        for f in existing:shutil.copy2(f,backup/f.name)
    assets=out/'assets';assets.mkdir(exist_ok=True)
    (assets/'no_teacher.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" width="480" height="320"><rect width="480" height="320" fill="#101925"/><text x="240" y="160" fill="#b5c9e1" text-anchor="middle" font-family="sans-serif" font-size="20">诊断视角：无教师</text></svg>',encoding='utf-8')
    for f in (inputs/'previews_r1').glob('*.webp'):shutil.copy2(f,assets/f.name)
    data=read(inputs/'results_r3/summary.json');details={a:read(inputs/f'results_r3/{a}/metrics.json')for a in data}
    decomp=read(inputs/'results_r3/directional_source_decomposition.json');rois=read(inputs/'frozen_r2/rois.json')['rois']
    persisted=read(inputs/'field_training/persistence_179_r1/receipt.json');lifecycle=read(inputs/'lifecycle_r1/receipt.json')
    range_proxy=read(inputs/'pixel_color_range_audit_r1/results.json');filter_probe=read(inputs/'source_footprint_filter_r1/results.json')
    M12=read(inputs/'M12_r1/metrics.json')if(inputs/'M12_r1/metrics.json').exists()else None
    ideal=read(inputs/'ideal_read_probe/results.json')
    if M12 is not None:
        details['M12']=M12
        for f in(inputs/'M12_r1/previews').glob('*.webp'):shutil.copy2(f,assets/f.name)
    result=dict(schema='teacher_detail_factorial_report.v1',scientific_verdict='NOT_PROMOTED_VISIBLE_NEW_HORIZONTAL_BANDING',human_verdict=None,
        arms={a:dict(label=LABELS[a],**data[a])for a in ARMS},initial_and_common={a:data[a]for a in data if a not in ARMS},
        fixed_denominators=details['M11']['fixed_denominators'],directional_source_decomposition=decomp,ROIs=rois,
        actual_native_qualification=dict(K77266_eligible_joint_weight_exact_zero=True,basis_constant_float_max_error=1.0132789611816406e-6,RGB8_max_byte_error=1,not_bitexact=True),
        T11=dict(role='bounded_projected_HF_initialization_only',new_TRELLIS_inference=0,strict_faithful_gate=False,exploratory_seed_valid=True,source_transfer_cosine=.8280024,
            original_patch_size_m=2.56,whole_street_prior=False,effective_HF_seed_cap=.02,A0_basecolor_replaced=False),
        source_roles=dict(A='measured visible road',C_S='frozen independent 2D pseudo teachers; not hidden truth',diagnostic='134/179/198 no teacher; existing199BUILD not blind'),
        training=dict(steps_per_logical_arm=800,A=200,B=300,C=300,physical_updates=3400,logical_updates=4000,main4_physical_updates=2800,
            low_spacing_m=.12,high_spacing_m=.02,high_bound=.34,geometry_opacity_scales_active_frozen=True,extra_engineering_updates='separate controller/lifecycle receipts'),
        CPU_evaluation_native_calls=0,CPU_evaluation_training_updates=0,
        readonly_range_proxy=dict(status='separate FP64 fixed-basis numeric proxy, not global capacity certificate',A_MAE_floor=.016136,A_zero_We_pixels=2962,A_zero_We_contribution=.015118,C_MAE_floor=.001304,S_MAE_floor=0.,source='pixel_color_range_audit_r1/results.json'),
        source_root='/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1/gpu_20261119',
        limitations=['Permission improves matched-reader teacher fit but introduces visible fine horizontal banding in diagnostics.',
            'Pixel teacher arm improves photo fit over Gaussian average but has slightly worse C/S HF1 MAE.',
            'Second-difference energy alone is not a semantic stripe classifier; compare fixed crops and readonly decomposition.',
            'TRELLIS arm uses a partial projected HF seed, not a faithful transfer or model capacity test.',
            'Pixel assets require persistent world field + reader + native basis refresh; a baked per-Gaussian RGB file does not reproduce this renderer.'])
    result.update(persistence=dict(M11_C800_raw179_fresh_basis=persisted,dual_reader_five_frame_smoke=lifecycle,
        other_formal_arms_fresh_basis_all5='not separately asserted'),source_footprint_zero_update_diagnostic=filter_probe,
        ROI_timing=dict(rule='old frozen mask/teacher only; fixed before first result read, not before result production',
            formal_completed_UTC='2026-11-18T18:56:19.302049Z',ROI_file_UTC='2026-11-18T18:56:30.366155Z',first_M11_metric_UTC='2026-11-18T18:58:15.547116Z',
            original_frozen_r2_before_output_wording='timing clarified here; mask/bbox data unchanged'))
    result['readonly_range_proxy'].update(exact_proxy_source=range_proxy)
    result['training']['extra_engineering_updates']=lifecycle['engineering_optimizer_updates']
    result['frozen_evaluation_manifest_SHA256']=hashlib.sha256((inputs/'frozen_r2/manifest.json').read_bytes()).hexdigest()
    result['frozen_ROI_SHA256']=hashlib.sha256((inputs/'frozen_r2/rois.json').read_bytes()).hexdigest()
    result['ideal_reader_control']=ideal
    if M12 is not None:
        M12_pair=read(inputs/'M12_r1/paired_M11_M12.json');M12_done=read(inputs/'source_filtered_training/formal_r1/completed.json')
        result['additional_registered_branch_M12']=dict(label=LABELS['M12'],metrics=M12,paired=M12_pair,completion=M12_done,
            verdict='Partial diagnostic-frequency reduction with worse teacher photo/HF/flat-noise; not promoted',original_factorial_unchanged=True)
        result['training'].update(original5_physical_updates=3400,physical_updates=4000,logical_updates=4800,extra_engineering_updates=24,
            additional_M12_physical_updates=600,additional_M12_engineering_updates=7)
    dump(out/'results.json',result);dump(out/'report_data.json',dict(results=result,details=details,labels=LABELS))
    (out/'architecture.svg').write_text(ARCH,encoding='utf-8')
    table=[];md=[]
    for a in ARMS:
        v=data[a];ey=np.mean([r['vertical_E']for r in v['stripe'][2:]]);ex=np.mean([r['horizontal_E']for r in v['stripe'][2:]])
        values=[LABELS[a]+' ('+a+')',fmt(v['photo']['A']),fmt(v['photo']['C']),fmt(v['photo']['S']),fmt(v['HF1']['C']['HF_MAE']),fmt(v['HF1']['S']['HF_MAE']),fmt(v['HF1']['C']['HF_cosine'],3),fmt(v['HF1']['S']['HF_cosine'],3),fmt(ey*1000,2),fmt(ey/ex,2)]
        table.append('<tr>'+''.join('<td>'+html.escape(s)+'</td>'for s in values)+'</tr>');md.append('| '+' | '.join(values)+' |')
    note='教师高频许可改善孔洞与阴影的拟合，但诊断视角出现新增细横纹；本轮不晋升。'
    mdtext=f'''# 道路教师高频：固定覆盖下的读取方式与学习许可\n\n{note}\n\n![架构](architecture.svg)\n\n86/110帧用于优化；134/179/198仅诊断。既存199帧参与BUILD，不是盲测。A为实拍可见道路，C/S为冻结二维伪教师；车下没有真实道路GT。\n\n每组逻辑800步（200+300+300），共享各读取方式的前200步，共3400次正式更新。几何、尺度、不透明度、激活和其他场景分支冻结。写域89,887；高频网格2cm、界0.34；低频网格12cm、界0.30。\n\n| 方法 | 实拍 MAE | 孔洞 MAE | 阴影 MAE | 孔洞 HF1 MAE | 阴影 HF1 MAE | 孔洞 HF余弦 | 阴影 HF余弦 | 诊断 Ey×1000 | Ey/Ex |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'''+ '\n'.join(md)+'''\n\n同读取方式下，教师高频许可降低C/S高频误差；像素着色的教师组比高斯平均教师组颜色误差更低，但C/S高频误差略高。固定诊断裁片中的细横纹同步增强，不能将能量增加等同于细节恢复。\n\nHF采用域内归一化高斯滤波（σ1/2像素），仅在固定域局部质量≥0.9处评价；边界不足单列，不改变完整A16904/C10830/S3131分母。完整阴影3187、新覆盖909、薄区56、原道路38416、K77266均保留。边缘F1与教师平坦区伪高频另外报告。\n\n原生只读贡献会限制部分实拍拟合：A内2962像素实际可写αT为0；另7像素world reader无材质支持，属于上述集合。点像素颜色范围代理的A/C/S最低MAE约0.016136/0.001304/0；该代理不构成全局容量证书，也不能解释全部高频误差。\n\nTRELLIS列只在B入口注入已有输出的受限、部分投影高频初值，之后损失与M11相同；没有新推理，没有替换底色。严格faithful-transfer门未通过，不能用该列否定TRELLIS容量。\n\n高斯平均资产以世界场→物理Σ积分→每行颜色重放；像素资产需要世界场、读取算子与刷新后的真实原生basis，不冒充普通RGB烘焙GS。五组本次保存图像的K联合RGB及道路/联合α均精确不变；持久恢复详情见各public_reload收据。\n\n可复现：`python -m evaluation.collect_evaluation --gpu-dir <Q/gpu_20261119> --out <new_evaluation_dir>`。输入固定在`evaluation/frozen_r2/manifest.json`；最终数据在`formal_r1/{arm}/C800`；本报告独立指标在`evaluation/results_r2`。评价只读CPU，0渲染、0优化更新。\n'''
    mdtext=mdtext.replace('持久恢复详情见各public_reload收据','M11最终field在179帧以新原生basis重载，RGB/α/K差值0（2次前向、0更新）；双读取方式五帧生命周期重放另有收据。其余组不冒称逐组五帧fresh-basis重验')
    mdtext=mdtext.replace('evaluation/results_r2','evaluation/results_r3')
    mdtext+='\nROI在首次读取本轮结果前，仅按冻结mask/teacher规则生成；文件落盘晚于GPU结果产出11秒，未宣称GPU前预注册。\n\n零更新来源足迹滤波诊断使用教师像素Jbox中位尺度σUV=(0.822528,0.299986)，未扫参。诊断Ey约降15%，但孔洞/阴影HF误差上升至0.005340/0.007523，颜色误差亦变差；不作为已训练方法或自动晋升依据。原5组3400更新结论保持，后续有据分支另行记账。\n'
    if M12 is not None:
        mdtext+='\n单一预登记追加分支M12从共同像素A200状态重启B300+C300，将同一固定来源足迹滤波放入反传，不改学习率、损失、许可、界或采样。额外600正式+7工程；合计4000正式、24工程、4800逻辑步。孔洞/阴影MAE为0.010270/0.012811，HF1误差0.005411/0.007264，均劣于M11；三诊断帧Ey约降7–9%，仍高于M01。教师平坦区伪HF亦略增。该分支部分缓和频率能量，但没有同时解决细节拟合与条纹，不晋升。完整field重查五帧exact0及实际179新basis证明见独立分支收据；不重复计算原共享A200预算。\n'
    ideal_lookup={(v['raw'],v['case'],v['method']):v for v in ideal['metrics']if v['mode']=='road'}
    ideal_rows=[];ideal_md=[]
    for raw in [86,110,134,179,198]:
        vals=[str(raw)]+[fmt(ideal_lookup[raw,case,method]['fixed_detail_metrics_after_readonly_cancellation']['contrast_retention'],3)for case,method in [('marking_w120','average_GH31'),('marking_w120','pixel_box'),('marking_w240','average_GH31'),('marking_w240','pixel_box')]]
        ideal_rows.append('<tr>'+''.join('<td>'+html.escape(x)+'</td>'for x in vals)+'</tr>');ideal_md.append('| '+' | '.join(vals)+' |')
    mdtext+='\n理想标线读取控制（30次真实原生前向、0更新）：独立12/24cm世界标线，经共同2cm源网格读取，原生标线减无标线沥青以消去只读分支。下表为道路层固定细节域对比度保持率。\n\n| 帧 | 12cm 高斯平均 | 12cm 像素 | 24cm 高斯平均 | 24cm 像素 |\n|---|---:|---:|---:|---:|\n'+'\n'.join(ideal_md)+'\n\nraw86的12cm像素保持率0.880仍未达旧0.90门。2cm源网格在像素中心约0.944，像素足迹平均再衰减；不通过更换真值隐藏这一差异。该合成读取控制不等于真实教师训练成功，也不是3DGS表示上限证书。\n'
    (out/'report.md').write_text(mdtext,encoding='utf-8')
    (out/'report.md').write_text(mdtext,encoding='utf-8')
    compact={a:dict(label=LABELS[a],HF=data[a]['HF1'],edges=data[a]['edges'])for a in ARMS}
    scriptdata=json.dumps(dict(labels=LABELS,rois=rois,details=compact),ensure_ascii=False).replace('</','<\\/')
    htmltext='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>道路教师高频对照</title><style>
:root{color-scheme:dark;--bg:#0a1019;--panel:#121d2b;--line:#2b3a4e;--text:#e4edf7;--muted:#9eafc3}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1440px;margin:auto;padding:28px}h1{font-size:27px;margin:0 0 8px}h2{font-size:20px;margin:28px 0 12px}p{margin:8px 0;color:var(--muted)}.notice{padding:14px 18px;background:#33231a;border:1px solid #9b6637;border-radius:9px;color:#ffcf9d}.meta{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}.meta span{border:1px solid var(--line);padding:5px 10px;border-radius:6px}section{margin-top:20px}.architecture{width:100%;max-height:300px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}th,td{padding:10px;text-align:right;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}th{color:#afc7e6;background:#182538}tr:nth-child(4){background:#162c38}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin:14px 0}select,button{background:#162537;color:var(--text);border:1px solid #3c506a;border-radius:5px;padding:8px}button{cursor:pointer}button.active{background:#2a5474}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}.frame{border:1px solid var(--line);border-radius:7px;background:#0b1420;padding:8px}.frame img{width:100%;display:block;image-rendering:pixelated}.caption{padding:3px 4px;color:#b8cce1}.roi img{max-width:none;display:block;image-rendering:pixelated}.roi{overflow:auto;border:1px solid var(--line);padding:8px}details{background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:10px 14px;margin:14px 0}summary{cursor:pointer;color:#b9d5f6}.small{font-size:12px}a{color:#8bc9ff}pre{white-space:pre-wrap;font-size:12px;color:#b5c8de}@media(max-width:760px){main{padding:14px}.pair{grid-template-columns:1fr}h1{font-size:23px}}</style>
<main><h1>道路教师高频：读取方式 × 学习许可</h1><div class="notice">'''+note+'''</div><div class="meta"><span>2帧优化 · 3帧诊断</span><span>共同写域 89,887</span><span>正式 3,400 更新</span><span>K 77,266 精确保护</span><span>无新 TRELLIS 推理</span></div><img class="architecture" src="architecture.svg" alt="输入、材质场、读取组件与原生渲染架构">
<h2>固定预算结果</h2><div class="scroll"><table><thead><tr><th>方法</th><th>实拍 MAE</th><th>孔洞 MAE</th><th>阴影 MAE</th><th>孔洞 HF1 MAE</th><th>阴影 HF1 MAE</th><th>孔洞 HF余弦</th><th>阴影 HF余弦</th><th>诊断 Ey ×1000</th><th>Ey/Ex</th></tr></thead><tbody>'''+''.join(table)+'''</tbody></table></div><p>误差越低越好。诊断 Ey 为134/179/198固定域的纵向二阶差分；横纹沿纵向变化，仍需结合图像判断。教师许可的匹配比较为 M00↔M10、M01↔M11；读取比较为 M10↔M11。</p>
<h2>同尺度图像对照</h2><div class="controls"><label>帧 <select id="raw"><option>86</option><option selected>110</option><option>134</option><option>179</option><option>198</option></select></label><label>视图 <select id="view"><option value="road_rgb">原生道路层</option><option value="joint_rgb">联合场景</option></select></label><label>左 <select id="left"></select></label><label>右 <select id="right"></select></label></div><div class="pair"><div class="frame"><div class="caption" id="lc"></div><img id="li" alt="左侧对照"></div><div class="frame"><div class="caption" id="rc"></div><img id="ri" alt="右侧对照"></div></div><p>相同曝光、原生480×320；WebP无损。原始观测的车下区域没有GT；C/S是冻结伪教师。诊断视角无教师，不是隐藏测试集。</p>
<h2>结果产生前固定的放大区域</h2><div class="controls"><select id="roi"></select></div><div class="roi"><img id="crop" alt="所有组的固定区域对照"></div><p id="roinote"></p>
<details open><summary>高频、伪细节与保护边界</summary><p>HF使用各自域内归一化σ1/2像素滤波，不将K或黑底边缘当成纹理；局部支持不足单列。边缘匹配容差1像素，阈值4/255；同时报告教师平坦区的伪高频能量。</p><div class="scroll"><table id="detailtable"></table></div><p>A完整16,904；C10,830；S3,131；完整S3,187；新覆盖909；残薄56；道路D38,416。A中实际可写αT为0的2,962像素与材质读取无支持7像素分别记账。点像素范围下界仅为固定basis数值代理，不能解释所有误差。</p></details>
<details><summary>方法与可复现证据</summary><p>每组200+300+300步；高斯平均/像素读取分别共享前200步。几何、覆盖、尺度、不透明度、原观测账本不变。12cm低频界0.30；2cm高频界0.34。</p><p>T11仅改变B入口高频初值，使用已有TRELLIS纹理的受限部分投影；严格忠实转移门失败，探索性非零初值有效。本列不是TRELLIS完整能力测试。</p><p>像素资产以持久世界场、读取算子、真实原生basis重放；不能用单个烘焙RGB文件冒充等价GS。历史旧材质/覆盖结果仅作背景，不是本轮因果对照。</p><p>实际渲染重投影对应固定为86→110的2,680处和110→86的4,470处；分离联合αT可见性与足迹过滤，只作诊断，不以同场重复查询制造零误差。</p><p><a href="results.json">完整结果</a> · <a href="report_data.json">逐帧指标与来源</a> · <a href="report.md">技术报告</a></p><pre>Q/gpu_20261119/formal_r1/{M00,M10,M01,M11,T11}/C800\nQ/gpu_20261119/evaluation/frozen_r2/manifest.json\nQ/gpu_20261119/evaluation/results_r2\nQ/gpu_20261119/native_qualification_r2</pre></details></main>
<script>const D='''+scriptdata+''';const $=x=>document.getElementById(x);let keys=['N_parent','teacher_A','teacher_C','teacher_S','pixel_initial','pixel_common_A200','avg_initial','avg_common_A200','M00','M10','M01','M11','T11'];for(let id of['left','right'])for(let k of keys){let o=new Option(D.labels[k]+' ('+k+')',k);$(id).add(o)}$('left').value='M10';$('right').value='M11';function update(){let r=+$('raw').value,v=$('view').value;for(let [sel,im,cap]of[['left','li','lc'],['right','ri','rc']]){let k=$(sel).value,teacher=k.startsWith('teacher_');$(im).src=teacher&&r>110?'':teacher?'assets/'+k+'_'+String(r).padStart(3,'0')+'.webp':'assets/'+k+'_'+String(r).padStart(3,'0')+'_'+v+'.webp';$(cap).textContent=D.labels[k]+(teacher&&r>110?'：此帧无教师':'')}let old=$('roi').value;$('roi').innerHTML='';D.rois.forEach((a,i)=>{if(a.raw===r&&a.bbox_xyxy)$('roi').add(new Option(a.label,i))});if([...$('roi').options].some(o=>o.value===old))$('roi').value=old;crop()}function crop(){let a=D.rois[+$('roi').value];if(!a)return;$('crop').src='assets/ROI_'+String(a.raw).padStart(3,'0')+'_'+a.key+'_'+$('view').value+'.webp';$('roinote').textContent='固定范围 '+a.bbox_xyxy.join(', ')+'；3倍最近邻显示，不锐化。'}for(let id of['raw','view','left','right'])$(id).onchange=update;$('roi').onchange=crop;let t='<tr><th>方法 / 域</th><th>HF余弦</th><th>预测/教师HF能量</th><th>边缘F1</th><th>平坦区伪HF RMS</th><th>HF有效 / 边界不足</th></tr>';for(let k of['M00','M10','M01','M11','T11'])for(let region of['C','S']){let h=D.details[k].HF[region],e=D.details[k].edges[region],f=x=>x==null?'—':x.toFixed(5);t+='<tr><td>'+k+' / '+(region==='C'?'孔洞':'阴影')+'</td><td>'+f(h.HF_cosine)+'</td><td>'+f(h.HF_energy_ratio)+'</td><td>'+f(e.F1)+'</td><td>'+f(e.spurious_HF_RMS_on_teacher_flat)+'</td><td>'+h.qualified_pixels+' / '+h.unsupported_boundary_pixels+'</td></tr>'}$('detailtable').innerHTML=t;update();</script></html>'''
    htmltext=htmltext.replace('</style>','.pair{grid-template-columns:repeat(4,minmax(280px,1fr));overflow-x:auto}@media(max-width:760px){.pair{grid-template-columns:repeat(4,minmax(280px,75vw))}}</style>')
    htmltext=htmltext.replace('结果产生前固定的放大区域','读取结果前固定的放大区域')
    htmltext=htmltext.replace('相同曝光、原生480×320；WebP无损。','相同曝光、原生480×320；WebP无损。指标使用原生预乘sRGB与固定α×教师，不除α。')
    htmltext=htmltext.replace('Q/gpu_20261119/evaluation/results_r2','Q/gpu_20261119/evaluation/results_r3')
    htmltext=htmltext.replace('<details><summary>方法与可复现证据</summary>',
        '<details><summary>零更新来源足迹诊断</summary><p>仅用C/S真实像素足迹的中位尺度σUV=(0.822528,0.299986)，不扫参数、不训练。诊断Ey约降15%，但孔洞/阴影HF误差由0.004929/0.006711变为0.005340/0.007523，颜色误差亦退化；不能视为已解决条纹。</p></details><details><summary>方法与可复现证据</summary>')
    htmltext=htmltext.replace('<p>像素资产以持久世界场',
        '<p>M11最终field在179诊断帧以fresh原生basis重载，road/joint RGB、α与K差值0（2前向、0更新）。双reader五帧smoke重放另有证据；不冒称其余每组均做了五帧fresh-basis验收。</p><p>ROI仅基于冻结mask/teacher，先于首次结果读取固定；文件晚于GPU完成11秒，不冒称GPU前预注册。</p><p>像素资产以持久世界场')
    htmltext=htmltext.replace('本列不是TRELLIS完整能力测试。','来源仅为已有2.56m单patch，实际HF初值上限±0.02，不替换A0底色。本列不是TRELLIS完整能力测试，也不是整街先验。')
    htmltext=htmltext.replace("teacher&&r>110?'':teacher?","teacher&&r>110?'assets/no_teacher.svg':teacher?")
    htmltext=htmltext.replace('<img id="li" alt="左侧对照">','<img id="li" alt="左侧对照"><div id="teacherAbsent" hidden style="min-height:180px;padding:70px 12px;text-align:center;color:#b8cce1">此诊断帧无二维教师</div>')
    htmltext=htmltext.replace('</div></div><p>相同曝光',
        '</div><div class="frame"><div class="caption">像素着色 · 教师高频 (M11)</div><img id="m11fixed" alt="M11固定列"></div><div class="frame"><div class="caption">同配置 + TRELLIS高频种子 (T11)</div><img id="t11fixed" alt="T11固定列"></div></div><p>相同曝光')
    htmltext=htmltext.replace("$('left').value='M10';$('right').value='M11';","$('left').value='teacher_S';$('right').value='M10';")
    htmltext=htmltext.replace('</script></html>',"""</script><script>const baseUpdate=update;update=function(){baseUpdate();let absent=$('left').value.startsWith('teacher_')&&+$('raw').value>110;$('li').style.display=absent?'none':'block';$('teacherAbsent').hidden=!absent;for(let a of['M11','T11'])$(a==='M11'?'m11fixed':'t11fixed').src='assets/'+a+'_'+String(+$('raw').value).padStart(3,'0')+'_'+$('view').value+'.webp'};for(let id of['raw','view','left','right'])$(id).onchange=update;update();</script></html>""")
    htmltext=htmltext.replace('<label>左 <select id="left">','<label>教师 / 参考 <select id="left">').replace('<label>右 <select id="right">','<label>对照方法 <select id="right">')
    ideal_html='<details><summary>理想教师下的读取能力：五帧对比度保持率（30原生前向 / 0更新）</summary><div class="scroll"><table><tr><th>帧</th><th>12cm 高斯平均</th><th>12cm 像素</th><th>24cm 高斯平均</th><th>24cm 像素</th></tr>'+''.join(ideal_rows)+'</table></div><p>独立世界标线函数，保持旧中心采样真值；共同2cm源网格，使用真实原生αT，标线减沥青以消去只读分支。raw86的12cm像素保持率0.880仍低于旧0.90门：源网格中心约0.944，像素足迹平均进一步衰减。合成读取控制不等于真实教师训练成功，也不是3DGS上限证书。</p></details>'
    htmltext=htmltext.replace('<h2>固定预算结果</h2>',ideal_html+'<h2>固定预算结果</h2>')
    if M12 is not None:
        ey11=np.mean([x['stripe']['vertical_E']for x in details['M11']['rows'][2:]]);ey12=np.mean([x['stripe']['vertical_E']for x in M12['rows'][2:]])
        appendix='<h2>来源足迹约束：单一追加分支</h2><p>从共同像素200步重启，同一800步逻辑预算；额外600正式 / 7工程。固定几何、许可、学习率与损失，不扫滤波尺度。</p><div class="scroll"><table><tr><th>方法</th><th>孔洞 MAE</th><th>阴影 MAE</th><th>孔洞 HF1</th><th>阴影 HF1</th><th>诊断 Ey×1000</th></tr>'
        for name,v in [('M11',details['M11']),('M12',M12)]:
            vals=[LABELS[name]+' ('+name+')',fmt(v['pooled']['C']['MAE']),fmt(v['pooled']['S']['MAE']),fmt(v['detail']['C']['1px']['HF_MAE']),fmt(v['detail']['S']['1px']['HF_MAE']),fmt((ey11 if name=='M11'else ey12)*1000,2)]
            appendix+='<tr>'+''.join('<td>'+html.escape(x)+'</td>'for x in vals)+'</tr>'
        appendix+='</table></div><p>诊断Ey下降约7–9%，但教师颜色、高频误差与平坦区伪HF变差，仍未同时解决细节与条纹。原5组结论不变，不晋升。</p>'
        htmltext=htmltext.replace('<h2>同尺度图像对照</h2>',appendix+'<h2>同尺度图像对照</h2>')
        htmltext=htmltext.replace('正式 3,400 更新','正式 4,000 · 原5组3,400 + 分支600')
        htmltext=htmltext.replace('<select id="roi"></select>','<select id="roi"></select><select id="roiSet"><option value="primary">原5组完整对照</option><option value="M12">来源足迹分支对照</option></select>')
        htmltext=htmltext.replace('</script></html>',"""</script><script>$('right').add(new Option(D.labels.M12+' (M12)','M12'));const primaryCrop=crop;crop=function(){primaryCrop();if($('roiSet').value==='M12'){let a=D.rois[+$('roi').value];$('crop').src='assets/M12_ROI_'+String(a.raw).padStart(3,'0')+'_'+a.key+'_'+$('view').value+'.webp'}};$('roiSet').onchange=crop;update();</script></html>""")
    (out/'index.html').write_text(htmltext,encoding='utf-8')
    print(str(out/'index.html'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--out',required=True);a=p.parse_args();make(a.inputs,a.out)

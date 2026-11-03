"""CPU-only Chinese C2 report, architecture SVG and 49-frame synchronized player."""
from pathlib import Path
import argparse
import html
import json
import os
import shutil
import subprocess

RUN = Path(__file__).resolve().parent.parent
REMOTE_RUN = '/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a37_C2_20261103_r1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def esc(value):
    return html.escape(str(value))


def number(value, digits=6):
    return '未测' if value is None else f'{value:.{digits}f}'


def local_path(value):
    path = Path(value)
    if path.exists():
        return path
    if str(value).startswith(REMOTE_RUN + '/'):
        return RUN / str(value)[len(REMOTE_RUN) + 1:]
    return path


def architecture():
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="330" viewBox="0 0 1100 330">
<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#496879"/></marker></defs>
<rect width="1100" height="330" fill="#f8fbfd"/>
<g fill="#e6f1f7" stroke="#7a9eb1" stroke-width="1.5">
<rect x="20" y="25" width="210" height="75" rx="9"/><rect x="20" y="130" width="210" height="75" rx="9"/>
<rect x="285" y="25" width="230" height="75" rx="9"/><rect x="285" y="130" width="230" height="75" rx="9"/>
<rect x="565" y="25" width="245" height="75" rx="9"/><rect x="565" y="130" width="245" height="75" rx="9"/>
<rect x="865" y="60" width="215" height="135" rx="9"/></g>
<g font-family="Microsoft YaHei,Noto Sans CJK SC,sans-serif" font-size="18" fill="#193b4c" text-anchor="middle">
<text x="125" y="53">原始 RGB / LiDAR</text><text x="125" y="79" font-size="15">199帧 BUILD · 固定身份/位姿</text>
<text x="125" y="158">冻结三维先验候选</text><text x="125" y="184" font-size="15">几何准入与失败凭据</text>
<text x="400" y="53">持久 a6 表面</text><text x="400" y="79" font-size="15">共享材质/局部门控 · 位移≤3cm</text>
<text x="400" y="158">冻结 B10k + 道路 r3</text><text x="400" y="184" font-size="15">原静态时间与环境光保持</text>
<text x="688" y="53">真实轨迹与物体存在集</text><text x="688" y="79" font-size="15">保留a6 · a37决定直接光可见性</text>
<text x="688" y="158">同一次深度排序合成</text><text x="688" y="184" font-size="15">可选道路遮光 · C2始终启用</text>
<text x="973" y="93">完整图：保留 a37</text><text x="973" y="126">编辑图：仅删 a37</text><text x="973" y="163" font-size="15">49帧重建 / 保车 / 道路验收</text>
<text x="550" y="276" font-size="17">参考颜色是固定观测辐射代理，非已知 albedo；没有独立可学习阴影图。</text>
<text x="550" y="306" font-size="16">BUILD 指标不能替代独立真值或人工视觉验收。</text></g>
<g fill="none" stroke="#496879" stroke-width="2" marker-end="url(#a)">
<path d="M230 63H285"/><path d="M230 168H258V89H285"/><path d="M515 63H565"/>
<path d="M515 168H565"/><path d="M688 100V130"/><path d="M810 168H865"/>
<path d="M400 100V118H540V149H565"/></g></svg>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, default=RUN / 'final_acceptance')
    parser.add_argument('--output', type=Path, default=RUN / 'report')
    parser.add_argument('--extra-json', type=Path, action='append', default=[], help='Candidate selection, failure ledger or engineering-replay receipts')
    parser.add_argument('--candidate-preview', type=Path, action='append', default=[])
    parser.add_argument('--video', action='store_true', help='Also encode four 49-frame MP4 streams on CPU')
    args = parser.parse_args()
    evaluation = args.evaluation
    metrics = read(evaluation / 'metrics.json')
    summary = metrics['summary']
    contract = read(evaluation / 'contract.json')
    acceptance = summary['acceptance']
    main_title='C2未通过：没有超过身份10k+道路修复1k基线' if not acceptance['automatic_numeric_checks_pass'] else 'C2数值结果待人工视觉复核'
    visibility_enabled = bool(contract['checkpoint_contract']['settings'].get('road_visibility', False))
    gates_enabled = bool(contract['checkpoint_contract']['settings'].get('actor_gates', False))
    contributor_binding = summary.get('contributor_binding',{})
    contributor_note = ('贡献候选域来自其他proposal；此对照不是纯初始化匹配实验。' if contributor_binding.get('source_asset_differs_from_actor_asset') else '贡献候选与当前资产的来源记录见JSON。')
    oracle_path=RUN/'evidence/readout_oracle/summary.json'
    oracle=read(oracle_path) if oracle_path.exists() else None
    oracle_rows=''
    if oracle:
        for row in oracle['rows']:
            alpha=row['actor_alpha'];colour=row['colour_range']['delete']
            oracle_rows+=f'<tr><td>{row["raw"]}</td><td>{row["visible_evidence_pixels"]}</td><td>{number(alpha["standalone"]["mean"],4)}</td><td>{number(alpha["C2_delete"]["mean"],4)}</td><td>{number(alpha["C2_full"]["mean"],4)}</td><td>{number(row["static_road_blocking"]["static_alpha_T_mean"],4)}</td><td>{number(colour["actual_C2_srgb_MAE"])}</td><td>{number(colour["strong_r3_srgb_MAE"])}</td><td>{number(colour["conditional_colour_L1_lower_bound"])}</td></tr>'
    oracle_section=(f'<h2>冻结状态读出诊断：覆盖、混合与颜色范围</h2><p>仅raw105/113/130的严格真实可见域，0优化更新；几何、opacity、门控均固定。临时把全部actor颜色logit设为−30/+30得到逐像素合法RGB范围[L,U]，随后逐张量精确恢复。三个锚standalone α均接近1，joint αT显著较低，主要存在静态混合阻挡；a37存在还会小幅降低贡献。本次可见域road贡献为0。</p><table><tr><th>raw</th><th>可见像素</th><th>standalone α</th><th>delete αT</th><th>full αT</th><th>static αT</th><th>实际C2删车MAE</th><th>强r3 MAE</th><th>条件颜色L1下界</th></tr>{oracle_rows}</table><p>下界只条件于当前冻结几何/混合，忽略共享多视角材质的更强耦合，是乐观逐像素界。实际误差仍明显高于此界；不能据此断言整个方法全局不可达，也不能把“先验看起来正常”当作成像接线通过。临时oracle颜色不是方法结果。</p>') if oracle else ''
    experiment_contract=read(RUN/'experiment_contract.json') if (RUN/'experiment_contract.json').exists() else {}
    ledger=experiment_contract.get('engineering_ledger_before_formal',{})
    accounting=(f'<p>计算记账：正式 {experiment_contract.get("method",{}).get("formal_optimizer_updates","未记录")} 更新；smoke/恢复/pilot额外 {ledger.get("total_extra_optimizer_updates","未记录")} 更新，单列。冻结先验采样步数与优化更新不同：Omni={ledger.get("Omni_sampling_steps","未记录")}，PVD={ledger.get("PVD_sampling_steps","未记录")}。</p>')
    formal_receipt=read(RUN/'formal/receipt.json') if (RUN/'formal/receipt.json').exists() else {}
    denoise_total=ledger.get('Omni_sampling_steps',0)+ledger.get('PVD_sampling_steps',0)
    accounting+=f'<p>先验共 {denoise_total} denoise步、0训练更新；正式优化耗时 {number(formal_receipt.get("seconds"),2)} 秒。正式1000从初始PVD proposal开始，未接着smoke权重训练。完整父源权重未修改。</p>'
    stats_path=RUN/'source/prior/pvd/pointflow_car_train_stats.json'
    transform_path=RUN/'source/prior/pvd/normalization_contract.json'
    stats=read(stats_path) if stats_path.exists() else {}
    transform=read(transform_path) if transform_path.exists() else {}
    stats_section=(f'<h2>PVD checkpoint、精确训练统计与候选资格</h2><p>官方car_completion_3199与真实partial→full点扩散已接通，唯一seed0/1000 DDPM。PointFlow car/train共 {stats.get("car_training_shapes","未记录")} 个形状，每形状 {stats.get("points_per_shape","未记录")} 点，核验 {stats.get("verified_npy_crc_count","未记录")} 个NPY CRC。精确float32训练均值={esc(stats.get("training_mean_xyz"))}，scalar std={esc(stats.get("training_scalar_std"))}；完整训练盒平均span={esc(stats.get("mean_full_bbox_span_xyz"))}。</p><p>使用真实整车标注尺寸和官方train统计变换，不按partial bounds估尺寸或中心。正面符号采用一份固定官方train参考的轴向假设，作者显式front-sign说明仍未确证，没有用目标完整车或符号搜索。PVD点proposal可进入受控优化，不代表最终新视角或隐藏表面已获真值验证。</p>')
    candidate_rows=''
    for filename,label in [('actor_surface_point_seed0_diagnostic_receipt.json','Omni point'),('actor_surface_bbox_seed0_receipt.json','Omni bbox'),('actor_surface_pvd_seed0_receipt.json','PVD actual point completion')]:
        path=RUN/'inputs'/filename
        if path.exists():
            value=read(path);extent=value.get('mesh',{}).get('native_extent_m',value.get('complete_extent_native_m'))
            reason='过宽；未准入，仅诊断' if 'point_seed0' in filename else '欠尺寸，几何拒绝' if 'bbox' in filename else '受控优化入口可用；最终成像未通过'
            candidate_rows+=f'<tr><td>{label}</td><td>{esc(extent)}</td><td>{esc(value.get("formal_eligible"))}</td><td>{reason}</td></tr>'
    stats_section+=f'<table><tr><th>候选</th><th>native L/W/H extent(m)</th><th>formal_eligible</th><th>结论</th></tr>{candidate_rows}</table><p>点先验本身可用与最终C2有效是两项证据；当前失败不能直接宣称先验无效。</p>'
    assert summary['frame_count'] == 49 and len(metrics['rows']) == 49
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    methods = ('native_full', 'C2_full', 'r3_delete37', 'C2_delete37')
    copied = 0
    from PIL import Image
    for raw in range(89, 138):
        folder = out / 'frames' / f'{raw:03d}'
        folder.mkdir(parents=True, exist_ok=True)
        for method in methods:
            source = evaluation / 'frames' / f'{raw:03d}' / f'{method}.png'
            with Image.open(source) as image:
                assert image.size == (480, 320)
                image.verify()
            target = folder / f'{method}.png'
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
            copied += 1
    from PIL import ImageDraw,ImageFont
    font=ImageFont.load_default()
    contact_files=[]
    for kind,top,bottom in [('full','native_full','C2_full'),('delete','r3_delete37','C2_delete37')]:
        sheet=Image.new('RGB',(7*240,7*340+28),'#e8eff4')
        draw=ImageDraw.Draw(sheet);draw.text((10,8),f'{kind}: {top} top / {bottom} bottom',fill='#173445',font=font)
        for index,raw in enumerate(range(89,138)):
            x=(index%7)*240;y=(index//7)*340+28
            draw.text((x+6,y+2),f'raw{raw:03d}',fill='#173445',font=font)
            for row,method in enumerate((top,bottom)):
                with Image.open(out/'frames'/f'{raw:03d}'/f'{method}.png') as image:
                    small=image.resize((240,160),Image.Resampling.LANCZOS)
                sheet.paste(small,(x,y+20+row*160))
        filename=f'{kind}_contactsheet_all49.png';sheet.save(out/filename);contact_files.append(filename)
    (out / 'architecture.svg').write_text(architecture(), encoding='utf-8')
    for name in ('summary.json','contract.json','restore.json','freeze_audit.json','thresholds.json'):
        shutil.copy2(evaluation / name, out / name)
    attachment = out / 'evidence'
    attachment.mkdir(exist_ok=True)
    receipts = [local_path(summary['proposal_receipt'])] if summary.get('proposal_receipt') else []
    receipts += list((RUN / 'inputs').glob('*receipt*.json'))
    receipts += list((RUN / 'evidence').glob('*quality*.json'))
    receipts += list((RUN / 'evidence').glob('*proposal*.json'))
    receipts += list((RUN / 'prior').glob('*contract*.json'))
    receipts += list((RUN / 'source/prior/proposals').glob('*/proposal_report.json'))
    receipts += [path for path in args.extra_json if path.exists()]
    receipts += [path for path in (RUN / 'smoke/receipt.json', RUN / 'formal/receipt.json', RUN / 'evaluation/restore.json') if path.exists()]
    receipts += [path for path in (RUN/'experiment_contract.json',RUN/'evidence/road_measurement_correction_preformal.json',RUN/'evidence/possible_a6_obb_domain/receipt.json',RUN/'evidence/road_visibility_probe/retained_a6_interface/cpu_check.json') if path.exists()]
    if oracle_path.exists():
        receipts.append(oracle_path)
    receipts += [path for path in (stats_path,transform_path,RUN/'source/prior/pvd/checkpoint_download_receipt.json',RUN/'source/prior/pvd/car_stats_download_receipt.json') if path.exists()]
    records = []
    seen = set()
    for index, path in enumerate(receipts):
        path = local_path(path)
        if not path.exists() or path.resolve() in seen:
            continue
        seen.add(path.resolve())
        try:
            data = read(path)
        except (ValueError, OSError):
            continue
        filename = f'{len(records):02d}_{path.name}'
        shutil.copy2(path, attachment / filename)
        excerpt = json.dumps(data, ensure_ascii=False, indent=2)
        suffix = '\n…详见完整JSON' if len(excerpt) > 6000 else ''
        records.append(f'<details><summary>{esc(path.name)}</summary><p><a href="evidence/{esc(filename)}">完整JSON凭据</a></p><pre>{esc(excerpt[:6000])}{suffix}</pre></details>')
    previews = []
    for i, path in enumerate(args.candidate_preview):
        if path.exists():
            filename = f'candidate_{i}{path.suffix}'
            shutil.copy2(path, out / filename)
            previews.append(f'<figure><img src="{esc(filename)}"/><figcaption>{esc(path.name)}</figcaption></figure>')
    full_rows = ''.join('<tr><td>'+label+'</td>'+''.join('<td>'+number(summary['full_image'][method][key])+'</td>' for key in ('psnr','ssim','lpips_vgg'))+'</tr>' for method,label in [('native_full','原基线不编辑流程：Native（强制比较）'),('r3_full','删除后处理直接常驻的压力测试'),('C2_full','C2 始终启用 + a37')])
    a6 = summary['masks']['a6_observed']
    a6_rows = ''.join(f'<tr><td>{esc(method)}</td><td>{number(a6["methods"][method]["gt_srgb_MAE"])}</td></tr>' for method in ('native_full','r3_full','C2_full','r3_delete37','C2_delete37'))
    road_rows = ''.join(f'<tr><td>{name}</td><td>{summary["road"]["r3_delete37"][name]["samples"]:,}</td><td>{number(summary["road"]["r3_delete37"][name]["linear_MAE"])}</td><td>{number(summary["road"]["C2_delete37"][name]["linear_MAE"])}</td></tr>' for name in ('fit','new_dark','original_shadow','inferred'))
    matched_rows = ''
    for name in ('fit','new_dark','original_shadow','inferred'):
        domain=summary['road_domain_accounting'][name]
        base=summary['road_matched']['r3_delete37'][name]
        final=summary['road_matched']['C2_delete37'][name]
        ratio=acceptance['road_core_checks'].get(name,{}).get('ratio')
        matched_rows += f'<tr><td>{name}</td><td>{domain["original_samples"]:,}</td><td>{domain["matched_samples"]:,}</td><td>{domain["excluded_samples"]:,} ({domain["excluded_fraction"]:.2%})</td><td>{number(base["linear_MAE"])}</td><td>{number(final["linear_MAE"])}</td><td>{number(ratio,4)}</td></tr>'
    damage_rows = ''
    for mask in ('other_retained','retained_a5','retained_a26','retained_a18','marking','road_boundary'):
        group = summary['masks'][mask]
        for method in ('C2_full','C2_delete37'):
            value = group['methods'][method]
            prefix = 'paired_baseline_' if 'paired_baseline_change_MAE_8bit' in value else 'baseline_'
            damage_rows += f'<tr><td>{mask}</td><td>{method}</td><td>{group["samples"]:,}</td><td>{number(value[prefix+"change_MAE_8bit"],4)}</td><td>{value[prefix+"change_max_8bit"]}</td><td>{value[prefix+"pixels_changed_over1"]:,}</td></tr>'
    diagnostics = ''.join(f'<tr><td>{method}</td><td>{number(summary["full_image"][method]["psnr"])}</td><td>{number(summary["full_image"][method]["ssim"])}</td><td>{number(summary["full_image"][method]["lpips_vgg"])}</td></tr>' for method in ('r3_delete37','C2_delete37'))
    eligible = acceptance.get('formal_candidate_eligible', False)
    title = '完整重建未达预登记门槛' if not acceptance['full_numeric_pass'] else '数值结果待人工视觉复核'
    if acceptance.get('missing_required_metrics'):
        title = '评价尚不完整：缺少 ' + ', '.join(acceptance['missing_required_metrics']) + ' · ' + title
    if not eligible:
        title = '仅诊断：先验候选未获正式准入 · ' + title
    delta = acceptance['full_vs_original_native_delta']
    status = f'''<div class="alert"><b>{esc(main_title)}</b><p>{esc(title)}。C2 相对原 Native：PSNR {number(delta['psnr'],4)} dB，SSIM {number(delta['ssim'])}，LPIPS {number(delta['lpips_vgg'])}。</p><p>旧r3 always-on full=28.10只是将删除后处理直接常驻的压力测试，不是原baseline不编辑流程bug；真正不编辑基线是Native30.45。本轮保留原基线，不以该压力测试降低门槛。</p><p>human_verdict=null；quality_accepted=false。所有输入属于 BUILD，尚无独立完整车身或清洁道路真值。</p></div>'''
    threshold_text = '完整图门槛：PSNR 下降≤0.05dB、SSIM 下降≤0.001、LPIPS 增加≤0.002，并且无明显可见退化；a6 真实可见域优于强组合基线；matched conservative-road核心指标不明显退化。5%仅为数值容差，不是改善目标。'
    css = '''body{font:16px/1.6 system-ui,"Microsoft YaHei",sans-serif;color:#173445;background:#f4f7fa;margin:0}main{max-width:1250px;margin:auto;padding:28px}h1{font-size:28px}h2{font-size:21px;margin-top:32px}table{border-collapse:collapse;background:white;width:100%;font-size:14px}th,td{padding:9px 12px;text-align:left;border-bottom:1px solid #dae3ea}th{background:#e6eff5}.alert{border-left:6px solid #b54c2d;background:#fff0e9;padding:16px 20px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.grid img{width:100%;display:block}.panel{background:white;padding:10px}.caption{font-weight:600}.controls{position:sticky;top:0;background:#e4eef5;padding:12px;z-index:2}.controls input{width:65%;vertical-align:middle}button{padding:8px 18px;border:0;background:#285d79;color:white;border-radius:5px}figure{margin:10px 0}figure img{max-width:100%}pre{white-space:pre-wrap;word-break:break-all;background:white;padding:12px;font-size:12px}details{margin:12px 0}a{color:#235d7c}.small{font-size:13px;color:#536c7b}'''
    player = '''<div class="controls"><button id="play">播放49帧</button> <input id="frame" type="range" min="89" max="137" value="105"/> <b id="frameLabel">raw105</b></div>
<div class="grid"><div class="panel"><div class="caption">原 Native：保留 a37</div><img id="native_full"/></div><div class="panel"><div class="caption">C2 始终启用：保留 a37</div><img id="C2_full"/></div><div class="panel"><div class="caption">强组合 r3：仅删 a37</div><img id="r3_delete37"/></div><div class="panel"><div class="caption">C2：仅删 a37，保留 a6</div><img id="C2_delete37"/></div></div>
<script>const slider=document.getElementById('frame'),button=document.getElementById('play');let timer=null;function show(){let raw=String(slider.value).padStart(3,'0');document.getElementById('frameLabel').textContent='raw'+raw;for(let id of ['native_full','C2_full','r3_delete37','C2_delete37'])document.getElementById(id).src='frames/'+raw+'/'+id+'.png';}slider.addEventListener('input',show);button.onclick=()=>{if(timer){clearInterval(timer);timer=null;button.textContent='播放49帧';}else{timer=setInterval(()=>{slider.value=Number(slider.value)>=137?89:Number(slider.value)+1;show();},100);button.textContent='暂停';}};show();</script>'''
    video_links = '<p>导出49帧视频：' + ' · '.join(f'<a href="{method}_all49.mp4">{method}</a>' for method in methods) + '</p>' if args.video else ''
    restore = read(evaluation / 'restore.json')
    contact_html=''.join(f'<figure><a href="{name}"><img src="{name}" alt="49帧contact sheet"/></a><figcaption>{"完整图" if name.startswith("full") else "删a37图"}：每格上为原基线、下为C2，覆盖全部49帧。</figcaption></figure>' for name in contact_files)
    content = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(main_title)}</title><style>{css}</style><main>
<h1>{esc(main_title)}</h1>{status}<p>Scene008：仅删a37，保留a6及其他车。</p><p>{threshold_text}</p>
<img src="architecture.svg" alt="输入、冻结组件、a6持久表面、几何遮光与完整/删车输出" style="width:100%"/>
<h2>49帧同步视觉核对</h2><p>四画面共享同一 raw 帧号。上排检查不编辑完整重建，下排检查道路与 a6 保留；不能靠关闭 C2 或修复分支通过上排验收。</p>{player}{video_links}
<details><summary>完整49帧contactsheet（点击可放大）</summary>{contact_html}</details>
<h2>完整重建：必须维持原 Native</h2><table><tr><th>方法</th><th>PSNR ↑</th><th>SSIM ↑</th><th>LPIPS-VGG ↓</th></tr>{full_rows}</table>
<p>原始 RGB 先归一化为浮点，再以 OpenCV INTER_LINEAR 缩放至480×320；三个指标先逐帧计算再49帧平均。a6训练缓存GT与本次原生GT最大差为 {number(summary['a6_view_truth_maxabs'])}。</p>
<h2>a6：真实可见观测域</h2><p>固定 visible_evidence_mask，共 {a6['samples']:,} 像素帧。Native_full为真正原基线不编辑；r3_full为常驻压力测试；r3_delete37为用户指定的身份10k+道路修复1k编辑基线。未知/被遮挡车身不作观测真值；可见域改善也不能证明完整隐藏表面正确。</p><table><tr><th>方法</th><th>sRGB MAE ↓</th></tr>{a6_rows}</table>
<h2>道路核心：保守匹配域与旧全proxy并列</h2><p>真实a6全盒轨迹投影至原Native相机，凸包+2px构成可能被保留a6占据的域；不用PVD/Omni输出或C2图选域，也不把OBB当真实body GT。双方从同一r3道路域排除此possible域，仍使用同一参考、同一分母。未知显露body没有独立RGB真值，不应被安排成road GT。</p>
<p>测量修正由root在正式前提出；凭据在formal进行中落盘，final评分前冻结，未根据final结果调域，不影响训练。legacy actor-list和box-list的顺序关联不可靠，本次依据真实raw track的a6几何，而非actors.index。</p>
<table><tr><th>匹配域</th><th>原像素帧</th><th>有效像素帧</th><th>排除数/比例</th><th>r3 MAE</th><th>C2 MAE</th><th>C2/r3</th></tr>{matched_rows}</table>
<p><b>旧全r3 proxy完整保留：仅历史拟合诊断，不用潜在新a6 body遮路造成的误差上升判失败。</b>误差在线性RGB计算；inferred单列，不当实测真值。</p><table><tr><th>旧全域</th><th>像素帧</th><th>r3删a37 MAE</th><th>C2删a37 MAE</th></tr>{road_rows}</table>
<h2>其他保留车与边界改动</h2><p>下表相对同编辑状态、同道路sun visibility的配对基线（仅关闭新a6分支作为参考；C2评价始终启用），单位8bit。旧r3原组合改动保留在JSON；边界为r3 repair_mask的2px带。合理新a6 body可能改变旧边界，须结合possible域与视觉判断，不能把所有改动自动叫损伤。</p><table><tr><th>保护域</th><th>状态</th><th>像素帧</th><th>MAE</th><th>最大差</th><th>差&gt;1像素数</th></tr>{damage_rows}</table>
<h2>候选可信性与科研限制</h2><p>先验：{esc(summary.get('prior_label',contract.get('prior_label','未记录')))}。资产：{esc(summary['actor_asset'])}。正式准入：{esc(summary.get('asset_formal_eligible'))}；当前为诊断：{esc(summary.get('diagnostic_only'))}；checkpoint step={summary.get('checkpoint_step')}。候选推理成功、网格完整或latent可恢复，均不能替代与真实点云、可见轮廓及多视角原始RGB的一致性验收。</p>
<p>共享材质和≤3cm缺失面法向位移用于局部拟合；没有逐帧材质，新a6表面opacity固定。旧静态贡献候选的局部opacity multiplier本次拟合：{esc(gates_enabled)}；候选不是ownership GT，保护域冲突须单列。{esc(contributor_note)} 道路直接光遮挡模块本次启用：{esc(visibility_enabled)}。固定观测道路颜色不是已知物理albedo，低维直接光可见性只是一项受限标定。全部199帧用于BUILD，排除优化的三个锚帧也不是独立测试。</p>{''.join(previews)}
<h2>工程状态与科研结论分开</h2><p>新进程完整状态恢复：{esc(restore['exact_full_state'])}；评价优化更新：0。freeze/restore 凭据验证参数、Adam、采样器及全部RNG；基线文件、道路参数和评价期间actor状态分别核对。工程接线通过不能覆盖候选几何失败或完整重建质量下降。</p>
{accounting}{oracle_section}
<details><summary>冻结与恢复凭据</summary><pre>{esc(json.dumps(summary['freeze_audit'],ensure_ascii=False,indent=2))}</pre></details>
<details><summary>删除图全图指标：仅诊断，不能判编辑好坏</summary><p>原观测含已删除a37，没有相应清洁道路真值；以下PSNR/SSIM/LPIPS不参与通过判定。</p><table><tr><th>输出</th><th>PSNR</th><th>SSIM</th><th>LPIPS</th></tr>{diagnostics}</table></details>
<details><summary>全部数值检查与测量时序</summary><pre>{esc(json.dumps(acceptance,ensure_ascii=False,indent=2))}</pre></details>
<h2>候选、实验与工程原始凭据</h2>{''.join(records)}<p class="small">人工最终视觉结论尚未填写。human_verdict=null；quality_accepted=false。</p></main></html>'''
    (out / 'result_report.html').write_text(content, encoding='utf-8')
    (out / 'result_summary_zh.txt').write_text(f'''{main_title}\n{title}\n{threshold_text}\n\n原Native完整图：{json.dumps(summary['full_image']['native_full'],ensure_ascii=False)}\nr3删除后处理常驻压力测试（非原baseline不编辑流程）：{json.dumps(summary['full_image']['r3_full'],ensure_ascii=False)}\nC2完整图：{json.dumps(summary['full_image']['C2_full'],ensure_ascii=False)}\n相对原Native：{json.dumps(delta,ensure_ascii=False)}\n完整数值门槛通过：{acceptance['full_numeric_pass']}\n正式候选准入：{eligible}\na6可见域改善：{acceptance['a6_observed_delete_improves']}\n工程恢复：{restore['exact_full_state']}\nhuman_verdict=null; quality_accepted=false\n全部199为BUILD，无独立清洁道路/完整隐藏车身真值。\n删除全图PSNR不参与编辑质量判定。\n''', encoding='utf-8')
    video_checks = {}
    if args.video:
        import imageio_ffmpeg
        executable = imageio_ffmpeg.get_ffmpeg_exe()
        for method in methods:
            video = out / f'{method}_all49.mp4'
            subprocess.run([executable,'-hide_banner','-loglevel','error','-y','-framerate','10','-start_number','89',
                '-i',str(out / 'frames/%03d' / f'{method}.png'),'-frames:v','49','-c:v','libx264','-threads','1','-crf','17','-pix_fmt','yuv420p',str(video)],check=True)
            iterator = imageio_ffmpeg.read_frames(str(video))
            metadata = next(iterator)
            count = sum(1 for _ in iterator)
            assert count == 49
            video_checks[method] = dict(frames=count,fps=metadata['fps'],file=video.name)
    verification = dict(frame_pngs_verified=copied, expected_frame_pngs=196, synchronized_player='four images, shared raw index89..137, common autoplay timer',
        architecture_svg=True, contact_sheets=dict(files=contact_files,frames_per_sheet=49),report_language='zh-CN', videos=video_checks,
        human_verdict=None, quality_accepted=False, GPU_used=False)
    from html.parser import HTMLParser
    from urllib.parse import unquote
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ('src','href') and value:self.links.append(value)
    links=Links();links.feed(content)
    missing=[link for link in links.links if not link.startswith(('#','http:','https:','data:')) and not (out/unquote(link)).exists()]
    verification['static_report_links_checked']=len(links.links)
    verification['missing_report_links']=missing
    assert not missing,missing
    (out / 'report_verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str(out / 'result_report.html'),verification=verification),ensure_ascii=False))


if __name__ == '__main__':
    main()

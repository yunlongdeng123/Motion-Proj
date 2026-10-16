import json, html, hashlib, shutil
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from PIL import Image

out=Path('outputs/shadow_diagnosis'); out.mkdir(exist_ok=True)
summary=json.loads((out/'measured_summary.json').read_text())
focus=json.loads((out/'focus/focus_GPU_receipt.json').read_text())
cpu=json.loads((out/'cpu/attribution.json').read_text())
fcpu=json.loads((out/'focus/focused_attribution_review.json').read_text())
gpu=json.loads((out/'gpu/GPU_diagnostic_receipt.json').read_text())
time=datetime.now(ZoneInfo('Asia/Singapore')).isoformat()
sources=out/'scripts';sources.mkdir(exist_ok=True)
for name in ['diagnose_shadow_cpu.py','render_shadow_gpu.py','shadow_math.py','experiment_plan.json','audited_gaussian_model.py','focus_cpu.py','focus_gpu.py','analyze.py','build_report.py','validate_delivery.py']:
    shutil.copy2(Path('work/shadow_diagnosis')/name,sources/name)
rows=[]
for observation in cpu['real_observations']:
    r=dict(observation)
    fid=r['frame_id']
    if fid in ['0008089','0008095','0008113']:
        r.update(shadow_state='visually reviewed: ordinary exposed road at probe center',other_vehicle_occlusion='no visible occluder at probe center',clean_background_observation_confirmed=True,review_limitation='road correspondence follows the fitted-plane ray hypothesis; no independent per-pixel ground-depth truth')
    elif fid=='0008101':
        r.update(shadow_state='visually reviewed: vehicle-associated road darkening/contact shadow outside body mask; exact material/light decomposition unresolved',other_vehicle_occlusion='no body at probe center in selected masks; visually adjacent to target vehicle',clean_background_observation_confirmed=False)
    else:
        r['review_status']='included in viewed 49-crop sheet; no per-frame semantic certification'
    rows.append(r)
(out/'cross_view_review.json').write_text(json.dumps({'reviewed_at':time,'three_49_crop_sheets_viewed':True,'full_shadow_GT_mask_created':False,'rows':rows},indent=2))
diagram='''<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="325" viewBox="0 0 1120 325">
<defs><marker id="a" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0L10,4L0,8" fill="#345675"/></marker></defs>
<rect width="1120" height="325" rx="16" fill="#f2f6fa"/>
<g font-family="Segoe UI,Microsoft YaHei,sans-serif" text-anchor="middle" font-size="18" fill="#17354d">
<rect x="20" y="55" width="220" height="95" rx="12" fill="#dceafa" stroke="#86a5c7"/><text x="130" y="87">输入：raw101 + 49帧 GT</text><text x="130" y="117">body mask · 相机 · 道路平面</text>
<rect x="285" y="55" width="230" height="95" rx="12" fill="#dceafa" stroke="#86a5c7"/><text x="400" y="87">实际 30k checkpoint</text><text x="400" y="117">完整静态点集 · 原始 GN11</text>
<rect x="560" y="55" width="245" height="95" rx="12" fill="#dceafa" stroke="#86a5c7"/><text x="682" y="87">CUDA α×T 贡献追踪</text><text x="682" y="117">固定相机 · 运动 / 显隐冻结</text>
<rect x="850" y="55" width="250" height="95" rx="12" fill="#d9eee5" stroke="#6ca891"/><text x="975" y="87">输出：观测与表示分开判断</text><text x="975" y="117">真车影证据 · 几何 · 时间机制</text>
<rect x="285" y="205" width="520" height="78" rx="12" fill="#fff1d7" stroke="#cfaa68"/><text x="545" y="234">真实路径：车身外 RGB → 静态黑色高斯</text><text x="545" y="261">时间 marginal 决定何时可见 → 删除车辆后仍残留</text>
</g><g stroke="#345675" stroke-width="3" fill="none" marker-end="url(#a)"><path d="M240 103H282"/><path d="M515 103H557"/><path d="M805 103H847"/><path d="M400 150V201"/><path d="M805 245H975V154"/></g></svg>'''
(out/'architecture.svg').write_text(diagram,encoding='utf-8')
groups=focus['groups']; near=groups['near_black_SH_max_lt_005']['mean_alpha_T']
group_rows=[]
labels={'near_black_SH_max_lt_005':'近黑色高斯：SH RGB 每通道均 &lt;0.05','near_black_center_abs_height_le_015m':'其中：中心距道路平面 ≤0.15m','near_black_center_abs_height_015_to_050m':'其中：中心距道路平面 0.15–0.50m','near_black_center_abs_height_gt_050m':'其中：中心距道路平面 &gt;0.50m','near_black_normal_3sigma_gt_050m':'另一个可重叠指标：法向 3σ 半厚度 &gt;0.50m'}
for key,label in labels.items():
    v=groups[key]
    group_rows.append(f'<tr><td>{label}</td><td>{v["share_of_total_alpha_T"]*100:.2f}%</td><td>{v["mean_alpha_T"]/near*100:.2f}%</td></tr>')
temporal_rows=[]
for r in summary['fixed_time']:
    temporal_rows.append(f'<tr><td>{r["frame_id"]}</td><td>{r["normal_vs_freeze_position_MAE"]:.5f}</td><td>{r["normal_vs_freeze_marginal_MAE"]:.5f}</td><td>{r["freeze_both"]["MAE_to_anchor"]:.1f}</td></tr>')
probe_rows=[]
for i,r in enumerate(fcpu['probes']):
    clear=next(o for o in rows if o['probe']==i and o['frame_id']=='0008089')
    after=next(o for o in rows if o['probe']==i and o['frame_id']=='0008113')
    fmt=lambda rgb: '('+', '.join(f'{x:.3f}' for x in rgb)+')'
    probe_rows.append(f'<tr><td>{r["xy"]}</td><td>{fmt(clear["RGB"])}</td><td>{fmt(r["GT_RGB"])}</td><td>{fmt(after["RGB"])}</td><td>0</td></tr>')
report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>DrivingEditor raw101：实际 GPU 黑块诊断</title>
<style>body{{font:17px/1.65 "Segoe UI","Microsoft YaHei",sans-serif;color:#24384b;background:#f4f7fa;margin:0}}main{{max-width:1100px;margin:auto;background:white;padding:34px}}h1{{font-size:30px}}h2{{margin-top:36px;font-size:23px}}img{{width:100%;height:auto;border-radius:8px}}table{{border-collapse:collapse;width:100%;font-size:15px;margin:18px 0}}th,td{{border:1px solid #cad5df;padding:10px;text-align:left}}th{{background:#eaf1f7}}.lead{{padding:18px;background:#e3f2eb;border-left:5px solid #3b8c6e}}.note{{padding:15px;background:#fff5de}}code{{word-break:break-all;font-size:13px}}a{{color:#17669b}}figure{{margin:22px 0}}figcaption{{font-size:14px;color:#566b7c}}</style><main>
<h1>raw101 黑块：真实车影被静态层拟合，时间显隐起主要作用</h1>
<p>完成时间：{html.escape(time)}（Asia/Singapore）。对象：scene0008_089_137 / 0008101，目标 SUV track 3S_GJHyCGGK4ypFPVKOOqA；完整背景 915,840 点。按用户指令已使用实际 GPU，训练更新 0，checkpoint 写入 0，永久删点 0。</p>
<p class="lead"><b>诊断：</b>实际 GT 中有随车辆出现的道路暗区；模型将它编码为静态分支中的近黑色高斯，主要通过时间边缘权重（marginal）控制出现和消失。部分贡献来自偏离道路或过厚的高斯，形成混合表示。仅凭尖角外形把整块判成“悬空假几何”，与本轮贡献和冻结实验不符。</p>
<img src="architecture.svg" alt="输入、checkpoint、CUDA贡献追踪、冻结消融与诊断输出组件图">
<h2>1. 真实观测：有车时变暗，之前和之后能看到普通路面</h2>
<p>三个探针均在当前 body mask 外。对实际 GT 的 49 帧 × 3 个道路投影探针裁剪进行了查看，明确核对 0008089、0008095、0008113 的暴露路面和 0008101 的车辆邻接暗区。道路对应使用已有道路平面与真实相机投影，仍有平面近似误差。目标暗区与车辆共同出现，支持车影／接触阴影来源；整个黑块没有独立阴影标注，边界污染仍不能全部排除。</p>
<table><tr><th>目标探针 xy</th><th>之前 0008089 GT RGB</th><th>车辆经过 0008101 GT RGB</th><th>之后 0008113 GT RGB</th><th>目标 body mask</th></tr>{''.join(probe_rows)}</table>
<figure><img src="real_road_before_during_after.png"><figcaption>红框中心跟踪同一道路平面假设点。图中保留了导出时的 shadow? 字样，最终逐点复核记录见 cross_view_review.json。</figcaption></figure>
<h2>2. 原始 GN11 已有暗斑，刚体修复后轮廓更尖</h2>
<figure><img src="target_comparison.png"><figcaption>同一相机、同一时间。红色多边形是人工查看后描出的 1,287 像素黑块范围，仅用于诊断，不是阴影 GT。与 body mask 重叠 14 像素（1.09%）。</figcaption></figure>
<p>黑块并非本次刚体修复才新生：原始 GN11 静态层已在相同位置留下暗斑。当前静态渲染在该范围内与含真实车影的 GT 的 RGB MAE 为 {summary['anchor']['MAE_to_GT']:.5f}（0–1），说明它很好地拟合了观测暗区，但这不等于获得了删除车辆后的正确光照。</p>
<h2>3. 固定相机消融：显隐远强于位移</h2>
<p>固定 raw101 相机，把场景时间改为 7 个时刻。位置冻结时，所有点使用 raw101 的位置；显隐冻结时，所有点使用 raw101 的时间 marginal（包含透明度加权和预筛选）。两者都冻结时应重现 raw101；本轮 7 个时刻均逐位一致。</p>
<figure><img src="temporal_ablation.png"><figcaption>hold position：固定位置，黑块仍能随时间消失。hold marginal：固定显隐，车辆不在场的时刻仍保留黑块。位置变化会改变残留形状，但不是本例的主要时间机制。</figcaption></figure>
<table><tr><th>场景时间对应帧</th><th>正常 vs 固定位置 MAE</th><th>正常 vs 固定显隐 MAE</th><th>同时固定 vs 锚点 MAE</th></tr>{''.join(temporal_rows)}</table>
<p>排除锚点后的 6 个时刻，前两列平均分别为 <b>0.00704 和 0.21329</b>，相差约 30.3 倍。它们是指定范围内的 RGB 差异，反映干预影响强弱，不能解读成“97% 原因来自显隐”。对同一批近黑色点的追踪也看到了几乎不动、marginal 却从 0 变到接近 1 的实例。</p>
<h2>4. 几何：有异常，但主体不能都算成离地黑片</h2>
<p>使用全静态场景排序和遮挡不变的 CUDA feature 通道，直接累计每组真实 α×T 权重。近黑色指当前相机方向下 SH 评估颜色三个通道均小于 0.05，是外观分组，不是阴影分类。道路平面来自已有 SGCM 拟合：844 个内点网格，整体 RMSE 0.0323m；本轮未新增独立 LiDAR 表面真值。</p>
<table><tr><th>表示属性</th><th>占全部 α×T 质量</th><th>占近黑色 α×T 质量</th></tr>{''.join(group_rows)}</table>
<p>在近黑色贡献中，中心 ≤0.15m 的占 <b>65.19%</b>，0.15–0.50m 占 26.75%，&gt;0.50m 占 8.06%。中心靠近道路不保证高斯薄、方向正确或位置可信；法向厚度另列且可重叠。以上比例描述本例表示属性，<b>不是“真实车影／假几何”的语义比例</b>。</p>
<p>异常例：静态 ID 624979 的法向 3σ 半厚度约 0.898m；ID 596846 在原矩形归因中距参考平面约 2.16m，最大尺度轴约 3.04m。它们证明不理想几何存在，但不能据此代表整个黑块。</p>
<figure><img src="causal_hiding_NOT_A_REPAIR.png"><figcaption>隐藏远离平面的近黑色点后，主体黑块仍在；隐藏近地面的近黑色点会显著削弱主体。隐藏全部近黑色点会露出亮斑和结构缺陷，所以这些操作只用于因果诊断，不能当成修复方案。</figcaption></figure>
<h2>5. “背景没有重建出来”不是这块黑色的直接解释</h2>
<p>原始独立 Static α 在黑块范围内均值为 <b>0.999853</b>，三个探针均约 0.9999。此处黑色来自已渲染、不透明的暗色高斯，不能用低透明度或黑色空背景解释。多帧有暴露道路，也削弱“从未观察过”的解释。但这些证据不证明每个被遮挡像素都重建正确；隐藏暗点后出现错误结构，说明干净背景的分层表示仍需处理。</p>
<figure><img src="opacity_check.png"><figcaption>右图固定显示范围 0–1，白色表示独立静态层近乎不透明。</figcaption></figure>
<h2>6. 对四个假设的更新</h2>
<table><tr><th>假设</th><th>实际诊断</th></tr><tr><td>车影被学进静态层</td><td><b>强支持。</b>GT 车外真实暗区、跨帧道路和静态渲染对应；原 GN11 已有暗斑。</td></tr><tr><td>车底／异常高斯错误归属</td><td><b>部分支持。</b>存在离平面、厚度较大的贡献，轮廓可能混有边界拟合；尚不能逐点确认来自轮胎或底盘。</td></tr><tr><td>静态高斯随时间运动</td><td><b>需要修正重点。</b>存在位置自由度，本例主要依赖时间显隐，位置干预影响小得多。</td></tr><tr><td>被遮挡背景缺少观测</td><td><b>作为唯一主因被削弱。</b>三个道路假设点前后均有可见路面，Static α 很高；清洁背景重建充分性仍未全面证明。</td></tr></table>
<p>SGCM 的中心到平面约束和 NPS 的邻域平滑都允许“近地、空间平滑、随时间显隐的黑色高斯”。因此加强速度正则或只清理离地点，均不足以消除本例主体。后续若做修复，应先处理车影的归属与背景有效监督，再处理异常几何；需要对 global RGB、static RGB 和 SSIM 窗口使用一致的阴影／不确定区域策略，不能简单将投影阴影强制设为刚体车辆不透明度。显式车辆关联阴影层是另一条建模路径。本轮未启动这些训练或修改。</p>
<h2>7. 验证、范围和可复现材料</h2>
<p>GPU raw Static 与已有 FP32 保存数组逐位相同（MAE=0、max=0）；feature 追踪不改变 RGB；全部点的分组 α×T 与 Static α 一致；同时冻结位姿时间与 marginal 的 7 次结果逐位一致；checkpoint 文件身份及模型参数版本检查通过。原矩形 CPU 回放 MAE 1.09e-6，黑块多边形 CPU 回放 MAE 2.38e-7。静态层未永久删除任何点。</p>
<p class="note">仅对一个 checkpoint、一个目标车和人工描出的黑块范围成立。49 帧来自训练重建窗口，没有独立留出测试；没有全黑块逐像素阴影真值，也没有每个高斯的物理表面归属真值。因此本轮提供贡献质量、几何属性和因果消融，仍不将两种语义强行分成相加为 100% 的比例。</p>
<ul><li><a href="measured_summary.json">测量汇总</a> · <a href="focus/focus_GPU_receipt.json">精确 GPU 分组贡献</a> · <a href="gpu/GPU_diagnostic_receipt.json">首轮 GPU 验证</a></li><li><a href="focus/focused_attribution_review.json">近黑色点 ID、几何、轨迹与探针</a> · <a href="cross_view_review.json">真实跨帧复核记录</a> · <a href="cpu/attribution.json">原矩形归因</a></li><li><a href="scripts/diagnose_shadow_cpu.py">原始 CPU 脚本</a> · <a href="scripts/render_shadow_gpu.py">原始 GPU 脚本</a> · <a href="scripts/focus_cpu.py">黑块范围归因脚本</a> · <a href="scripts/focus_gpu.py">精确分组与隐藏脚本</a></li></ul>
<p>远程证据目录：<code>{html.escape(str(Path(cpu['checkpoint_identity']['path']).parent.parent.parent.parent/'black_shadow_audit_20261016_v1'))}</code>。checkpoint：<code>{html.escape(cpu['checkpoint_identity']['path'])}</code>。完整运行命令保存在 <a href="reproduce.txt">reproduce.txt</a>。</p>
</main></html>'''
(out/'diagnosis.html').write_text(report,encoding='utf-8')
remote='/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/black_shadow_audit_20261016_v1'
commands=f'''Executed on wm-vgpu-1011, read-only model diagnosis. Training updates: 0.
cd {remote}/prelaunch
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= /root/autodl-tmp/envs/drivingeditor/bin/python -u diagnose_shadow_cpu.py --execute-cpu
PYTHONDONTWRITEBYTECODE=1 /root/autodl-tmp/envs/drivingeditor/bin/python -u render_shadow_gpu.py --execute-gpu --user-opened-card --cpu-attribution {remote}/cpu_attribution/attribution.json
PYTHONDONTWRITEBYTECODE=1 /root/autodl-tmp/envs/drivingeditor/bin/python -u focus_cpu.py
PYTHONDONTWRITEBYTECODE=1 /root/autodl-tmp/envs/drivingeditor/bin/python -u focus_gpu.py

Scripts intentionally reject reuse of an existing output directory to preserve evidence.
For a separate reproduction, stage under a fresh diagnostic directory and update
output paths consistently. Do not overwrite the completed results.
Local images: bundled Python ran analyze.py. Report: build_report.py.
'''
(out/'reproduce.txt').write_text(commands,encoding='utf-8')
manifest={'completed_at':time,'remote_evidence_root':remote,'scene':cpu['scene'],'frame_id':cpu['frame_id'],'GPU_used':True,'training_updates':0,'checkpoint_written':False,'background_points_permanently_removed':0,'source_gaussian_model_sha256':cpu['source_model_sha256'],'checkpoint_identity':cpu['checkpoint_identity'],'full_shadow_truth_available':False,'report_montages_and_three_49_crop_sheets_viewed':True,'files':{}}
for p in sorted(out.rglob('*')):
    if p.is_file() and p.name!='run_manifest.json':
        manifest['files'][p.relative_to(out).as_posix()]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(out/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'report':str(out/'diagnosis.html'),'files':len(manifest['files']),'status':'actual_GPU_diagnosis_complete'},ensure_ascii=False))

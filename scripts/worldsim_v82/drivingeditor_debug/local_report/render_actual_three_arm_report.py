"""Add only completed, independently checked real three-arm endpoint evidence."""
import hashlib
import html
import json
import shutil

import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_actual_three_arm_evidence(here, out, page):
    own = here / 'sgcm_data_audit/long_pair_review'
    interpretation_path = own / 'actual_three_arm_interpretation.json'
    if not interpretation_path.exists():
        return page, False
    interpretation = json.loads(interpretation_path.read_text(encoding='utf8'))
    if interpretation['status'] != 'ACTUAL_NUMERIC_AND_FIXED_IMAGE_INTERPRETATION_COMPLETE':
        raise ValueError('Actual completed qualitative interpretation required')
    root = here / 'continuation_diagnostic/opacity_outside30k_to32000'
    effects_path = root / 'receipts/three_arm_locked_metrics.json'
    effects = json.loads(effects_path.read_text(encoding='utf8'))
    summary = json.loads((root / 'receipts/full49_summary.json').read_bytes())
    review = json.loads((own / 'independent_three_arm_numeric_review.json').read_text(encoding='utf8'))
    if effects['status'] != 'completed_three_arm_locked_endpoint_metric_export' or review['canonical_three_arm_metrics_sha256'] != sha(effects_path):
        raise ValueError('Actual canonical three-arm source differs from independent numeric review')
    target = out / 'evidence/V_opacity_outside30k_to32000'
    target.mkdir(parents=True, exist_ok=True)
    for relative in ('status.json', 'continuation_diagnostic.json', 'first_backward_reference_comparison.json', 'first_step_reference_comparison.json', 'runtime_import_pins.json', 'receipts/full49_summary.json', 'receipts/three_arm_locked_metrics.json', 'receipts/common49_inputs_after_V_cpu.json', 'receipts/gpu_release_handoff.json', 'terminal_32000/diagnostic_snapshot_manifest.json'):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, destination)
    for name in ('actual_three_arm_interpretation.json', 'independent_three_arm_numeric_review.json', 'removal_locality_actual_cpu.json'):
        shutil.copy2(own / name, target / name)
    for name in ('paired_raw10_32000.png', 'outside32000_fixed5.png', 'raw10_long_recovery.png'):
        shutil.copy2(root / 'receipts' / name, out / 'assets' / ('V32000_' + name))
    video = json.loads((out / 'long_three_arm_video_validation.json').read_bytes())
    if video['actual_arm_summary_sha256']['V'] != sha(root / 'receipts/full49_summary.json') or video['frame_count'] != 49 or not video['all_frames_decoded'] or sha(out / 'assets' / video['file']) != video['sha256']:
        raise ValueError('Actual B1/B2/V synchronous49 video and decoded-source receipt required')
    video_html = f'<p><video controls muted playsinline preload="metadata" poster="assets/B1_B2_V32000_synchronous_decoded_raw10.png" style="width:100%;background:#000;border-radius:8px" src="assets/{video["file"]}"></video></p><p class="muted">真实32k三臂49帧同步：从上到下B1、B2、V；每排GT、composite、static、dynamic。所有同相机面板编码在一条时间轴，10fps／4.9秒。新增视频49帧全部CPU解码核对，B1与短窗旧视频保持原字节；浏览器播放未验证。<a href="long_three_arm_video_validation.json">588原PNG SHA及全49帧解码receipt</a> · <a href="assets/V_opacity_outside32000_four_panel_actual49.mp4">V单独四面板视频</a> · <a href="long_V_video_validation.json">V单独视频全帧解码receipt</a></p>'
    lookup = {(r['region'], r['array'], r['metric']): r for r in effects['primary_metrics'] + effects['context_metrics_all_six_domains']}
    def metric(region, array, key):
        return lookup[(region, array, key)]
    def fmt(value):
        return f'{value:.9f}' if abs(value) >= .001 else f'{value:.6e}'
    names = {('alpha_d', 'mean'): '背景动态α均值', ('alpha_d', 'fraction_gt_0_1'): '背景α&gt;0.1占比', ('alpha_d', 'fraction_gt_0_5'): '背景α&gt;0.5占比', ('alpha_d', 'fraction_gt_0_9'): '背景α&gt;0.9占比', ('dynamic', 'rgb_support_gt2_uint8_pixel_fraction'): '背景动态RGB支撑占比', ('static', 'clamped_rgb_mae_vs_gt'): '背景static MAE', ('static', 'clamped_rgb_mse_vs_gt'): '背景static MSE', ('static', 'clamped_rgb_psnr_db'): '背景static PSNR（dB）', ('composite', 'clamped_rgb_mae_vs_gt'): '全图composite MAE', ('composite', 'clamped_rgb_mse_vs_gt'): '全图composite MSE', ('composite', 'clamped_rgb_psnr_db'): '全图composite PSNR（dB）'}
    primary = ''.join(f'<tr><td>{names[(r["array"], r["metric"])]}</td><td>{fmt(r["equal_frame_mean_B1"])}</td><td>{fmt(r["equal_frame_mean_B2"])}</td><td>{fmt(r["equal_frame_mean_V"])}</td><td>{fmt(r["V_vs_each_original"]["B1"]["equal_frame_mean_V_minus_reference"])}</td><td>{fmt(r["V_vs_each_original"]["B2"]["equal_frame_mean_V_minus_reference"])}</td><td>{fmt(r["N_observed_original_repeat_abs_difference_of_means"])}</td></tr>' for r in effects['primary_metrics'])
    context_specs = [('actor', 'composite', 'clamped_rgb_mae_vs_gt', '目标车辆composite MAE'), ('actor', 'composite', 'clamped_rgb_mse_vs_gt', '目标车辆composite MSE'), ('actor', 'composite', 'clamped_rgb_psnr_db', '目标车辆composite PSNR（dB）'), ('actor', 'alpha_d', 'mean', '目标车辆动态α'), ('actor', 'dynamic', 'rgb_support_gt2_uint8_pixel_fraction', '目标车辆动态RGB支撑'), ('non_sky_outside_actor', 'alpha_s', 'mean', '已观测背景静态α'), ('non_sky_outside_actor', 'static_foreground_contribution', 'mean', '背景静态高斯颜色贡献均值'), ('non_sky_outside_actor', 'static_env_contribution', 'mean', '背景静态环境残差贡献均值')]
    context = ''.join(f'<tr><td>{label}</td><td>{fmt(metric(region,array,key)["equal_frame_mean_B1"])}</td><td>{fmt(metric(region,array,key)["equal_frame_mean_B2"])}</td><td>{fmt(metric(region,array,key)["equal_frame_mean_V"])}</td></tr>' for region,array,key,label in context_specs)
    area = interpretation['actor_area_and_existing_canonical_pooled_context']
    pooled = area['canonical_summary_actor_by_arm']
    pooled_rows = ''.join(f'<tr><td>{arm}</td><td>{pooled[arm]["pixel_count_summed"]:,}</td><td>{pooled[arm]["composite"]["pooled_mse"]:.12f}</td><td>{pooled[arm]["composite"]["pooled_psnr_db"]:.9f}</td><td>{pooled[arm]["alpha_d"]["mean"]:.9f}</td></tr>' for arm in ('B1','B2','V'))
    actor_area_html = f'<h3>目标车辆面积与已有像素汇总口径</h3><p>固定body-mask的全49帧合计{area["total_area_pixels"]:,}像素；单帧面积为{area["minimum_area_pixels"]:,}–{area["maximum_area_pixels"]:,}像素，raw45为{area["raw45_area_pixels"]}像素，车辆已接近出画边缘。49帧等权actor PSNR不是整车视觉分数，不能把小面积边缘域的读数描述成整车视觉崩坏。下面直接引用原canonical summary的像素汇总MSE及其pooled PSNR，同时保留全部49帧和原主口径。</p><table><tr><th>actor · 原固定49域</th><th>总像素</th><th>像素汇总composite MSE</th><th>pooled PSNR（dB，先汇总MSE）</th><th>像素加权dynamic α</th></tr>{pooled_rows}</table><p class="muted">像素汇总目标误差也升高，因此量化代价不能仅用边缘帧很小解释；真实raw45整体仍清晰。没有重新定义11个主指标、剔除边缘帧或加入新质量阈值。各帧原面积保留在实际解释JSON和canonical逐帧receipt。</p>'
    adverse = review['frame_adverse_error_screen']
    counts = ''.join(f'<tr><td>{label}</td><td>{adverse[key]["higher_error_vs_B1_count"]}/49</td><td>{adverse[key]["higher_error_vs_B2_count"]}/49</td></tr>' for key,label in [('full/composite','全图composite MSE上升'), ('actor/composite','目标车辆composite MSE上升'), ('non_sky_outside_actor/static','已观测背景static MSE上升')])
    composite = metric('full','composite','clamped_rgb_psnr_db')
    static = metric('non_sky_outside_actor','static','clamped_rgb_psnr_db')
    support = metric('non_sky_outside_actor','dynamic','rgb_support_gt2_uint8_pixel_fraction')
    actor = metric('actor','composite','clamped_rgb_psnr_db')
    distributions = []
    for row,label in [(composite,'全图composite PSNR'), (actor,'目标车辆composite PSNR'), (static,'背景static PSNR')]:
        original = row['original_repeat_paired_absolute_difference_distribution']
        for arm in ('B1','B2'):
            pair = row['V_vs_each_original'][arm]
            dist = pair['paired_absolute_difference_distribution']
            distributions.append(f'<tr><td>{label} · V−{arm}</td><td>{fmt(pair["equal_frame_mean_V_minus_reference"])}</td><td>{fmt(dist["mean"])}</td><td>{fmt(dist["finite_values"]["p95"])}</td><td>{fmt(original["mean"])}</td><td>{fmt(original["finite_values"]["p95"])}</td></tr>')
    fig,axes = plt.subplots(2,2,figsize=(11,7),layout='constrained',facecolor='#101724')
    plot_specs = [('full','composite','clamped_rgb_psnr_db','Composite PSNR, full'), ('non_sky_outside_actor','static','clamped_rgb_psnr_db','Static PSNR, observed background'), ('non_sky_outside_actor','alpha_d','mean','Dynamic alpha, observed background'), ('actor','composite','clamped_rgb_mse_vs_gt','Actor composite MSE')]
    for ax,(region,array,key,title) in zip(axes.flat,plot_specs):
        row = metric(region,array,key)
        pairs = row['V_vs_each_original']
        for arm,color in [('B1','#69c5dc'), ('B2','#e7c45f'), ('V','#f388a1')]:
            frame_rows = pairs['B1']['frames'] if arm=='V' else pairs[arm]['frames']
            values = [f['V_value'] if arm=='V' else f['reference_value'] for f in frame_rows]
            ax.plot(range(49),values,label=arm,color=color,linewidth=1.4)
        ax.set(title=title,xlabel='Raw frame, all49')
        ax.set_facecolor('#101724');ax.tick_params(colors='#e9f0fa');ax.xaxis.label.set_color('#e9f0fa');ax.title.set_color('#e9f0fa')
        for spine in ax.spines.values():spine.set_color('#48556d')
        ax.grid(alpha=.2);ax.legend(facecolor='#182235',labelcolor='#e9f0fa',fontsize=8)
    fig.savefig(out/'assets/V32000_all49_three_arm_metrics.png',dpi=160);plt.close(fig)
    fixed_images = []
    for raw in (0,10,17,20,24,30,40,48):
        for channel,label in (('static','静态／移除'),('dynamic','动态')):
            source = root / 'extra_eval/32000' / channel / f'0017{raw:03}.png'
            name = f'V32000_{channel}_raw{raw:02}.png'
            shutil.copy2(source,out/'assets'/name)
            fixed_images.append(f'<figure><img src="assets/{name}" alt="真实V32000 raw{raw} {label}"><figcaption>预选raw{raw} · {label}</figcaption></figure>')
    local = json.loads((own/'removal_locality_actual_cpu.json').read_text(encoding='utf8'))['domains']['non_sky_outside_actor']['arms']
    locality = ''.join(f'<tr><td>{arm}</td><td>{local[arm]["equal_frame_mean"]["mean_absolute_composite_static_difference"]:.9f}</td><td>{100*local[arm]["equal_frame_mean"]["edited_support_gt2_uint8_pixel_fraction"]:.6f}%</td><td>{100*local[arm]["pixel_weighted_mean"]["edited_support_gt2_uint8_pixel_fraction"]:.6f}%</td></tr>' for arm in ('B1','B2','V'))
    adverse_images = []
    for run, channel, label in ((root, 'gt', '真实GT'), (here / 'continuation_diagnostic/original30k_to32000_noise_reference', 'composite', 'B2原版'), (root, 'composite', 'V候选')):
        name = f'V32000_adverse_raw45_{"B2" if label == "B2原版" else channel}.png'
        shutil.copy2(run / 'extra_eval/32000' / channel / '0017045.png', out / 'assets' / name)
        adverse_images.append(f'<figure><img src="assets/{name}" alt="实际raw45 {label}"><figcaption>raw45 · {label}</figcaption></figure>')
    adverse_html = '<h3>不利帧补充：目标车辆保真</h3><p>按全部49帧的目标车辆MSE检查，raw43–47属于误差代价较大的尾段。这里额外展示raw45：目标已到左侧边缘，V的目标颜色／轮廓更软。此帧在终点后按不利指标选择，属于披露的补充，不能当作预选8帧或独立抽样；全部49帧代价均保留。</p><div class="grid">' + ''.join(adverse_images) + '</div>'
    section = f'''<section class="section" id="long-v-three-arm"><h2>长V已完成：静态背景改善，整体与目标车辆有明确代价</h2><p>V从同一原始30k完整状态恢复，只加λ=0.05掩码外动态透明度项，真实完成2000步并退出0。首步backward前十个原目标标量精确相等；全2000步采样／time_shift／LR对B1和B2一致；四节点评估恢复RNG，末49帧GT与域配对、四件完整终态及保护源／输入hash通过。50k学习率horizon、原mask、其他loss、SGCM、depth、static时间字段和reset均保持；没有进入33k reset。这是恢复实验，不是从零训练或正常效果验收。</p><div class="grid"><div class="metric"><strong>{static['equal_frame_mean_V']:.5f} dB</strong><br><small>V已观测静态背景 · 49帧PSNR均值；原版约11.96</small></div><div class="metric"><strong>{composite['equal_frame_mean_V']:.5f} dB</strong><br><small>V全图composite；对双原版约下降0.53 dB</small></div></div><p>目标车辆composite PSNR从B1的{actor['equal_frame_mean_B1']:.5f}／B2的{actor['equal_frame_mean_B2']:.5f}降至{actor['equal_frame_mean_V']:.5f} dB。全部49帧的背景static MSE均改善，但全部49帧的全图和目标车辆composite MSE均升高。动态背景RGB支撑仍为{100*support['equal_frame_mean_V']:.6f}%，不是背景已经清空。</p><p class="warn">按执行前锁定的解释，本次属于“分配方向改善且有重建／目标保真代价”。不能升级为正常、修复完成或论文复现通过；一次原版重复也不是显著性检验或通用噪声界。</p><table><tr><th>逐帧代价：只计MSE上升</th><th>V对B1</th><th>V对B2</th></tr>{counts}</table><div style="overflow-x:auto"><table><tr><th>锁定11主指标（占比0–1）</th><th>B1等权均值</th><th>B2等权均值</th><th>V等权均值</th><th>V−B1均值差</th><th>V−B2均值差</th><th>原版N=|均值差|</th></tr>{primary}</table></div><table><tr><th>PSNR效应分布（dB）</th><th>均值之差</th><th>逐帧|Δ|均值</th><th>逐帧|Δ| P95</th><th>原重复逐帧|Δ|均值</th><th>原重复逐帧|Δ| P95</th></tr>{''.join(distributions)}</table><p class="muted">正负均值差与逐帧绝对差分开呈现；每项完整49帧signed／abs值在canonical receipt中保留。没有从结果删除帧或更改support&gt;2阈值。目标车辆动态α仍约0.961、RGB支撑仍为1：目标未被全局抹掉，但这些数值不能替代颜色、轮廓及细节保真。</p><img src="assets/V32000_all49_three_arm_metrics.png" alt="三臂完整49帧的全图、已观测静态、动态alpha与目标车辆误差"><table><tr><th>目标保真与背景贡献：49帧等权</th><th>B1</th><th>B2</th><th>V</th></tr>{context}</table><p>静态alpha与高斯本身颜色贡献提高，环境残差贡献下降；静态输出仍含环境贡献。改进不能被归为SGCM、depth或static时变语义已修复，这些变量均未更改。</p><img src="assets/V32000_paired_raw10_32000.png" alt="同一真实raw10，B1 B2 V的GT/composite/static/dynamic及alpha"><p class="muted">同一32k终点、同一raw10，从上到下B1、B2、V。所有图直接来自实际PNG；静态道路彩噪显著下降，动态图道路反射／彩点和纹理仍残留。</p><details><summary>预定8帧真实静态／动态图：raw0/10/17/20/24/30/40/48</summary><div class="grid">{''.join(fixed_images)}</div></details><h3>移除局部性：改善仍有广泛车体外变化</h3><table><tr><th>observed非天空车体外</th><th>comp↔static绝对差等权均值</th><th>编辑支撑 · 49帧等权</th><th>编辑支撑 · 像素加权</th></tr>{locality}</table><p class="muted">补充来自真实NPZ，按既有协议计算mean(abs(clip(composite)−clip(static)))；支撑是两图分别量化8位后任一通道差&gt;2。原mask解码、全六域counts与实际评估一致。它是相对composite的变化，不是edited GT误差；body-mask外可有原车影子，去除影子也可能是合理变化。没有dense edited GT、独立runtime mask像素或actualScene pose快照。</p><p>真实图像中静态道路大幅恢复，V的动态通道仍有道路／背景残留，边缘移除也有颜色和残影风险。完整数值与固定图支持继续排查分配机制，同时要求解释目标车辆代价。下一次训练或修改只在root另行批准后执行；当前没有新GPU任务或正式源变更。</p><div class="links"><a href="evidence/V_opacity_outside30k_to32000/receipts/three_arm_locked_metrics.json">唯一canonical三臂：11主＋126上下文／全部49逐帧效应</a><a href="evidence/V_opacity_outside30k_to32000/independent_three_arm_numeric_review.json">独立CPU逐项核验与不利帧</a><a href="evidence/V_opacity_outside30k_to32000/removal_locality_actual_cpu.json">真实NPZ的预定局部性补充</a><a href="evidence/V_opacity_outside30k_to32000/actual_three_arm_interpretation.json">实际图像复核与有限结论</a><a href="evidence/V_opacity_outside30k_to32000/receipts/full49_summary.json">V完整运行／49帧／四件状态receipt</a><a href="evidence/V_opacity_outside30k_to32000/terminal_32000/diagnostic_snapshot_manifest.json">V四件32000终态</a></div></section>'''
    section = section.replace('<details><summary>预定8帧真实静态／动态图', video_html + adverse_html + '<details><summary>预定8帧真实静态／动态图')
    section = section.replace('<p>静态alpha与高斯本身颜色贡献提高', actor_area_html + '<p>静态alpha与高斯本身颜色贡献提高')
    weighted_support = summary['final49_regions']['non_sky_outside_actor']['dynamic']['rgb_support_gt2_uint8_pixel_fraction']
    section = section.replace('不是背景已经清空。', f'不是背景已经清空。同一区域V动态支撑的像素加权值是{100*weighted_support:.6f}%，与49帧等权值分别呈现。')
    page = page.replace('<section class="section"><h2>下一轮受控实验</h2>',section+'<section class="section"><h2>下一轮受控实验</h2>')
    page = page.replace('长V尚无终点效果结论；今后须同时相对B1和B2报告逐帧效应、composite代价、静态背景、dynamic alpha/support、目标车辆保真与真实移除局部性。', '长V真实同窗结果已纳入下方；静态背景改善与目标／整体代价同时保留，未升级为正常效果。')
    page = page.replace('长V运行已退出0；本页尚未纳入其完整终态与效果配对，不能据此提前给效果结论。','长V真实2000步、49帧和四件终态已完成，独立CPU核验与真实图像解释见下方；仍未达到正常效果验收。')
    page = page.replace('长实验状态：B1／B2完成，V待终点验证','长实验状态：B1／B2／V均完成，效果仍未验收')
    page = page.replace('长V已获单独批准并运行中，目标停在32k，避开下一次33k reset。','长V已完成32k，同窗结果显示静态改善伴随目标／整体代价；没有进入33k reset。')
    page = page.replace('长V须单独批准并验证，停在32k，避开下一次33k reset。','长V也已完成32k；静态背景改善伴随目标／整体代价，完整配对见下方。')
    page = page.replace('长V须单独批准并完成实际运行后才有同窗效果结论。','长V同窗2000步及真实49帧配对已完成，详见下方；整体与目标代价仍需排查。')
    page = page.replace('长V已获单独批准并启动，完整运行及终点配对后才有同窗效果结论。','长V真实同窗结果已完成，静态背景改善与目标／整体代价一并保留。')
    page = page.replace('长V待验证。新增mask外项仍是诊断假设','长窗V已显示背景改善与目标／整体代价，仍未验收。新增mask外项仍是诊断假设')
    page = page.replace('长窗V待验证。新增mask外项仍是诊断假设','长窗2000步也已显示静态背景改善，但仍有动态背景残留及目标／整体代价。新增mask外项仍是诊断假设')
    page = page.replace('当前没有新GPU任务或正式源变更。','本次报告工作没有启动新GPU任务或更改正式源。')
    page = page.replace('完整配对见下方。两份原版用于测量', '完整配对见上方。两份原版用于测量')
    page = page.replace('长实验还需检查组合画质、已观测静态背景、目标车辆保真、车外泄漏和真实移除／平移；不能只用某个指标变好验收。','真实长实验已同时保留组合画质、已观测静态背景、目标车辆保真与车外泄漏；静态改善伴随全部49帧目标／整体误差上升。真实移除仍有残留，平移编辑未完成验收，不能只用某个指标变好验收。')
    page = page.replace('同一 30k 状态只加入掩码外约束，300 步后完整 49 帧的已观测背景静态 PSNR 提高 11.79 dB；整体 PSNR 下降 0.43 dB。实际 raw10 图像中静态道路已明显恢复，动态图仍含道路光斑和背景纹理，静态图也仍有模糊与残影。','同一30k初态的真实2000步三臂结果显示：候选已观测背景static PSNR为27.33124 dB，双原版约11.96 dB；全图PSNR约下降0.53 dB，目标车辆49帧等权PSNR约下降4 dB（已有pooled误差与每帧面积在上方并列，不代表整车视觉崩坏）。全部49帧static误差改善，全部49帧目标与全图误差上升。短300步结果单独保留，未与32k混作配对。真实静态道路明显恢复，动态背景与边缘移除残留仍存在。')
    page = page.replace('<h2>下一轮受控实验</h2>', '<h2>已做诊断与待核查项</h2>')
    return page, True

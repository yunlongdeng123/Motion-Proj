"""Render S only after actual endpoint evidence and explicit root verification.

Preparing this module does not create results or modify the current report.
"""
import hashlib
import html
import json
import shutil

import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_actual_S_evidence(here, out, page):
    own = here / 'sgcm_data_audit/observed_static_rgb_S_review'
    path = own / 'actual_S_interpretation.json'
    if not path.exists():
        return page, False
    interpretation = json.loads(path.read_bytes())
    if interpretation['status'] != 'ACTUAL_S_NUMERIC_AND_REAL_IMAGE_INTERPRETATION_COMPLETE_ROOT_VERIFIED' or interpretation['root_verified_completed_endpoint'] is not True:
        raise ValueError('Root verification and actual image interpretation required before publishing S')
    root = here / 'continuation_diagnostic/observed_static_rgb30k_to32000'
    effect_path = root / 'receipts/S_vs_originals_locked_metrics.json'
    effect = json.loads(effect_path.read_bytes())
    numeric = json.loads((own / 'independent_S_numeric_review.json').read_bytes())
    if effect['endpoint_iteration'] != 32000 or effect['status'] != 'completed_S_vs_both_originals_locked_endpoint_metric_export' or sha(effect_path) != numeric['canonical_three_arm_metrics_sha256'] or sha(effect_path) != interpretation['canonical_S_metrics_sha256']:
        raise ValueError('S endpoint/canonical/independent evidence binding differs')
    target = out / 'evidence/S_observed_static_rgb30k_to32000'
    target.mkdir(parents=True, exist_ok=True)
    for name in ('actual_S_interpretation.json', 'independent_S_numeric_review.json', 'actual_S_locality_core_cpu.json', 'actual_S_image_contact_sources.json'):
        shutil.copy2(own / name, target / name)
    for relative in ('status.json', 'continuation_diagnostic.json', 'first_original_compose_reference_comparison.json', 'first_backward_reference_comparison.json', 'receipts/full49_summary.json', 'receipts/S_vs_originals_locked_metrics.json', 'receipts/common49_inputs_after_S_cpu.json', 'terminal_32000/diagnostic_snapshot_manifest.json'):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, destination)
    def fmt(x):
        return f'{x:.9f}' if abs(x) >= .001 else f'{x:.6e}'
    summary = json.loads((root / 'receipts/full49_summary.json').read_bytes())
    lookup = {(r['region'], r['array'], r['metric']): r for r in effect['primary_metrics'] + effect['context_metrics_all_six_domains']}
    names = {('alpha_d', 'mean'): '背景动态α均值', ('alpha_d', 'fraction_gt_0_1'): '背景α>0.1占比', ('alpha_d', 'fraction_gt_0_5'): '背景α>0.5占比', ('alpha_d', 'fraction_gt_0_9'): '背景α>0.9占比', ('dynamic', 'rgb_support_gt2_uint8_pixel_fraction'): '背景动态RGB支撑', ('static', 'clamped_rgb_mae_vs_gt'): '背景static MAE', ('static', 'clamped_rgb_mse_vs_gt'): '背景static MSE', ('static', 'clamped_rgb_psnr_db'): '背景static PSNR（dB）', ('composite', 'clamped_rgb_mae_vs_gt'): '全图composite MAE', ('composite', 'clamped_rgb_mse_vs_gt'): '全图composite MSE', ('composite', 'clamped_rgb_psnr_db'): '全图composite PSNR（dB）'}
    primary = ''.join(f'<tr><td>{r["region"]}／{r["array"]}／{r["metric"]}</td><td>{fmt(r["equal_frame_mean_B1"])}</td><td>{fmt(r["equal_frame_mean_B2"])}</td><td>{fmt(r["equal_frame_mean_S"])}</td><td>{fmt(r["S_vs_each_original"]["B1"]["equal_frame_mean_S_minus_reference"])}</td><td>{fmt(r["S_vs_each_original"]["B2"]["equal_frame_mean_S_minus_reference"])}</td><td>{fmt(r["N_observed_original_repeat_abs_difference_of_means"])}</td></tr>' for r in effect['primary_metrics'])
    adverse = numeric['frame_adverse_error_screen']
    counts = ''.join(f'<tr><td>{label}</td><td>{adverse[key]["higher_error_vs_B1_count"]}/49</td><td>{adverse[key]["higher_error_vs_B2_count"]}/49</td></tr>' for key, label in [('full/composite', '全图composite MSE上升'), ('actor/composite', '目标composite MSE上升'), ('non_sky_outside_actor/static', '已观测背景static MSE上升')])
    actor = numeric['actor_area_and_existing_pooled_context']
    pooled = ''.join(f'<tr><td>{a}</td><td>{r["pixel_count_summed"]:,}</td><td>{r["composite"]["mean_frame_psnr_db"]:.9f}</td><td>{r["composite"]["pooled_mse"]:.12f}</td><td>{r["composite"]["pooled_psnr_db"]:.9f}</td></tr>' for a, r in actor['canonical_summary_actor_by_arm'].items())
    supplemental = json.loads((own / 'actual_S_locality_core_cpu.json').read_bytes())
    core_rows = ''.join(f'<tr><td>{region} · {a}</td><td>{r["pixel_count_summed"]:,}</td><td>{r["metrics"]["composite_clamped_MSE"]["pixel_weighted_mean"]:.12f}</td><td>{r["metrics"]["composite_pooled_PSNR"]:.9f}</td><td>{r["metrics"]["alpha_d_mean"]["pixel_weighted_mean"]:.9f}</td></tr>' for region in ('actor_core', 'actor_inner_boundary', 'outside_boundary') for a, r in supplemental['core_regions'][region].items() if a in ('B1', 'B2', 'S'))
    observed = supplemental['locality_regions']['non_sky_outside_actor']
    locality = ''.join(f'<tr><td>{a}</td><td>{r["metrics"]["mean_absolute_composite_static_difference"]["equal_frame_mean"]:.9f}</td><td>{100*r["metrics"]["edited_support_gt2_uint8_pixel_fraction"]["equal_frame_mean"]:.6f}%</td><td>{100*r["metrics"]["edited_support_gt2_uint8_pixel_fraction"]["pixel_weighted_mean"]:.6f}%</td></tr>' for a, r in observed.items() if a in ('B1', 'B2', 'S'))
    figures = []
    for pin in interpretation['actual_image_pins_for_report']:
        source = here / pin['relative_path']
        if sha(source) != pin['sha256']:
            raise ValueError('Actual S report image differs')
        name = 'S32000_' + pin['label'] + '.png'
        shutil.copy2(source, out / 'assets' / name)
        figures.append(f'<figure><img src="assets/{name}" alt="{html.escape(pin["caption"])}"><figcaption>{html.escape(pin["caption"])}</figcaption></figure>')
    # Interpretation text is a manually reviewed actual-data artifact, never a placeholder.
    findings = ''.join(f'<p>{html.escape(text)}</p>' for text in interpretation['actual_findings_for_report'])
    distributions = []
    for key, label in [(('full', 'composite', 'clamped_rgb_psnr_db'), '全图composite PSNR'), (('actor', 'composite', 'clamped_rgb_mse_vs_gt'), 'actor composite MSE'), (('actor', 'composite', 'clamped_rgb_psnr_db'), 'actor composite PSNR'), (('non_sky_outside_actor', 'static', 'clamped_rgb_psnr_db'), '背景static PSNR')]:
        row = lookup[key]
        original = row['original_repeat_paired_absolute_difference_distribution']
        for arm in ('B1', 'B2'):
            pair = row['S_vs_each_original'][arm]
            dist = pair['paired_absolute_difference_distribution']
            distributions.append(f'<tr><td>{label} · S−{arm}</td><td>{fmt(pair["equal_frame_mean_S_minus_reference"])}</td><td>{fmt(dist["mean"])}</td><td>{fmt(dist["finite_values"]["p95"])}</td><td>{fmt(original["mean"])}</td><td>{fmt(original["finite_values"]["p95"])}</td></tr>')
    distribution_html = '<table><tr><th>全部49帧效应分布</th><th>均值之差</th><th>逐帧|Δ|均值</th><th>逐帧|Δ|P95</th><th>原重复逐帧|Δ|均值</th><th>原重复逐帧|Δ|P95</th></tr>' + ''.join(distributions) + '</table>'
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained', facecolor='#101724')
    for ax, (key, label) in zip(axes.flat, [(('full', 'composite', 'clamped_rgb_psnr_db'), 'Composite PSNR, full'), (('non_sky_outside_actor', 'static', 'clamped_rgb_psnr_db'), 'Static PSNR, observed background'), (('non_sky_outside_actor', 'alpha_d', 'mean'), 'Dynamic alpha, observed background'), (('actor', 'composite', 'clamped_rgb_mse_vs_gt'), 'Actor composite MSE')]):
        row = lookup[key]
        for arm, color in [('B1', '#69c5dc'), ('B2', '#e7c45f'), ('S', '#bd91ff')]:
            frames = row['S_vs_each_original']['B1' if arm == 'S' else arm]['frames']
            ax.plot(range(49), [r['S_value'] if arm == 'S' else r['reference_value'] for r in frames], label=arm, color=color, linewidth=1.4)
        ax.set(title=label, xlabel='Raw frame, all49 actual32000')
        ax.set_facecolor('#101724'); ax.tick_params(colors='#e9f0fa'); ax.xaxis.label.set_color('#e9f0fa'); ax.title.set_color('#e9f0fa')
        for spine in ax.spines.values():
            spine.set_color('#48556d')
        ax.grid(alpha=.2); ax.legend(facecolor='#182235', labelcolor='#e9f0fa', fontsize=8)
    fig.savefig(out / 'assets/S32000_all49_paired_metrics.png', dpi=160); plt.close(fig)
    video = json.loads((out / 'long_S_video_validation.json').read_bytes())
    if video['row_order'] != ['B1', 'B2', 'S'] or video['frame_count'] != 49 or not video['all_frames_decoded'] or video['actual_arm_summary_sha256']['S'] != sha(root / 'receipts/full49_summary.json') or sha(out / 'assets' / video['file']) != video['sha256']:
        raise ValueError('Actual synchronous S49 source/decoded video binding differs')
    video_html = f'<h3>真实49帧同步视频</h3><video controls muted playsinline preload="metadata" poster="assets/B1_B2_S32000_synchronous_decoded_raw10.png" style="width:100%;background:#000;border-radius:8px" src="assets/{video["file"]}"></video><p class="muted">从上到下B1、B2、S；每排GT、composite、static、dynamic。49帧10fps／4.9秒，全部588原PNG来源与49帧CPU解码已校验。只新增S对照视频，旧B1／B1-B2-V／short视频未重编码。指标来自真实浮点结果；浏览器播放未验证。<a href="long_S_video_validation.json">新视频来源与解码receipt</a></p>'
    V = json.loads((here / 'continuation_diagnostic/opacity_outside30k_to32000/receipts/three_arm_locked_metrics.json').read_bytes())
    if sha(here / 'continuation_diagnostic/opacity_outside30k_to32000/receipts/three_arm_locked_metrics.json') != '922a3dea16a427539ec944765680b315e9b69b8e7a217af4c5d990b7f7679a9d':
        raise ValueError('Frozen V comparison differs')
    vlookup = {(r['region'], r['array'], r['metric']): r for r in V['primary_metrics'] + V['context_metrics_all_six_domains']}
    sv_specs = [(('full', 'composite', 'clamped_rgb_psnr_db'), '全图composite PSNR'), (('non_sky_outside_actor', 'static', 'clamped_rgb_psnr_db'), '背景static PSNR'), (('actor', 'composite', 'clamped_rgb_psnr_db'), 'actor composite PSNR'), (('non_sky_outside_actor', 'alpha_d', 'mean'), '背景dynamic α'), (('non_sky_outside_actor', 'dynamic', 'rgb_support_gt2_uint8_pixel_fraction'), '背景dynamic RGB支撑')]
    sv_table = '<h3>S／V仅作描述：附加目标不同</h3><table><tr><th>49帧等权</th><th>S静态RGB</th><th>V车外透明度</th></tr>' + ''.join(f'<tr><td>{label}</td><td>{fmt(lookup[key]["equal_frame_mean_S"])}</td><td>{fmt(vlookup[key]["equal_frame_mean_V"])}</td></tr>' for key, label in sv_specs) + '</table>'
    architecture_html = '<div style="display:flex;align-items:center;gap:10px;overflow-x:auto;padding:12px 0"><div class="metric">相同30k状态<br>相同2000相机／LR</div><span>→</span><div class="metric">原paper objective<br>原loss保持</div><span>＋</span><div class="metric">static／GT × (1−Md)<br>coef1静态RGB诊断</div><span>→</span><div class="metric">真实backward／Adam<br>32k完整终态</div><span>→</span><div class="metric">固定49帧<br>原11指标＋真实编辑</div></div>'
    section = f'''<section class="section" id="long-s-32000"><h2>S真实32k终态：额外静态RGB诊断</h2><p>从同一原始30k完整状态恢复，S只在原paper objective外加coef1的全HW静态RGB项：两图乘1−Md，0.8L1＋0.2DSSIM。它包括sky和共享环境，未加入V透明度项、改原mask／depth／time／SGCM或重启正式50k。真实2000步、S执行前原10标量、backward前FP32总分解、采样／time_shift／LR、RNG安全评估、49GT／域配对及四件状态已由实际receipt和root核验，CPU审阅不冒充训练结果。</p>{findings}<table><tr><th>全部49帧误差代价，不删除不利帧</th><th>S对B1</th><th>S对B2</th></tr>{counts}</table><div style="overflow-x:auto"><table><tr><th>原锁定11主指标</th><th>B1帧均值</th><th>B2帧均值</th><th>S帧均值</th><th>S−B1均值差</th><th>S−B2均值差</th><th>原重复|均值差|</th></tr>{primary}</table></div><p class="muted">完整49帧signed／abs效应和原重复逐帧分布保留在canonical JSON。均值差不能抵消掩盖某些帧退化；一次原版重复不是显著性检验或通用噪声界。31k固定5中途观察不与32k49帧混作同一曲线。S对V只作不同附加目标的描述，不当作单变量切换。</p><h3>目标面积、pooled误差与既定core／边界</h3><table><tr><th>actor</th><th>总像素</th><th>49帧等权PSNR</th><th>像素汇总MSE</th><th>pooled PSNR</th></tr>{pooled}</table><p class="muted">每帧原actor面积在独立JSON中保留；出画边缘小区域的均权PSNR不是整车视觉分数。core／inner boundary沿既定5像素SSIM11半径，只是描述域，不改变原11指标、训练mask或验收阈值。</p><table><tr><th>补充域／模型</th><th>像素汇总</th><th>composite pooled MSE</th><th>pooled PSNR</th><th>动态α像素加权</th></tr>{core_rows}</table><h3>真实移除局部性与动态图残留</h3><table><tr><th>已观测非天空车体外</th><th>comp↔static绝对差等权均值</th><th>编辑支撑 · 帧等权</th><th>编辑支撑 · 像素加权</th></tr>{locality}</table><p class="muted">沿已有clip／8位支撑&gt;2定义。body外可含cast shadow，移除差异不是dense edited-GT准确率。官方eval dynamic不含环境；补充中心光线env填充不是jittered training RGB/loss，RGB／alpha仅描述条件辐射，不能证明单个高斯非法颜色。</p><div class="grid">{''.join(figures)}</div><p class="warn">本页不把单场景、从退化30k的恢复实验升级为正常效果或论文复现通过。实际目标／整体代价、动态残留和编辑局部性须同时保留。</p><div class="links"><a href="evidence/S_observed_static_rgb30k_to32000/receipts/S_vs_originals_locked_metrics.json">唯一canonical S对双原版：11主／126上下文／全部49效应</a><a href="evidence/S_observed_static_rgb30k_to32000/independent_S_numeric_review.json">独立CPU复算／不利帧／actor原面积与pooled</a><a href="evidence/S_observed_static_rgb30k_to32000/actual_S_locality_core_cpu.json">沿既有定义的core／边界与局部性补充</a><a href="evidence/S_observed_static_rgb30k_to32000/actual_S_interpretation.json">root核验与真实图像解释</a></div></section>'''
    section = section.replace(findings, architecture_html + findings, 1)
    for row in effect['primary_metrics']:
        raw_label = f'{row["region"]}／{row["array"]}／{row["metric"]}'
        section = section.replace(raw_label, html.escape(names[(row['array'], row['metric'])]))
    section = section.replace('<h3>目标面积、pooled误差与既定core／边界</h3>', distribution_html + '<img src="assets/S32000_all49_paired_metrics.png" alt="S对双原版全部49帧；只有真实32k终点"><h3>目标车辆面积、pooled误差与既定core／边界</h3>')
    area = interpretation['actor_area_context']
    section = section.replace('每帧原actor面积在独立JSON中保留；', f'原固定actor域合计{area["total_pixels"]:,}像素，单帧{area["minimum_per_frame"]}–{area["maximum_per_frame"]}，raw45为{area["raw45_pixels"]}像素；每帧面积在独立JSON中保留。')
    core_counts = []
    for region in ('actor_core', 'actor_inner_boundary', 'outside_boundary'):
        pair = supplemental['S_core_supplement_vs_each_reference'][region]['composite_clamped_MSE']
        core_counts.append(f'{region} MSE升高：对B1 {pair["B1"]["higher_count"]}/49、对B2 {pair["B2"]["higher_count"]}/49')
    section = section.replace('<h3>真实移除局部性与动态图残留</h3>', '<p class="muted">' + '；'.join(core_counts) + '。保留全部49帧正负效应，不删除改善或退化帧。</p><h3>真实移除局部性与动态图残留</h3>')
    s_obs = summary['final49_regions']['non_sky_outside_actor']
    section = section.replace('<div class="grid">' + ''.join(figures), f'<p>动态背景RGB支撑的49帧等权值为{100*lookup[("non_sky_outside_actor", "dynamic", "rgb_support_gt2_uint8_pixel_fraction")]["equal_frame_mean_S"]:.6f}%，像素加权为{100*s_obs["dynamic"]["rgb_support_gt2_uint8_pixel_fraction"]:.6f}%；动态α分别为{lookup[("non_sky_outside_actor", "alpha_d", "mean")]["equal_frame_mean_S"]:.9f}／{s_obs["alpha_d"]["mean"]:.9f}。两项与comp↔static编辑支撑分开呈现，低编辑差异不证明动态对象已隔离。</p>' + sv_table + video_html + '<div class="grid">' + ''.join(figures))
    section = section.replace('<h3>真实49帧同步视频</h3>', '<h3>真实49帧同步视频</h3><p class="muted">root已目视实际解码raw10，确认静态道路恢复与动态道路残留；另目视原生raw48 static，确认左缘车辆形状彩色残片。浏览器播放仍未验证。<a href="S_report_validation.json">本轮root真实图像QA与静态引用／媒体验证</a></p>')
    marker = '<section class="section"><h2>已做诊断与待核查项</h2>'
    if marker not in page:
        raise ValueError('Actual report insertion marker absent')
    page = page.replace(marker, section + marker)
    page = page.replace('长实验状态：B1／B2／V均完成，效果仍未验收', '长实验状态：B1／B2／V／S均完成，效果仍未验收')
    page = page.replace('长V真实2000步、49帧和四件终态已完成，独立CPU核验与真实图像解释见下方；仍未达到正常效果验收。', '长V和额外静态RGB诊断S均已真实完成2000步、49帧及四件终态，独立数值与真实图像解释见下方；静态改善与重建／目标代价同时保留，仍未达到正常效果验收。')
    current_marker = '<section class="section"><h2>当前结论：找到改善分层的方向，尚未修复完成</h2>'
    start = page.index(current_marker)
    end = page.index('</section>', start) + len('</section>')
    current = current_marker + '<p>从同一退化30k初态、同一2000步样本与学习率出发，原版B1／B2全图composite恢复到约36.43 dB，但已观测背景static仍约11.96 dB。两项独立诊断均改善static：V为27.33124 dB，S为31.76342 dB，真实静态道路也恢复；S只增加原Eq10外的coef1静态RGB项，V只增加车外透明度项，没有将二者混合。</p><p>两个诊断都有代价。V全图／目标MSE在全部49帧升高；S全图MSE也在全部49帧升高，目标MSE对B1有46帧、对B2有47帧升高，已有pooled与core结果支持代价不只来自小面积边界。S动态背景α仍约0.51635，RGB支撑约79.01%，raw48静态移除还留有车辆形状彩色残片。它支持监督与分配是可干预方向，不证明正常分离、编辑准确或论文公式已认证。</p><p class="warn">整体排错没有完成。一次原版重复只是本次观察的波动，不是置信区间或通用噪声界；短30300、长32000和固定pose时间scan分别保留。初始化原生标签见证与预处理缺口是独立证据，尚未证明confetti因果。未验证从零训练、后续reset和移动编辑，也没有后续训练授权。</p></section>'
    page = page[:start] + current + page[end:]
    return page, True

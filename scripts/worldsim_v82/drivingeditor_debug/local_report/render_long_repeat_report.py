"""Publish real completed B2 and observed repeat evidence; no training or GPU."""
import hashlib
import html
import json
import shutil

from PIL import Image


NOISE_SHA = 'e4ae32c07b63ed7be0171a727f8a12ec153f66e63d6ac89f4594fa4985145f08'
SUMMARY_SHA = '2ded7dda0952e0bf8605154a642576049ce45b41e8c1df1ddda3bec62085c92e'
COMMON_SHA = 'dfd33893200805b4aebf147121b2be72270163a2ec61674e7704d2053de7f381'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_evidence(root, target, relative):
    destination = target / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / relative, destination)
    return destination


def add_depth_definition(here, out, page):
    """Clarify the cross-page definition without changing frozen audit files."""
    text_path = here.parents[1] / 'work/paper_audit/DrivingEditor.txt'
    text = text_path.read_text(encoding='utf8')
    required = 'where DLiDAR and Dimage respectively represent the inverse of'
    if required not in text or 'the sparse LiDAR-depth map and the rendered depth map.' not in text:
        raise ValueError('Actual paper inverse-depth definition not found')
    source = here / 'loss_audit/long_pair_30k_32000/variant_frozen_repository/train.py'
    train = source.read_text(encoding='utf8')
    if 'sky_depth = 900' not in train or 'depth_blend_mode == 0:  # harmonic mean' not in train:
        raise ValueError('Actual inherited harmonic/900 depth code differs')
    clarification = {
        'status': 'CPU_actual_paper_cross_page_definition_clarified',
        'paper_text_sha256': sha(text_path), 'equation': 16,
        'printed_pages': [1702, 1703],
        'inverse_depth_explicitly_defined_in_paper': True,
        'definition': 'DLiDAR and Dimage are respectively the inverse of sparse LiDAR-depth and rendered depth; Eq16 is continued across the page boundary.',
        'current_train_sha256': sha(source),
        'inherited_current_choices': {'depth_blend_mode': 0, 'harmonic_mean': True, 'sky_depth': 900},
        'classification': 'Inverse-depth loss is paper-specified. Harmonic environment-depth blending and900 sky depth are inherited implementation choices not specified in the supplied paper, not a proven implementation bug.',
        'frozen_loss_source_configuration_or_audit_references_modified': False,
        'GPU_used': False,
    }
    path = out / 'evidence/depth_definition_clarification_cpu.json'
    path.write_text(json.dumps(clarification, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    paragraph = '''<p>深度定义补充：Eq.16 在印刷页1702给出公式，下一页1703明确将 D<sub>LiDAR</sub> 和 D<sub>image</sub> 定义为稀疏LiDAR深度和渲染深度的倒数。因此 inverse-depth loss 本身是论文明确要求。当前实现继承的环境深度 harmonic blend 与 sky_depth=900 未由论文公开说明，应列为待核对的复现选择，不能据此将逆深度目标误判为确定bug。<a href="evidence/depth_definition_clarification_cpu.json">跨页定义与实际代码证据</a></p>'''
    return page.replace('<a href="evidence/official_git_refs.json">公开分支、标签及提交时间核对</a>', paragraph + '<a href="evidence/official_git_refs.json">公开分支、标签及提交时间核对</a>')


def candidate_state(here, out):
    root = here / 'continuation_diagnostic/opacity_outside30k_to32000'
    status_path = root / 'status.json'
    if not status_path.exists():
        return '长V的实际新binding和CPU独立审阅已就绪；本页没有V训练或终点结果。须由root最后批准并交唯一GPU任务执行。', []
    status = json.loads(status_path.read_text(encoding='utf8'))
    target = out / 'evidence/V_opacity_outside30k_to32000'
    copy_evidence(root, target, 'status.json')
    links = ['<a href="evidence/V_opacity_outside30k_to32000/status.json">V实际运行快照</a>']
    if status.get('status') == 'complete' and status.get('actual_exit_code') == 0:
        message = '长V运行已退出0；本页尚未纳入其完整终态与效果配对，不能据此提前给效果结论。'
    elif status.get('status') == 'failed':
        message = '长V实际运行失败，保留失败状态；本页没有V效果结论。'
    else:
        step = status.get('iteration')
        message = '长V已实际启动，终点结果尚未产生。'
        if status.get('started_utc'):
            message += f'worker启动记录为{html.escape(status["started_utc"])}。'
        if isinstance(step, int):
            message += f'此份记录到迭代{step:,}。'
        restore = status.get('restore', {})
        if restore.get('status') == 'actual_restore_pass' and all(restore.get(k) is True for k in ('all_rng_restored_exactly', 'camera_sampler_scale_ema_restored_exactly', 'model_runtime_sgcm_restored_exactly', 'conditional_adam_restored_exactly', 'saved_optimizer_tensor_placement_preserved', 'optimizer_bound_to_saved_parameters')):
            message += '原30k参数／Adam／全部RNG／sampler／SGCM完整恢复通过。'
        first_path = root / 'first_backward_reference_comparison.json'
        if first_path.exists():
            first = json.loads(first_path.read_text(encoding='utf8'))
            copy_evidence(root, target, first_path.name)
            links.append('<a href="evidence/V_opacity_outside30k_to32000/first_backward_reference_comparison.json">V首步backward前核验</a>')
            exact = len(first.get('comparisons', {})) == 10 and all(r.get('exact_equal') is True for r in first.get('comparisons', {}).values()) and first.get('tolerance_applied') is False
            if exact and first.get('original_backward_returned') is True:
                message += '首30001的十个原目标标量在真实backward前逐项精确相等；此核验不代表后续2000步损失相等或效果通过。'
                message += f'首步原目标loss={first["original_objective_loss"]:.9f}，额外项={first["background_opacity_unweighted"]:.9f}×{first["added_weight"]:.2f}，新总loss={first["actual_new_total_loss"]:.9f}；原目标等价与新增项分别记录。'
            else:
                message += '实际恢复与首步完整核验仍待证据。'
        else:
            message += '实际恢复与首步完整核验仍待证据。'
    return message, links


def add_long_repeat_evidence(here, out, page):
    root = here / 'continuation_diagnostic/original30k_to32000_noise_reference'
    paths = {name: root / 'receipts' / name for name in ('full49_summary.json', 'B1_B2_observed_noise.json', 'common49_inputs_after_B2_cpu.json')}
    if not all(path.exists() for path in paths.values()):
        return page, False
    if [sha(paths[k]) for k in paths] != [SUMMARY_SHA, NOISE_SHA, COMMON_SHA]:
        raise ValueError('Actual completed B2/noise/common49 canonical receipt SHA differs')
    summary = json.loads(paths['full49_summary.json'].read_text(encoding='utf8'))
    noise = json.loads(paths['B1_B2_observed_noise.json'].read_text(encoding='utf8'))
    status = json.loads((root / 'status.json').read_text(encoding='utf8'))
    if status['status'] != 'complete' or status['actual_exit_code'] != 0 or summary['step_count'] != 2000 or len(noise['primary_metrics']) != 11 or noise['endpoint_iteration'] != 32000:
        raise ValueError('Actual full-window B2 evidence incomplete')
    target = out / 'evidence/B2_original30k_to32000_noise_reference'
    copied = []
    for relative in ('status.json', 'continuation_diagnostic.json', 'first_step_reference_comparison.json', 'runtime_import_pins.json', 'receipts/full49_summary.json', 'receipts/B1_B2_observed_noise.json', 'receipts/common49_inputs_after_B2_cpu.json', 'receipts/gpu_release_handoff.json', 'terminal_32000/diagnostic_snapshot_manifest.json', 'extra_eval/32000/diagnostic_eval.json'):
        copied.append(copy_evidence(root, target, relative))
    image_receipts = []
    labels = [('gt', '原始GT'), ('composite', '整体重建'), ('static', '静态／移除通道'), ('dynamic', '动态通道')]
    for channel, label in labels:
        source = root / 'extra_eval/32000' / channel / '0017010.png'
        pin = next(p for p in summary['artifact_pins'] if p['path'].endswith('/' + channel + '/0017010.png'))
        if sha(source) != pin['sha256'] or source.stat().st_size != pin['bytes']:
            raise ValueError('B2 raw10 actual PNG pin differs: ' + channel)
        with Image.open(source) as image:
            image.load()
            if image.size != (480, 320):
                raise ValueError('B2 raw10 actual PNG dimensions differ')
        name = f'B2_original32000_{channel}_raw10.png'
        shutil.copy2(source, out / 'assets' / name)
        image_receipts.append({'channel': channel, 'file': name, 'sha256': sha(source), 'decoded_size': [480, 320]})
    def metric(region, array, key):
        return next(r for r in noise['primary_metrics'] if (r['region'], r['array'], r['metric']) == (region, array, key))
    composite = metric('full', 'composite', 'clamped_rgb_psnr_db')
    static = metric('non_sky_outside_actor', 'static', 'clamped_rgb_psnr_db')
    support = metric('non_sky_outside_actor', 'dynamic', 'rgb_support_gt2_uint8_pixel_fraction')
    alpha = metric('non_sky_outside_actor', 'alpha_d', 'mean')
    observed = summary['final49_regions']['non_sky_outside_actor']
    names = {('alpha_d', 'mean'): '背景动态α均值', ('alpha_d', 'fraction_gt_0_1'): '背景α&gt;0.1占比', ('alpha_d', 'fraction_gt_0_5'): '背景α&gt;0.5占比', ('alpha_d', 'fraction_gt_0_9'): '背景α&gt;0.9占比', ('dynamic', 'rgb_support_gt2_uint8_pixel_fraction'): '背景动态非黑支撑占比', ('static', 'clamped_rgb_mae_vs_gt'): '背景static MAE', ('static', 'clamped_rgb_mse_vs_gt'): '背景static MSE', ('static', 'clamped_rgb_psnr_db'): '背景static PSNR（dB）', ('composite', 'clamped_rgb_mae_vs_gt'): '全图composite MAE', ('composite', 'clamped_rgb_mse_vs_gt'): '全图composite MSE', ('composite', 'clamped_rgb_psnr_db'): '全图composite PSNR（dB）'}
    def fmt(value):
        return f'{value:.9f}' if abs(value) >= .001 else f'{value:.6e}'
    rows = []
    for row in noise['primary_metrics']:
        distribution = row['paired_absolute_difference_distribution']
        quantiles = distribution['finite_values']
        rows.append(f'<tr><td>{names[(row["array"], row["metric"])]}</td><td>{fmt(row["equal_frame_mean_B1"])}</td><td>{fmt(row["equal_frame_mean_B2"])}</td><td>{fmt(row["equal_frame_mean_B2_minus_B1"])}<br><small>|Δ均值|={fmt(row["N_abs_difference_of_equal_frame_means"])}</small></td><td>{fmt(distribution["mean"])}</td><td>{fmt(quantiles["p50"])}<br>{fmt(quantiles["p95"])}</td><td>{fmt(quantiles["max"])}</td></tr>')
    images = ''.join('<h3>' + arm + ' · 32k · 相同真实raw10</h3><div class="grid">' + ''.join(f'<figure><img src="assets/{arm}_original32000_{channel}_raw10.png" alt="{arm}真实32000 raw10 {label}"><figcaption>{label}</figcaption></figure>' for channel, label in labels) + '</div>' for arm in ('B1', 'B2'))
    section = f'''<section class="section" id="long-b2-noise"><h2>B2重复已完成：整体相近，分层退化重复出现</h2><p>B2从同一原始30k四件状态独立恢复，真实完成30,001–32,000全部2,000步，实际worker／launcher均退出0，保存末49帧及四件完整终态并释放GPU。初始模型／Adam／RNG／sampler恢复通过；全2,000个camera／time_shift／LR与B1精确一致，四次额外评估均还原RNG。末49帧GT浮点数组、尺寸与六域像素数配对通过。首十标量相等，后续CUDA损失轨迹仍有变化。</p><table><tr><th>32k终点 · 同窗原版重复</th><th>B1 · 49帧等权均值</th><th>B2 · 49帧等权均值</th><th>B2 · 按区域像素汇总</th></tr><tr><td>全图composite PSNR</td><td>{composite['equal_frame_mean_B1']:.9f} dB</td><td>{composite['equal_frame_mean_B2']:.9f} dB</td><td>{summary['final49_regions']['full']['composite']['pooled_psnr_db']:.9f} dB（先汇总MSE）</td></tr><tr><td>已观测非天空车体外static PSNR</td><td>{static['equal_frame_mean_B1']:.9f} dB</td><td>{static['equal_frame_mean_B2']:.9f} dB</td><td>{observed['static']['pooled_psnr_db']:.9f} dB（先汇总MSE）</td></tr><tr><td>同一区域动态α</td><td>{alpha['equal_frame_mean_B1']:.9f}</td><td>{alpha['equal_frame_mean_B2']:.9f}</td><td>{observed['alpha_d']['mean']:.9f}</td></tr><tr><td>同一区域动态非黑RGB支撑</td><td>{100 * support['equal_frame_mean_B1']:.6f}%</td><td>{100 * support['equal_frame_mean_B2']:.6f}%</td><td>{100 * observed['dynamic']['rgb_support_gt2_uint8_pixel_fraction']:.6f}%</td></tr></table><p>真实raw10再次出现相同问题：composite接近GT，static道路仍大面积彩噪，dynamic仍包含道路和背景。两份原版均没有达到正常分层，原样续训恢复整体重建不足以证明编辑可用。</p>{images}<h3>一次原版重复的差异：均值之差与逐帧差分开看</h3><p>令Δ<sub>i</sub>=q<sub>B2,i</sub>−q<sub>B1,i</sub>。“Δ均值”是两组49帧等权均值之差；N=|Δ均值|可能因正负抵消而很小。“逐帧|Δ|”先对每帧取绝对值，再汇总均值和分位数。例：全图PSNR的N={composite['N_abs_difference_of_equal_frame_means']:.9f} dB，但逐帧|Δ|均值为{composite['paired_absolute_difference_distribution']['mean']:.9f} dB，P95为{composite['paired_absolute_difference_distribution']['finite_values']['p95']:.9f} dB。</p><div style="overflow-x:auto"><table><tr><th>锁定的11项主指标</th><th>B1帧均值</th><th>B2帧均值</th><th>Δ均值／N</th><th>逐帧|Δ|均值</th><th>逐帧|Δ|<br>P50／P95</th><th>逐帧|Δ|最大值</th></tr>{''.join(rows)}</table></div><p class="muted">“背景”均指固定的已观测非天空、车体mask外区域；占比在上表保持0–1单位。每项使用全部49个非空区域，没有从结果挑选帧。完整receipt还保留126项六域上下文指标、各帧signed／abs差、方向计数及真实PNG像素差异。B2动态支撑的49帧等权值为{100 * support['equal_frame_mean_B2']:.6f}%，像素汇总值为{100 * observed['dynamic']['rgb_support_gt2_uint8_pixel_fraction']:.6f}%，两者不能混用。</p><p class="warn">这是一次观察到的原版重复差异，不是置信区间、统计显著性或所有CUDA运行的通用噪声界。长V尚无终点效果结论；今后须同时相对B1和B2报告逐帧效应、composite代价、静态背景、dynamic alpha/support、目标车辆保真与真实移除局部性。短候选30300与原版32000不可当同窗配对；没有完整论文复现结论。</p><div class="links"><a href="evidence/B2_original30k_to32000_noise_reference/receipts/B1_B2_observed_noise.json">真实B1–B2：11主指标／126上下文／全部49逐帧差异</a><a href="evidence/B2_original30k_to32000_noise_reference/receipts/full49_summary.json">B2真实完整49帧及终态receipt</a><a href="evidence/B2_original30k_to32000_noise_reference/receipts/common49_inputs_after_B2_cpu.json">原467输入／245共有输入／98原mask核验与限制</a><a href="evidence/B2_original30k_to32000_noise_reference/terminal_32000/diagnostic_snapshot_manifest.json">B2四件32000终态</a></div></section>'''
    page = page.replace('<section class="section"><h2>下一轮受控实验</h2>', section + '<section class="section"><h2>下一轮受控实验</h2>')
    page = page.replace('须等B2完整同窗重复、真实采样与49帧域配对，再对同一λ=0.05的长V单独审批和验证。', 'B2完整同窗重复、真实采样与49帧域配对已通过，详见下方。长V须单独批准并完成实际运行后才有同窗效果结论。')
    page = page.replace('继续同初态原版重复B2，之后才单独批准同一λ=0.05的长V；均停在32k，避开下一次33k reset。', '原版重复B2也已完成，重复出现整体恢复而静态道路仍坏；其观察到的逐帧差异已落盘。长V须单独批准并验证，停在32k，避开下一次33k reset。')
    task = json.loads((here / 'task_manifest.json').read_text(encoding='utf8'))
    candidate, candidate_links = candidate_state(here, out)
    if '长V已实际启动' in candidate:
        page = page.replace('长V须单独批准并验证，停在32k，避开下一次33k reset。', '长V已获单独批准并运行中，目标停在32k，避开下一次33k reset。')
        page = page.replace('长V须单独批准并完成实际运行后才有同窗效果结论。', '长V已获单独批准并启动，完整运行及终点配对后才有同窗效果结论。')
    progress = f'''<section class="section" id="long-experiment-status"><h2>长实验状态：B1／B2完成，V待终点验证</h2><p>B2实际结束于{html.escape(status['finished_utc'])}，完整49帧与CPU原版重复差异已核验。root维护的任务快照为{html.escape(task['updated_utc'])}；若时间较早，以本页绑定的实际exit0和完成receipt确认B2已完成。{candidate}</p><p>V的完整恢复、首步backward前原目标等价、全2000步采样／LR配对、RNG安全49评估和四件终态仍是运行约束。其他场景保持暂停。</p><div class="links"><a href="evidence/debug_task_status.json">root任务快照</a><a href="evidence/B2_original30k_to32000_noise_reference/receipts/gpu_release_handoff.json">B2完成与GPU释放记录</a>{''.join(candidate_links)}</div></section>'''
    page = page.replace('<section class="section"><h2>四种假说，分别找证据</h2>', progress + '<section class="section"><h2>四种假说，分别找证据</h2>')
    receipt = {'status': 'B2_ACTUAL_FULL49_AND_ORIGINAL_REPEAT_REPORT_INPUTS_VERIFIED', 'source_receipt_sha256': {'full49_summary': SUMMARY_SHA, 'B1_B2_observed_noise': NOISE_SHA, 'common49_inputs_after_B2_cpu': COMMON_SHA}, 'copied_actual_raw10_PNGs': image_receipts, 'copied_evidence_sha256': {str(p.relative_to(out)): sha(p) for p in copied}, 'primary_metric_count': 11, 'context_metric_count': 126, 'frame_count': 49, 'already_encoded_short_and_B1_MP4_reencoded': False, 'noise_is_confidence_interval_or_general_bound': False, 'browser_UI_or_playback_checked': False, 'GPU_used': False, 'frozen_source_config_plan_reference_or_task_manifest_modified': False}
    (out / 'long_B2_report_inputs_validation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    page = add_depth_definition(here, out, page)
    from render_actual_three_arm_report import add_actual_three_arm_evidence
    page,_=add_actual_three_arm_evidence(here,out,page)
    return page, True

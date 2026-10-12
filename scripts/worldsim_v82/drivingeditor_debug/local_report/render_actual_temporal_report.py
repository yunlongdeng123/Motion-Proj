"""Small completed fixed-pose temporal appendix, using canonical contacts only."""
import hashlib
import json
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_actual_temporal_evidence(here, out, page):
    own = here / 'sgcm_data_audit/fixed_pose_time_review/actual_temporal_interpretation_cpu.json'
    if not own.exists():
        return page
    review = json.loads(own.read_bytes())
    root = here / 'component_diagnostic/fixed_pose_temporal32k'
    results = root / 'results'
    if review['manifest_sha256'] != sha(results / 'diagnostic_manifest.json') or review['summary_sha256'] != sha(results / 'temporal_summary.json'):
        raise ValueError('Completed temporal interpretation is not bound to actual immutable receipts')
    destination = out / 'evidence/fixed_pose_temporal32k'
    destination.mkdir(parents=True, exist_ok=True)
    for source, name in [(own, own.name), (root / 'plan.json', 'plan.json'), (root / 'cpu_preflight.json', 'cpu_preflight.json'), (results / 'diagnostic_manifest.json', 'diagnostic_manifest.json'), (results / 'temporal_summary.json', 'temporal_summary.json'), (results / 'contact_sheets/receipt.json', 'contact_source_receipt.json')]:
        shutil.copy2(source, destination / name)
    for contact in review['contacts']:
        source = results / 'contact_sheets' / contact['file']
        if sha(source) != contact['sha256']:
            raise ValueError('Actual temporal contact changed: ' + contact['file'])
        shutil.copy2(source, out / 'assets' / ('temporal32k_' + contact['file']))
    rows = ''.join(f'<tr><td>{r["arm"]}</td><td>{r["time_source_raw_index"]}／{r["timestamp_model"]:.2f}</td><td>{ {"actual":"原模型", "freeze_SHM_position":"仅冻结SHM位置", "freeze_temporal_opacity_and_support":"仅冻结时间opacity／点支持", "freeze_both":"两者冻结"}[r["mode"]] }</td><td>{r["static_RGB_MAE_vs_same_arm_anchor"]:.9f}</td><td>{r["static_RGB_MAE_vs_same_time_actual"]:.9f}</td></tr>' for r in review['endpoint_static_RGB_deltas'])
    maxima = review['all49_max_raw_FP32_regional_MAE_vs_anchor']
    maxrows = ''.join(f'<tr><td>{label}</td><td>{maxima["B1"][key]:.9f}</td><td>{maxima["V"][key]:.9f}</td></tr>' for key, label in [('static_full', 'static完整RGB'), ('static_Gaussian_RGB', '高斯颜色贡献（减环境项得到）'), ('environment_transmission', '环境透射贡献'), ('static_alpha', 'static α')])
    contacts = ''.join(f'<img src="assets/temporal32k_{r["file"]}" alt="{r["file"]} 实际固定raw10位姿时间诊断">' for r in review['contacts'])
    progress = ''
    approval_path = here / 'loss_audit/next_diagnostics_cpu/observed_static_rgb_S/materialized_plan/static_anchor_root_coordination_receipt.json'
    if approval_path.exists():
        approval = json.loads(approval_path.read_bytes())
        if approval['decision'] == 'APPROVED_SINGLE_VARIABLE_OBSERVED_STATIC_RGB_30K_TO32000':
            progress = '<p class="muted">后续S：原论文目标额外加入coef1静态RGB（两图乘1−Md的全图0.8L1＋0.2DSSIM），已获独立批准；实际执行进度见root任务快照。本页没有S终点结果，S也不是论文认证公式。</p>'
    section = f'''<section class="section" id="fixed-pose-temporal32k"><h2>只读时间诊断：固定相机后，静态分支仍随时间改变</h2><p>B1与V各取实际32k模型，始终固定raw10的位姿、投影、图像尺寸与环境光线；仅扫描49个原观测timestamp。共98次原模型渲染，加12次只读端点位置／opacity控制，实际110/110完成并退出0。原时间的FP32与PNG对既有渲染精确相等，RNG、相机／光线、参数／环境和原输入hash保持。没有训练、Adam恢复或backward。</p><p>已观测非天空车体外固定域有142,486像素。下面是同一位姿、同一模型相对原时间输出的raw FP32差异，未clip，不是GT误差或画质提升；其它timestamp与raw10位姿组合为反事实相机／时间。</p><table><tr><th>49个时间点中最大的区域mean|Δ|</th><th>B1</th><th>V</th></tr>{maxrows}</table><details><summary>两个端点的static RGB差异与独立冻结控制</summary><table><tr><th>模型</th><th>时间来源raw／模型t</th><th>控制</th><th>相对原raw10时间mean|Δ|</th><th>相对同时间原模型mean|Δ|</th></tr>{rows}</table></details><p>在所测端点，冻结时间opacity及原时间点支持集的影响大于冻结SHM位置。raw48时，B1仅冻结opacity后的anchor差为0.017568915，仅冻结位置仍为0.092413198；V对应为0.009565662与0.053627555。两种控制有相互作用，不能把差值当可相加的因果占比。这里的支持是原marginal&gt;0.05点预过滤，不是屏幕可见性。</p><p class="warn">raw0的模型时间−0.48与anchor −0.28相差一个cycle=0.2，SHM位置几乎周期等价；不能据raw0认为其它时间的运动不重要。独立冻结控制只测两端，49扫描仅为原模型输出。两者同时冻结与anchor精确相等验证控制路径，不是修复验收或作者硬冻结意图的证明。现有static时间语义仍是高风险一致性差距。</p><p class="muted">单位已核：实际PCA空间缩放为0.1模型单位／原世界米，点位移计算已除0.1，因此其_m字段是原世界米，不改写原manifest。保存的归一化前景depth数组及统计仍为模型单位；图中的depth preview另除0.1转换为米，固定0–100m灰度。timestamp标签是模型时间。可能有效点按原时间marginal与光栅器alpha阈值分层，不能称已验证实际像素可见。</p><details><summary>四张实际contact：B1／V的观测时间扫描与端点控制</summary><p class="muted">图上rawNN只表示timestamp的来源帧；相机位姿始终为raw10。各列static、推导的高斯颜色、环境透射、alpha和深度预览来自实际输出，没有生成示意画质或改色。</p>{contacts}</details>{progress}<div class="links"><a href="evidence/fixed_pose_temporal32k/actual_temporal_interpretation_cpu.json">独立110记录／49最大差／端点与单位解释</a><a href="evidence/fixed_pose_temporal32k/diagnostic_manifest.json">实际不可改写manifest</a><a href="evidence/fixed_pose_temporal32k/temporal_summary.json">canonical真实摘要与释放记录</a><a href="evidence/fixed_pose_temporal32k/contact_source_receipt.json">四contact的原PNG来源</a></div></section>'''
    page = page.replace('<section class="section"><h2>已做诊断与待核查项</h2>', section + '<section class="section"><h2>已做诊断与待核查项</h2>')
    preprocessing = here / 'preprocessing_gap_audit'
    names = ('mdetector_readonly_audit.json', 'mdetector_source_receipt.json', 'shadow_readonly_audit.json', 'shadow_source_receipt.json')
    if all((preprocessing / name).exists() for name in names):
        pp = out / 'evidence/preprocessing_gap_audit'
        pp.mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copy2(preprocessing / name, pp / name)
        note = '<p class="muted">预处理缺口已有只读可行性记录：<a href="evidence/preprocessing_gap_audit/mdetector_readonly_audit.json">M-detector</a>／<a href="evidence/preprocessing_gap_audit/mdetector_source_receipt.json">对应公开源码pin</a>，<a href="evidence/preprocessing_gap_audit/shadow_readonly_audit.json">shadow</a>／<a href="evidence/preprocessing_gap_audit/shadow_source_receipt.json">对应公开源码pin</a>。这里只核查所需流程，没有实际执行数据／模型下载或补预处理，也未证明这些缺口导致当前退化。</p>'
        page = page.replace('没有无遮挡真值的移除区域，不宣称质量已获真值验证。</p>', '没有无遮挡真值的移除区域，不宣称质量已获真值验证。</p>' + note)
    return page

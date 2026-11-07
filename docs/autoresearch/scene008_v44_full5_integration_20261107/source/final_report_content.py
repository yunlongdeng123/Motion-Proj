"""Final evidence-driven content. The original negative cases remain immutable."""
import csv
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path


def build_final(b):
    root = b.root
    read = b.read
    link = b.link
    h = lambda value: html.escape(str(value), quote=True)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    index = read('evidence/evidence_index.json')
    assert index['full49_case_id'] == 'full49_semantic_actor5_all49fit'
    findings = read('evidence/v44_baseline_contract_findings.json')
    summary = read(index['full49_summary'])
    metrics = read(index['full49_metrics'])
    integrity = read(index['full49_integrity'])
    visual = read(index['full49_visual_review'])
    calibration = read('evidence/semantic_calibration49/receipt.json')
    sidecar = read('evidence/semantic_calibration49/semantic_actor5_manifest.json')
    mask_audit = read('evidence/semantic_mask_input_audit.json')
    ledger = read('evidence/execution_forward_ledger_final.json')
    cli = read('evidence/public_cli_replay105.json')
    delivery = read('evidence/execution_delivery_verification.json')
    publication = read('evidence/viewer_publication_receipt.json')
    runtime = read('evidence/viewer_runtime_qa.json')
    package = read('evidence/review/semantic_identity_package49_local_verification.json')
    pixel_proof = read('evidence/review/cpu_fixed_protection_pixel_exact49.json')
    browser_qa = read('evidence/report_browser_qa.json')
    assert summary['frame_count'] == 49 and len(metrics['numeric_gates']) == 25
    assert len(metrics['failed_gates']) == 4 and sum(g['passed'] for g in metrics['numeric_gates']) == 21
    assert metrics['fixed_original_domain_hashes_exact'] and summary['a5_mask_pixels'] == 121128
    assert summary['a5_upper_pixels'] == 25906 and visual['candidate_report_and_optional_central_viewer_release']
    assert ledger['total_raster_kernel_forwards'] == 303 and ledger['diagnostic_dummy_feature_backwards'] == 49
    assert ledger['optimization_updates'] == 0 and cli['exact_encoded_PNG_bytes']
    assert publication['original_v44_default_preserved'] and runtime['status'] == 'PASS'
    archive = root/'model/semantic_identity_package49.tar.gz'
    assert digest(archive) == package['archive_sha256'] and package['all_file_SHA256_exact']
    b.inputs.add(archive)
    figure_sources = b.figures(index)
    assert all(row['path'] for row in figure_sources)
    all49_html, all49_data = b.all49(index)
    assert all49_data['images_complete']
    pooled = metrics['evaluated_a5_and_upper_summary']
    keep = pooled['keep']
    ratio_full = keep['a5_RGB']['delta']/keep['a5_RGB']['baseline']
    ratio_upper = keep['a5_upper_RGB']['delta']/keep['a5_upper_RGB']['baseline']
    regressed = summary['a5_per_frame_distribution']['keep']['a5_RGB']['regressed_raw_ids']
    assert regressed == list(range(89,100))+[112]
    worst = max(summary['a5_per_frame_distribution']['keep']['a5_RGB']['per_frame'], key=lambda row: row['delta'])
    failed_rows = ''.join('<tr><td>'+h(g['name'])+'</td><td class="worse">'+f'{g["value"]:+.6f}'+'</td><td>&lt; 0（原阈值）</td><td>FAIL</td></tr>' for g in metrics['failed_gates'])
    guard_names = ['keep:a6_RGB:worst_frame','keep:a6_depth_m:worst_frame','keep:a37_RGB:worst_frame',
                   'delete:a6_RGB:worst_frame','delete:a6_depth_m:worst_frame','delete:road_RGB:worst_frame',
                   'delete:fixed_road_absolute_blue:worst_frame','keep:protected_RGB:worst_frame','delete:protected_RGB:worst_frame']
    labels = ['KEEP a6 RGB','KEEP a6深度（m）','KEEP a37 RGB','DELETE a6 RGB','DELETE a6深度（m）',
              'DELETE道路 RGB','DELETE固定道路蓝值','KEEP保护域 RGB','DELETE保护域 RGB']
    guard_rows = ''
    for name,label in zip(guard_names,labels):
        g = next(g for g in metrics['numeric_gates'] if g['name']==name)
        guard_rows += '<tr><td>'+label+'</td><td>'+f'{g["value"]:+.7f}'+'</td><td>'+f'{g["threshold"]:.3f}'+'</td><td>PASS</td></tr>'
    matrix = ''.join('<tr><td>'+h(row['source'])+'</td><td>'+f'{row["rows"]:,}'+'</td><td>'+f'{row["selected_rows"]:,}'+'</td><td>'+f'{row["unobserved_rows"]:,}'+'</td></tr>' for row in calibration['source_matrix'])
    history = [
        ('pilot_candidate','5 / 18 / 26来源一起接入','四帧有黑糊车辆，视觉失败；未扩49','evidence/pilot_all3/root_visual_gate.json'),
        ('full49_actor5','只接来源5','raw105改善，但全49 a5误差+1.68%，13帧回退；停止发布','actor5_additive49_hold.html'),
        ('pilot_actor5_retire','退役旧static中可证明的a5源后代','raw94仍黑、道路/保护域失败；未扩49','evidence/pilot_actor5_retire/paired_metrics.json'),
        ('pilot_semantic_actor5','4个BUILD视图贡献归属','黑暗大幅缓解；raw94四光度门失败','evidence/pilot_semantic_actor5/paired_metrics.json'),
        ('pilot_semantic_actor5_all49fit','49个BUILD视图贡献归属，先四帧','21/25；四项FAIL保留。root明确仅授权完整49诊断','evidence/pilot_semantic_actor5_all49fit/paired_metrics.json'),
    ]
    history_rows = ''.join('<tr><td>'+h(label)+'</td><td>'+h(outcome)+'</td><td>'+link(path,'原始证据')+'</td></tr>' for case,label,outcome,path in history)
    history_images = ''
    for case,label,_,_ in history:
        prefix = 'pilot_all3' if case=='pilot_candidate' else case
        if case=='full49_actor5':
            history_images += '<p>'+link('actor5_additive49_hold.html','冻结的来源5加法全49 HOLD报告')+' · '+link('evidence/actor5_additive49_hold_freeze.json','冻结SHA')+'</p>'
            continue
        images = ''
        for state in ('keep','delete'):
            p = f'figures/{prefix}_{state}_four_frames.png'
            if (root/p).is_file():
                images += '<a href="'+p+'"><img src="'+p+'" alt="'+h(label)+' '+state+'"></a>'
        history_images += '<details><summary>'+h(label)+'：原四帧图板</summary>'+images+'</details>'
    native_rows = ''
    for raw in index['pilot_raws']:
        entries = [(f'evidence/v44_all49_reference/frames/{raw:03d}/observed.png','Observed'),
                   (f'evidence/v44_all49_reference/frames/{raw:03d}/keep.png','v44 KEEP'),
                   (f'evidence/v44_all49_reference/frames/{raw:03d}/delete.png','v44 DELETE'),
                   (f'evidence/full49_semantic_actor5_all49fit/frames/{raw:03d}/keep.png','候选KEEP'),
                   (f'evidence/full49_semantic_actor5_all49fit/frames/{raw:03d}/delete.png','候选DELETE')]
        native_rows += '<tr><td>'+str(raw)+'</td><td>'+' · '.join(link(p,label) for p,label in entries)+'</td></tr>'
    evidence_links = [
        ('evidence/full49_semantic_actor5_all49fit/summary.json','完整49原始summary（保留photometric FAIL）'),
        (index['full49_metrics'],'独立CPU逐帧指标/25项门'),
        (index['full49_integrity'],'每状态参数、前缀mask与环境守恒'),
        ('evidence/review/cpu_fixed_protection_pixel_exact49.json','49双状态固定保护域逐像素float最大差0'),
        (index['full49_visual_review'],'root实际全24结构图检与候选发布门'),
        ('evidence/execution_delivery_verification.json','单一执行/取回总核验'),
        ('evidence/execution_forward_ledger_final.json','全部303F / 49B账本'),
        ('evidence/public_cli_replay105.json','公开CLI真实1F对照'),
        ('evidence/review/execution_scope_verify_final.json','所有本任务worker退出/无未来队列'),
        ('evidence/viewer_publication_receipt.json','中央viewer备份/147图SHA/默认保留'),
        ('evidence/viewer_runtime_qa.json','真实Chrome22项运行QA'),
        ('evidence/report_browser_qa.json','报告真实Chrome八项双状态控制QA'),
        ('evidence/contracts/execution_contract_semantic_actor5_all49.json','完整49诊断预封合同'),
        ('evidence/baseline_replay_verification.json','原v44四帧原链重放'),
        ('evidence/v44_baseline_contract_findings.json','基准/路径与逐图来源'),
        ('evidence/review/sh_precompute_raw130_retry1/receipt.json','原SH与预计算RGB控制'),
        ('evidence/review/sh_precompute_raw130_failure.json','SH首失败3F原收据'),
    ]
    evidence_list = ''.join('<li>'+link(path,label)+'</li>' for path,label in evidence_links)
    style = 'body{margin:0;background:#f3f7f9;color:#193b4a;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1400px;margin:auto;padding:28px}section{background:white;border:1px solid #dce7ec;border-radius:12px;padding:24px;margin:20px 0}h1{font-size:29px}h2{font-size:22px}h3{font-size:18px}img{display:block;max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #dce6eb;text-align:left}th{background:#edf4f7}a{color:#006b83}.worse{color:#ad3636}.notice{background:#fff4de;border-left:4px solid #c99435;padding:14px}summary{cursor:pointer;font-weight:600}footer,figcaption{font-size:13px;color:#607783}.toolbar{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.toolbar input{min-width:240px;flex:1}.three{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}code{overflow-wrap:anywhere;font-size:13px}@media(max-width:800px){.three{grid-template-columns:1fr}}'
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scene008：v44＋a5车顶恢复候选</title><style>{style}</style></head><body><main>
<h1>Scene008：保留v44删车效果的a5车顶恢复候选</h1>
<p class="notice">49帧双状态结果已交付。root实际查看全部24个a5可见帧后，确认车顶、挡风玻璃与车体连续可辨，早段黑暗车体问题消失；细节仍偏软。中央<code>#fullclip</code>新增可选“v44＋a5车顶恢复（候选，细节偏软）”，旧v44保持默认并可切回。严格光度门仍为21/25，raw94四项FAIL与12/24可见帧RGB回退原样保留；不是无损升级，也没有人工质量采纳。</p>
<p>使用已完成的Full5 10K重建，通过原BUILD实例mask核对实际成像贡献，再把固定3D行接入冻结v44。原v44背景、道路去蓝、完整C2 a6、环境与a37删除规则保持。此次接入新增训练更新为0。</p>
<section><h2>组件与数据流</h2><img src="pipeline_semantic5.svg" alt="原BUILD实例mask与Full5联合贡献生成固定3D行，查询按固定三维行sidecar选择，v44原输入依赖保持，共同原生深度排序，输出49KEEP和仅删a37"></section>
<section><h2>车顶与挡风玻璃：raw105同尺度对照</h2><p>左到右：真实Observed、正式v44、候选。固定窗口(0,160)–(175,320)，直接粘贴原像素；数值仍在完整固定mask域计算。</p><h3>仅DELETE a37</h3><img src="figures/pilot_semantic_actor5_all49fit_raw105_a5_delete.png" alt="raw105相同裁剪Observed、v44DELETE、结构修复候选"><h3>完整KEEP</h3><img src="figures/pilot_semantic_actor5_all49fit_raw105_a5_keep.png" alt="raw105相同裁剪Observed、v44KEEP、结构修复候选"><p>KEEP全a5可见域MAE变化{ratio_full:+.2%}，固定上部变化{ratio_upper:+.2%}。这两个加权结果不覆盖早帧回退；只判断实际可见表面。</p></section>
{all49_html}
<section><h2>保住了哪些v44效果</h2><p>下表是同固定评价域的逐帧最坏<strong>指标变化</strong>，不是宣称整图像素相同。DELETE道路与固定道路蓝值逐帧变化为0；两状态a6 RGB/深度、KEEP a37指标变化为0。其余保护域在原阈值内。另有更强的保存float逐像素核对：全部49两状态的原a6可见RGB/测深、KEEP a37可见RGB，以及DELETE固定road_edit∩ground域RGB，candidate对正式v44最大差均为0；只限这些固定域，不延伸为整帧或全车无变化。</p><table><thead><tr><th>原守护项</th><th>最坏帧Δ指标</th><th>原上限</th><th>结果</th></tr></thead><tbody>{guard_rows}</tbody></table><p>相机、时间、旧v44行、颜色、前缀mask与Env hash均守恒；DELETE原物理a37组贡献为0。它们说明受测合同得到保持，不以ID计数或alpha替代图像质量。原蓝ROI在raw113/137分母为0时继续为null；原silver残片延期项没有被冒称解决。</p></section>
<section><h2>明确保留的画质取舍</h2><p>KEEP全a5 RGB误差回退的12帧为raw89–99及112；最坏raw{worst['raw']}为ΔMAE {worst['delta']:+.6f}。近处缺顶结构恢复，但早段高光和细节较软。raw112只见出画边缘，不推断完整隐藏车体。</p><table><thead><tr><th>原严格改善门</th><th>ΔRGB MAE，0–1</th><th>要求</th><th>结果</th></tr></thead><tbody>{failed_rows}</tbody></table><p>四项失败没有改阈值、换域或删帧。root是在保留这些失败的前提下授权全49诊断与<strong>可选候选</strong>交付，原worker的<code>quality_accepted=false</code>和<code>publication_accepted=false</code>原文保留；另存root结构图检/发布门。<code>human_verdict=null</code>。</p><p>这些49帧是同场景训练视图的重建与工程诊断，不是独立泛化测试。Observed中a37仍在，没有删除后隐藏道路的真实GT。alpha覆盖不能单独证明车体完整；同一BUILD mask也不作为独立语义真值。</p></section>
<section><h2>先四帧，再完整49</h2><p>最初四帧105/113/130/137只含一个a5视角。完整49暴露早段失败后，新四帧改为94/105/130/137并保留原门。下面为最终49BUILD-fit候选的四帧原图板；同帧重放到完整49的编码PNG字节完全相同。</p><details><summary>KEEP / DELETE四帧全图，原480×320直接拼接</summary><img src="figures/pilot_semantic_actor5_all49fit_keep_four_frames.png" alt="94 105 130 137三列KEEP"><img src="figures/pilot_semantic_actor5_all49fit_delete_four_frames.png" alt="94 105 130 137三列DELETE"></details><details><summary>四帧未resize的源PNG</summary><table><thead><tr><th>raw</th><th>原生文件</th></tr></thead><tbody>{native_rows}</tbody></table></details></section>
<section><h2>身份来自实际贡献，来源谱系仍保留</h2><p>49×5共245张原BUILD实例mask已经CPU逐像素校验：原body mask最近邻缩到480×320后，等于保存的actor_labels。a5对应label1；完整M5与补集进入贡献累计，不按alpha选评价域。</p><p>完整Full5 joint的dummy-feature梯度给出M5内贡献与总贡献的未归一化SUM。固定选择条件为inside/total &gt; 0.5且total &gt; 0。最终只读sidecar包含static 232行、dynamic 120,722行（原来源5：118,133行；来源37：2,589行）。捕获这些BUILD视图M5贡献的{calibration['selection_capture_fraction']:.2%}；所选点的M5外贡献占其总贡献{calibration['selection_spill_fraction']:.2%}。这些是贡献比例，不是语义概率或质量通过率。</p><p>QUERY按固定三维行sidecar与模型指纹做行选择，不依赖当帧a5 mask或GT做选择；原v44输入依赖保持，其原始观测载荷可含RGB/actor_labels用于原评估。两状态接入相同语义5行。原来源37中参与a5成像的行被单独纳入语义5保留组，原owner与谱系元数据并未改写；DELETE仍去除v44原物理a37组。没有按bbox、近邻或逐帧渲染mask改行。v44旧static没有退役，新旧表示可能重叠；没有声称获得完美语义分解。</p><details><summary>来源矩阵、BUILD输入与独立sidecar审计</summary><table><thead><tr><th>原来源</th><th>模型行</th><th>选中行</th><th>49视图未观测行</th></tr></thead><tbody>{matrix}</tbody></table><p>{link('evidence/semantic_mask_input_audit.json','245张输入CPU审计')} · {link('source/audit_semantic_mask_inputs.py','审计源码')} · {link('evidence/semantic_actor5_masks49_v1/manifest.json','49张合法BUILD M5来源/SHA')} · {link('evidence/semantic_calibration49/receipt.json','49贡献收据')} · {link('evidence/review/cpu_semantic_actor5_sidecar49_audit.json','独立sidecar CPU复算')} · {link('evidence/semantic_calibration49/semantic_actor5_manifest.json','固定行与权重指纹合同')}</p><p>4个已有view mass精确复用，仅补45次forward/backward；没有把49视图重算记两遍。未观测行不自动视为已知背景。原joint loss只消费动态union，没有当前逐actor RGB/alpha监督合同。</p></details></section>
<section><h2>保留的探索与负结果</h2><table><thead><tr><th>候选</th><th>实际结果</th><th>不可覆盖的原证据</th></tr></thead><tbody>{history_rows}</tbody></table><p>SH桥接控制中，原生SH与相同模型的预计算RGB最大差1.192×10⁻⁷、alpha差0；受控桥接通过不等于混合质量通过，也不能把来源标签当当前语义。</p><details><summary>历史原图板与冻结报告</summary>{history_images}<p>{link('architecture.svg','早期仅按来源ID加法组件图（历史）')}</p></details></section>
<section><h2>复现与交付</h2><p>公开CLI已真实执行raw105 DELETE：1次forward，编码PNG与完整49结果字节相同；CPU describe在隐藏CUDA时也通过。累计本次接入303次raster kernel forward、49次dummy-feature backward，0次优化更新；其中SH首失败3次与重试6次已单列。所有本任务worker已退出、无后续GPU队列；GPU保持开机。</p><p>{link('model/semantic_identity_package49.tar.gz','下载必要语义身份包（17.13 MB）')} · {link('evidence/review/semantic_identity_package49_local_verification.json','本机归档及43文件逐SHA校验')}。包SHA256：<code>{package['archive_sha256']}</code>。包含固定行、贡献累计、来源谱系、协议与依赖代码；没有重复打包Gaussian权重，<strong>不是自含完整v44权重包</strong>。运行还依赖冻结v44资产、已验证Full5 inference10k权重和兼容的原生CUDA扩展。</p><p>原Full5权重备份位于同工作区<code>outputs/scene008_identity_edit_interface_20261107/model/inference10k_delivery.tar.gz</code>（SHA256 <code>6581bfc907f527e322aebdf674c66cd3b75b0cf7321a7f67fff22a0e67277a80</code>）。原训练Adam完整状态仍保留远端；此页的语义包不替代训练恢复包。</p><details><summary>来源、合同、验证、回执与源码</summary><ul>{evidence_list}</ul><p>{link('source/build_report.py','报告入口')} · {link('source/final_report_content.py','最终内容构建器')} · {link('source/publish_optional_viewer.py','可选viewer发布脚本')} · {link('source/qa_optional_viewer.py','中央viewer真实Chrome QA')} · {link('source/qa_report.py','报告真实Chrome QA')} · {link('comparison.csv','逐帧全域/上部RGB表')} · {link('viewer_frames_manifest.json','49动态PNG manifest')} · {link('report_data.json','完整报告数据/输入SHA manifest')}</p></details></section>
<footer>状态：DELIVERED_OPTIONAL_STRUCTURE_REPAIR_CANDIDATE_WITH_PHOTOMETRIC_EXCEPTIONS。原v44默认未替换；human_verdict=null。构建时间{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}。</footer></main></body></html>'''
    with (root/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        fields=['raw','state','domain','pixels','baseline_RGB_MAE','candidate_RGB_MAE','delta_RGB_MAE']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in metrics['comparisons']:
            for domain in ['a5_RGB','a5_upper_RGB']:
                writer.writerow(dict(raw=row['raw'],state=row['state'],domain=domain,
                                     pixels=row['baseline'][domain]['pixels'],baseline_RGB_MAE=row['baseline'][domain]['mean'],
                                     candidate_RGB_MAE=row['candidate'][domain]['mean'],delta_RGB_MAE=row['delta'][domain]))
    (root/'viewer_frames_manifest.json').write_text(json.dumps(dict(scope='saved RGB only; all actual PNGs',raw_ids=list(range(89,138)),
        size=[480,320],paths=all49_data['frame_paths'],source_images=b.image_sources),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (root/'index.html').write_text(page,encoding='utf-8')
    (root/'report_data.json').write_text('{}\n',encoding='utf-8')
    for name in ['pipeline_semantic5.svg','architecture.svg','source/build_report.py','source/final_report_content.py',
                 'source/publish_optional_viewer.py','source/qa_optional_viewer.py','source/qa_report.py']:
        b.inputs.add(root/name)
    qa=b.qa()
    inputs=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(b.inputs)]
    data=dict(status='DELIVERED_OPTIONAL_STRUCTURE_REPAIR_CANDIDATE_WITH_PHOTOMETRIC_EXCEPTIONS',
              active_case=index['full49_case_id'],generated_utc=datetime.now(timezone.utc).isoformat(),
              human_verdict=None,quality_accepted=False,registered_quality_gates_passed=False,
              visible_structure_restored_by_agent_review=True,optional_viewer_release_authorized=True,
              old_v44_still_default=True,worker_original_summary=summary,metrics=metrics,source_integrity=integrity,
              root_visual_review=visual,all49=all49_data,BUILD_calibration49=calibration,semantic_sidecar=sidecar,
              original_mask_input_audit=mask_audit,ledger=ledger,public_CLI=cli,execution_delivery_verification=delivery,
              viewer_publication=publication,viewer_runtime_qa=runtime,report_browser_qa=browser_qa,identity_package=package,fixed_protection_pixel_exact=pixel_proof,
              frozen_v44_findings=findings,figure_sources=figure_sources,input_manifest=inputs,qa=qa)
    (root/'report_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=data['status'],KEEP_a5_change=ratio_full,KEEP_upper_change=ratio_upper,qa=qa),ensure_ascii=True))

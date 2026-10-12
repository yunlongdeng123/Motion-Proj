"""Add canonical B1 long-window evidence to the existing local dark report."""
import hashlib
import html
import json
import shutil

import matplotlib.pyplot as plt


def add_long_original_evidence(here, out, page):
    summary_path=here/'long_original_summary_cpu.json'
    if not summary_path.exists():
        return page,False
    data=json.loads(summary_path.read_text(encoding='utf8'))
    if data['case']!='original30k_to32000' or data['steps']!=2000 or data['verified_PNG_count']!=294:
        raise ValueError('Long B1 is not the complete canonical32k experiment')
    root=here/'continuation_diagnostic/original30k_to32000'
    evidence=out/'evidence/B1_original30k_to32000'
    evidence.mkdir(parents=True,exist_ok=True)
    shutil.copy2(summary_path,evidence/summary_path.name)
    for relative in ['status.json','continuation_diagnostic.json','receipts/full49_summary.json','receipts/local_sync_receipt.json','first_step_reference_comparison.json','runtime_import_pins.json','terminal_32000/diagnostic_snapshot_manifest.json','extra_eval/32000/diagnostic_eval.json']:
        target=evidence/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/relative,target)
    for channel in ('gt','composite','static','dynamic'):
        shutil.copy2(root/'extra_eval/32000'/channel/'0017010.png',out/'assets'/f'B1_original32000_{channel}_raw10.png')
    full=data['domains']['full'];outside=data['domains']['non_sky_outside_actor']
    fm=full['equal_frame_mean'];om=outside['equal_frame_mean'];pw=outside['pixel_weighted_mean']
    fig,axes=plt.subplots(1,2,figsize=(11,3.5),layout='constrained')
    for ax,key,title in zip(axes,['composite_psnr_full','static_psnr_observed_non_sky_outside_actor'],['B1 original objective / composite, same fixed5','B1 original objective / observed static, same fixed5']):
        nodes=data['nodes_same_fixed5']
        ax.plot([n['iteration'] for n in nodes],[n['fixed5_equal_frame_means'][key] for n in nodes],marker='o',color='#70d9c8')
        ax.set(xlabel='Iteration (50k learning-rate horizon)',ylabel='PSNR (dB)',title=title)
        ax.grid(alpha=.3)
    fig.savefig(out/'assets/B1_original32000_fixed5_trend.png',dpi=170);plt.close(fig)
    validation=json.loads((out/'long_B1_video_validation.json').read_text(encoding='utf8'))
    if validation['status']!='B1_LONG_TRUE49_MP4_ALL245_FRAMES_DECODED' or validation['summary_cpu_sha256']!=hashlib.sha256(summary_path.read_bytes()).hexdigest():
        raise ValueError('B1 media validation does not bind reviewed long metrics')
    for video in validation['videos']:
        if hashlib.sha256((out/'assets'/video['file']).read_bytes()).hexdigest()!=video['sha256']:
            raise ValueError('Actual B1 MP4 differs from the all-frame decode receipt')
    video=next(v for v in validation['videos'] if v['channel']=='four_panel')
    links=''.join(f'<a href="assets/{html.escape(v["file"])}">B1 32k {html.escape(v["channel"])} · 49帧 MP4</a>' for v in validation['videos'] if v['channel']!='four_panel')
    cards=''.join(f'<figure><img src="assets/B1_original32000_{channel}_raw10.png" alt="B1真实32000 raw10 {label}"><figcaption>B1 32k · raw10 · {label}</figcaption></figure>' for channel,label in [('gt','原始GT'),('composite','整体重建'),('static','静态／移除通道'),('dynamic','动态通道')])
    section=f'''<section class="section" id="long-b1-32000"><h2>B1长窗已完成：整体恢复，静态分层仍坏</h2><p>原版从相同原始30k完整状态重新恢复，真实训练30,001–32,000共2,000步，保留50k学习率horizon；没有复用短实验30,300的终态。四个节点评估保存并恢复RNG，末次完整评估49帧并保存static／dynamic／environment／training state四件，实际退出码为0。没有进入下一次33k static reset。</p><div class="grid"><div class="metric"><strong>{fm['composite']['clamped_rgb_psnr_db']:.5f} dB</strong><br><small>B1 32k全图composite · 49帧PSNR算术均值</small></div><div class="metric"><strong>{om['static']['clamped_rgb_psnr_db']:.5f} dB</strong><br><small>非天空、车体mask外已观测static · 49帧PSNR算术均值</small></div></div><p>真实raw10中，组合图接近原图；静态移除通道的道路仍是大面积彩噪，动态通道继续包含路面和背景纹理。原样长续训恢复了reset后的整体重建，却没有修复用于编辑的背景分离。</p><table><tr><th>B1 32k指标与区域</th><th>49帧等权均值</th><th>按区域像素汇总</th></tr><tr><td>全图composite PSNR</td><td>{fm['composite']['clamped_rgb_psnr_db']:.9f} dB</td><td>{full['pixel_weighted_mean']['composite']['pooled_mse_psnr_db']:.9f} dB（先汇总MSE）</td></tr><tr><td>已观测非天空车体外static PSNR</td><td>{om['static']['clamped_rgb_psnr_db']:.9f} dB</td><td>{pw['static']['pooled_mse_psnr_db']:.9f} dB（先汇总MSE）</td></tr><tr><td>同一区域动态α</td><td>{om['alpha_d']['mean']:.6f}</td><td>{pw['alpha_d']['mean']:.6f}</td></tr><tr><td>同一区域动态非黑RGB支撑</td><td>{100*om['dynamic']['rgb_support_gt2_uint8_pixel_fraction']:.6f}%</td><td>{100*pw['dynamic']['rgb_support_gt2_uint8_pixel_fraction']:.6f}%</td></tr></table><p class="muted">“49帧等权”是预先约定的主口径；像素汇总按各帧区域面积加权。PSNR汇总先加权MSE再取对数，不是逐帧dB均值。非天空车体外合计{outside['pixel_count_summed']:,}像素，各帧区域面积不同。非黑支撑使用8位RGB任一通道大于2的既有定义，不等同于动态alpha阈值或正确物体分割。车体内遮挡区没有dense edited GT。</p><div class="grid">{cards}</div><p><video controls muted playsinline preload="metadata" poster="assets/B1_original32000_four_panel_decoded_raw10.png" style="width:100%;background:#000;border-radius:8px" src="assets/{html.escape(video['file'])}"></video></p><p class="muted">B1独立32k真实49帧：从左到右为GT、composite、static、dynamic。四面板来自相同相机和迭代，编码在同一时间轴；10fps、4.9秒。另有四份单通道MP4，全部视频共245帧已CPU解码核对。指标来自未压缩浮点结果的原始区域记录，未用有损视频重算；浏览器播放未验证。</p><div class="links">{links}<a href="long_B1_video_validation.json">视频来源与全帧解码验证</a></div><img src="assets/B1_original32000_fixed5_trend.png" alt="B1独立2000步原版续训，同5相机的组合与静态背景质量"><p class="muted">曲线只来自B1长轨迹；30001／30300／31000／32000均取同5帧，32000的49帧评估也只取预定5帧作图。这里的30300节点不是短配对实验的终点数据。</p><p class="warn">B1是原版长对照，不是长候选V结果。不能将B1 32k与短候选30300相减当配对效应；须等B2完整同窗重复、真实采样与49帧域配对，再对同一λ=0.05的长V单独审批和验证。没有正常效果或完整论文复现结论。</p><div class="links"><a href="evidence/B1_original30k_to32000/long_original_summary_cpu.json">独立复算：逐帧均值／像素加权／294PNG来源</a><a href="evidence/B1_original30k_to32000/receipts/full49_summary.json">实际运行完整49帧receipt</a><a href="evidence/B1_original30k_to32000/terminal_32000/diagnostic_snapshot_manifest.json">32k完整四件状态</a></div></section>'''
    anchor='<section class="section"><h2>下一轮受控实验</h2>'
    page=page.replace(anchor,section+anchor)
    page=page.replace('<h2>完整状态短续训：恢复整体画质是否也恢复分层？</h2>','<h2>短实验30k→30300配对：恢复画质是否也恢复分层？</h2>')
    page=page.replace('<h2>同一时间轴的真实 49 帧对照</h2>','<h2>短实验30300：同一时间轴的真实49帧配对</h2>')
    page=page.replace('接下来从相同原始 30k 状态做三组 2,000 步检验：原版、原版重复、同一 λ=0.05 候选；停在 32k，避开下一次 33k reset。',f'长窗B1已完成：全49帧组合PSNR均值{fm["composite"]["clamped_rgb_psnr_db"]:.5f} dB，已观测静态背景仍只有{om["static"]["clamped_rgb_psnr_db"]:.5f} dB，道路分离仍退化。继续同初态原版重复B2，之后才单独批准同一λ=0.05的长V；均停在32k，避开下一次33k reset。')
    from render_long_repeat_report import add_long_repeat_evidence
    page,b2_complete=add_long_repeat_evidence(here,out,page)
    task_path=here/'task_manifest.json'
    if task_path.exists() and not b2_complete:
        task=json.loads(task_path.read_text(encoding='utf8'))
        snapshot=html.escape(task['updated_utc'])
        b2root=here/'continuation_diagnostic/original30k_to32000_noise_reference'
        b2status=json.loads((b2root/'status.json').read_text(encoding='utf8')) if (b2root/'status.json').exists() else {}
        b2first=json.loads((b2root/'first_step_reference_comparison.json').read_text(encoding='utf8')) if (b2root/'first_step_reference_comparison.json').exists() else {}
        first_pass=b2status.get('restore',{}).get('status')=='actual_restore_pass' and len(b2first.get('comparisons',{}))==10 and all(r.get('exact_equal') for r in b2first.get('comparisons',{}).values()) and b2first.get('tolerance_applied') is False
        state_links=[]
        for relative in ('status.json','first_step_reference_comparison.json','runtime_import_pins.json'):
            source=b2root/relative
            if source.exists():
                target=out/'evidence/B2_original30k_to32000_noise_reference'/relative
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
                state_links.append(f'<a href="{target.relative_to(out).as_posix()}">B2 {html.escape(relative)}</a>')
        if task.get('current_experiment_status')=='B2_running':
            last=task.get('latest_runtime_check',{}).get('last_completed_step')
            state='B2原版重复正在独立运行'+(f'；实际完成迭代快照到{last:,}' if isinstance(last,int) else '')+'。'
            state+=('原始30k完整恢复通过，首30001的10项原始标量逐项精确相等，未放宽容差。' if first_pass else '实际恢复与首步核验记录仍待同步。')
        elif b2status.get('status')=='complete' and b2status.get('actual_exit_code')==0:
            state='B2已结束，完整2000步和49帧重复差异仍须后续独立CPU核验；本页未给长候选配对结论。'
        else:
            state='B2已获单独批准，实际启动／恢复状态以root下一次同步记录为准。'
        progress=f'''<section class="section"><h2>长实验状态：B1完成，B2原版重复</h2><p>截至{snapshot}的实际记录，B1原版2000步、末49帧和完整四件终态均已完成且exit0。{state}</p><p>首步等价只证明这项初始核验，不能替代全2000步采样／LR／RNG／GT域配对或同窗原版波动测量。长V尚未启动；须等真实B1/B2完成参考再重新绑定、独立CPU审阅和root单独批准。其它场景仍暂停。</p><div class="links"><a href="evidence/debug_task_status.json">root维护的运行快照</a>{''.join(state_links)}</div></section>'''
        page=page.replace('<section class="section"><h2>四种假说，分别找证据</h2>',progress+'<section class="section"><h2>四种假说，分别找证据</h2>')
    return page,True

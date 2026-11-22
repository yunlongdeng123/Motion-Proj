"""实际场景诊断组适配器；仅读取轻量JSON和既有图像路径。"""
from pathlib import Path
import json,os
from urllib.parse import quote

ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))


def assemble(joint_root, output_root):
    url=lambda p:quote(os.path.relpath(Path(p).resolve(),Path(output_root).resolve()).replace('\\','/'),safe='/._-:')
    image=lambda p:url(p) if Path(p).is_file() else None
    receipt=lambda label,p:dict(label=label,href=url(p))
    groups=[]
    def base(id,label,run):
        return dict(id=id,label=label,state='exported',role='scene_residual_cleanup',stage='context_diagnostic',
            planned_steps=None,actual_steps=0,smoke_counts=None,smoke_passed=None,gate_passed=None,
            scene_qualified=None,real=[],novel=[],views={},receipts=[],range_summary=None,reload_passed=None,
            engineering_gate=None,teacher_bg_alpha_max=None,grey_overview=None,extra_figures=[],
            architecture=url(ROOT/'architecture.svg'),crop_xyxy=[455,390,730,650],internal_grey_is_hard_gate=False,
            scene_acceptance=dict(rgb='待视觉验收',silhouette='待验收',road_contact='待视觉验收',protected_instances='待验收'))
    run=ROOT/'takeback/attribution_r1'
    if (run/'proof.json').is_file():
        proof=read(run/'proof.json')
        if proof.get('complete') is not True:raise ValueError('来源归因收据未完成')
        g=base('scene_attribution_r1','来源归因',run)
        g.update(summary='原生静态背景为横条与黑拖影的主要来源；当前a20与道路资产固定，0参数更新',
            scene_status='实际来源归因完成；尚未作清理质量结论',scene_kind='attribution',
            actual_counts=dict(optimizer_updates=proof['optimizer_updates'],native_calls=proof['native_calls'],feature_VJP_calls=proof['feature_VJP_calls']),
            mode_labels={'shared':'完整场景','source':'静态背景贡献','removed':'移除整类背景（诊断）','difference':'来源移除差值','observed':'实拍RGB','actor':'固定a20','road':'道路单独'},
            default_modes=['shared','source'],scene_table=dict(headers=['相机','横条：背景αT','拖影：背景αT','横条：a20 αT','拖影：a20 αT'],rows=[]))
        for raw in [86,110]:
            d=read(run/f'raw{raw:03d}_metrics.json');layers={x['layer']:x for x in d['layers']}
            val=lambda layer,domain:layers[layer]['domains'][domain]['shared_source_alpha_mean']
            actor=lambda domain:val('a20_old_source0',domain)+val('a20_T_source1',domain)
            g['scene_table']['rows'].append(['raw'+str(raw),f"{val('native_static_background','strip'):.6f}",f"{val('native_static_background','shadow'):.6f}",f"{actor('strip'):.6f}",f"{actor('shadow'):.6g}"])
            names={'shared':'shared','source':'native_static_background_shared_source','removed':'without_native_static_background','difference':'without_native_static_background_difference','observed':'observed_RGB','actor':'actor_only','road':'road_standalone'}
            g['views'][f'raw{raw:03d}']={mode:dict(rgb=image(run/'preview'/f'raw{raw:03d}_{name}_crop.png')) for mode,name in names.items()}
        d=read(run/'raw086_metrics.json');closure=d['additive_closure']
        g['scene_stats']=[['主要来源','原生静态背景','原始归属owner=-1'],['横条 / 拖影：背景αT','0.672 / 0.492','真实86；同深度排序平均贡献'],['当前a20：横条 / 拖影αT','0.086 / ≈0','原始与教师两源之和'],['RGBA贡献闭合',f"{max(closure.values()):.2e}",'真实86；float最大残差']]
        g['receipts']=[receipt('来源收据',run/'proof.json'),receipt('按源贡献',run/'metrics.json'),receipt('实际调用账目',run/'resource_ledger.json'),receipt('截图定位：真实86',ROOT/'user_reference_registration.json'),receipt('008只读参考',ROOT/'reference_008.json')]
        groups.append(g)
    run=ROOT/'takeback/cleanup_red64_r1'
    if (run/'proof.json').is_file():
        proof=read(run/'proof.json')
        if proof.get('complete') is not True:raise ValueError('局部清退收据未完成')
        ledger=read(run/'cleanup_ledger.json')
        g=base('scene_cleanup_red64_r1','局部清退（270行，0更新）',run)
        g.update(summary='272候选：270接受、2拒绝；23次整组移除检验。两帧保护检查通过，残影视觉待验收',
            scene_status='局部清退实际完成；两帧保护通过；RGB / 接地视觉待验收',scene_kind='cleanup',
            actual_counts=dict(optimizer_updates=proof['optimizer_updates'],native_calls=proof['native_calls'],accepted_context_rows=proof['accepted_context_rows'],rejected_rows=proof['rejected_rows'],whole_set_trials=ledger['trials']),
            mode_labels={'before':'清理前','after':'清理后','difference':'前后差值','road':'道路单独'},default_modes=['before','after'],
            scene_acceptance=dict(rgb='待视觉验收',silhouette='a20完全不变',road_contact='无新增孔洞；接地待视检',protected_instances='两帧RGB / Alpha差0'),
            scene_stats=[['RGB残影','待视觉验收','尚未登记清除成功'],['轮廓 / 固定a20','完全不变','RGB / Alpha / 深度差0'],['接地保护','新增孔洞0','物理接地仍待视觉验收'],['其他车辆保护','RGB / Alpha差0','真实86 / 110整组检查通过']],
            scene_table=dict(headers=['相机','保留车RGB max','保留车Alpha max','区域外RGB max','新增地面孔洞'],rows=[]))
        for guard in proof['gates']:
            raw=guard['raw'];g['scene_table']['rows'].append(['raw'+str(raw),f"{guard['K_RGB_max']:.6g}",f"{guard['K_visible_alpha_max']:.6g}",f"{guard['outside_edit_RGB_max']:.6g}",str(guard['new_ground_alpha_below95_pixels'])])
            names={'before':'before','after':'after','difference':'before_after_difference','road':'road_standalone'}
            g['views'][f'raw{raw:03d}']={mode:dict(rgb=image(run/'preview'/f'raw{raw:03d}_{name}_crop.png')) for mode,name in names.items()}
        g['receipts']=[receipt('清退收据',run/'proof.json'),receipt('逐次整组检验',run/'cleanup_ledger.json'),receipt('实际调用账目',run/'resource_ledger.json'),receipt('恢复sidecar绑定',run/'scene_context_binding.json')]
        g['sidecar']=proof['context_sidecar'];g['reload_passed']=all(all(n==0 for n in v['maximum'].values()) for v in proof['sidecar_reload_parity'])
        # 可选的root视觉判断独立覆盖四项验收；不从数值guard推断RGB清理成功。
        review_path=ROOT/'report_integration/root_scene_review.json'
        if review_path.is_file():
            review=read(review_path)
            if review.get('group_id')==g['id']:
                g['scene_acceptance']=review['acceptance'];g['scene_qualified']=review.get('scene_qualified')
                g['scene_status']=review['summary'];g['receipts'].append(receipt('视觉验收',review_path))
        groups.append(g)
    stages=[
        ('cleanup_visible_red_r1','scene_cleanup_visible_red_r1','亮红条清退（959行，0更新）','亮红条基本去除；整体部分改善，暗雾仍明显'),
        ('cleanup_red_black_r2_retry1','scene_cleanup_red_black_r2','静态残影清退（累计1468行，0更新）','继承959行并新增509行；部分改善，暗雾仍明显'),
        ('cleanup_dynamic_strays_r3','scene_cleanup_dynamic_strays_r3','外露杂散清退（累计1500行，0更新）','1468静态行＋32条外露动态杂散；部分改善，暗雾仍明显'),
        ('cleanup_dark_static_r4','scene_cleanup_dark_static_r4','暗残影清退（累计1754行，0更新）','1722静态行＋32条外露动态杂散；相较1500行改善很小，灰紫暗雾仍在'),
        ('cleanup_rgb_residual_r5','scene_cleanup_rgb_residual_r5','RGB残影清退（累计2006行，0更新）','1959静态行＋47条动态杂散；紫雾去除，但新黑拖影暴露，最终质量未通过')]
    for folder,id,label,visual_summary in stages:
        run=ROOT/'takeback'/folder
        if not (run/'proof.json').is_file():continue
        proof=read(run/'proof.json');ledger=read(run/'cleanup_ledger.json')
        if proof.get('complete') is not True:raise ValueError('清退收据未完成：'+folder)
        if proof['actor_asset']['sha256']!='ab4fa110440d579bcfd57e4bd0b8a7f61c3f7821189450b58bd027c2f28f67a7':raise ValueError('固定a20资产SHA不符')
        dynamic=proof.get('other_actor_stray_rows_retired',0)
        g=base(id,label,run)
        g.update(summary=visual_summary+'；两帧保护通过，a20和道路字节不变',
            scene_status=visual_summary+'；当前任务最终质量未定',scene_kind='cleanup',visual_review='partial',
            actual_counts=dict(optimizer_updates=proof['optimizer_updates'],native_calls=proof['native_calls'],accepted_context_rows=proof['accepted_context_rows'],rejected_rows=proof['rejected_rows'],whole_set_trials=ledger['trials']),
            mode_labels={'before':'本次清理前','after':'本次清理后','difference':'本次差值','road':'道路单独'},default_modes=['before','after'],
            scene_acceptance=dict(rgb='部分改善；暗雾仍明显',silhouette='a20完全不变',road_contact='无新增孔洞；接地未作最终通过',protected_instances='两帧RGB / Alpha差0'),
            scene_stats=[['RGB残影','部分改善','亮红条基本去除；暗雾仍明显'],['轮廓 / 固定a20','完全不变','RGB / Alpha / 深度差0'],['接地保护','新增孔洞0','覆盖检查通过；最终接地未定'],['其他车辆保护','RGB / Alpha差0','完整保留区域逐像素检查']],
            scene_table=dict(headers=['相机','保留车RGB max','保留车Alpha max','区域外RGB max','新增地面孔洞'],rows=[]))
        names={'before':'before','after':'after','difference':'before_after_difference','road':'road_standalone'}
        if (run/'preview/raw086_initial_crop.png').is_file():
            names.update(initial='initial',cumulative_difference='initial_to_after_difference')
            g['mode_labels'].update(initial='初始场景',cumulative_difference='初始至当前差值');g['default_modes']=['initial','after']
        for guard in proof['gates']:
            raw=guard['raw'];g['scene_table']['rows'].append(['raw'+str(raw),f"{guard['K_RGB_max']:.6g}",f"{guard['K_visible_alpha_max']:.6g}",f"{guard['outside_edit_RGB_max']:.6g}",str(guard['new_ground_alpha_below95_pixels'])])
            g['views'][f'raw{raw:03d}']={mode:dict(rgb=image(run/'preview'/f'raw{raw:03d}_{name}_crop.png')) for mode,name in names.items()}
        g['receipts']=[receipt('清退收据',run/'proof.json'),receipt('整组检验与继承账目',run/'cleanup_ledger.json'),receipt('实际调用账目',run/'resource_ledger.json'),receipt('恢复sidecar绑定',run/'scene_context_binding.json')]
        g['sidecar']=proof['context_sidecar'];g['reload_passed']=all(all(n==0 for n in v['maximum'].values()) for v in proof['sidecar_reload_parity'])
        if dynamic:
            g['dynamic_stray_rows']=dynamic;g['whole_actor_deletions']=proof['whole_actor_deletions']
            g['checked_owner_masks_exact_zero']=proof['checked_owner_masks_exact_zero']
            g['summary']+=f'；{dynamic}条杂散属于原动态来源8/14，其可见车辆区域RGB/Alpha差0，未删除车辆主体'
            g['scene_stats'][3]=['其他车辆保护','完整可见区差0','来源8/14可见区域另行核验']
        if folder=='cleanup_dark_static_r4':
            g['scene_stats'][0]=['RGB残影','改善很小','相较1500行；灰紫暗雾仍明显']
            g['cumulative_static_context_rows']=proof['cumulative_static_context_rows']
            g['new_static_context_rows_retired']=proof['new_static_context_rows_retired']
            diagnostic=ROOT/'residual_diagnostics.json'
            if diagnostic.is_file():
                data=read(diagnostic);g['residual_diagnostic_reference']=data['reference'];g['residual_diagnostics']=[]
                for view in data['views']:
                    stages_by_name={x['stage']:x for x in view['stages']};before=stages_by_name['original'];after=stages_by_name[folder]
                    g['residual_diagnostics'].append([str(view['raw']),str(before['dark_gap_gt_08_pixels'])+' → '+str(after['dark_gap_gt_08_pixels']),str(before['bright_gap_gt_08_pixels'])+' → '+str(after['bright_gap_gt_08_pixels']),f"{before['absolute_RGB_gap_p95']:.3f} → {after['absolute_RGB_gap_p95']:.3f}"])
                g['receipts'].append(receipt('冻结道路残差诊断',diagnostic))
        if folder=='cleanup_rgb_residual_r5':
            g['scene_stats'][0]=['RGB残影','新黑拖影暴露','紫雾去除；整体未通过最终验收']
            g['scene_acceptance']['rgb']='紫雾去除但新黑拖影暴露；部分改善'
            g['cumulative_static_context_rows']=proof['cumulative_static_context_rows']
            g['cumulative_dynamic_context_rows']=proof['cumulative_dynamic_context_rows']
        groups.append(g)
    for g in groups:
        g['diagnostic_ellipses']={'raw086':[128.5,155,69.5,34],'raw110':[141,195,86,35]}
        g['scene_summary']=g['summary']+'；RGB / 轮廓 / 接地 / 保留车为主验收，内部灰模仅诊断。'
        if not all(any(v for channels in modes.values() for v in channels.values()) for modes in g['views'].values()):raise ValueError('实际组缺少对应图像')
    return groups

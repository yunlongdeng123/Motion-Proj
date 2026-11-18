"""只读拼接本轮机械/公平性/冻结质量门；不训练、不渲染、不赋人类视觉结论。"""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib

P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for value in iter(lambda:f.read(8*1024**2),b''):h.update(value)
    return h.hexdigest()


def markdown(result):
    t=result['material_table'];lines=[
        '# 共享多尺度道路材质验收',
        '',
        '机械与持久化验收通过。共享多尺度方案通过三帧条纹相对下降门，但六项逐视角拟合非劣门均未通过，因此本轮未联合通过质量验收。人类视觉结论保持空值，不晋升为默认资产。',
        '',
        '```mermaid',result['component_diagram_mermaid'],'```','',
        'A 为真实可见 RGB；C 为冻结孔洞伪教师；S 为冻结阴影区域伪教师。仅使用 86/110 训练，134/179/198 为诊断视角；继承的 199 帧构建输入使其不能称为盲测。',
        '',
        '## 固定条件与公平性','',
        '- 同一道路覆盖、84,266 行写许可、过滤后的零修正初值、A/C/S 掩码、q=1 和 800 步平衡序列；每个训练视角各 400 步。',
        '- 低频控制与多阶段方案复用一次 A200 前缀。多阶段方案为低频 200、高频 300、联合 300；低频控制续训 600。',
        '- 独立颜色控制采用同初值的逐高斯颜色修正。共享场与独立颜色的正则不同，是固定监督/采样/步数的策略比较，不能称为完全单变量消融或等算力比较。',
        '- 固定道路几何、法向、尺度、opacity、active 与相机；全五帧道路 alpha 相同，77,266 个保护像素的联合 RGB 差为精确 0。',
        '',
        '## 材质误差','',
        '下表为原生道路预乘 RGB 与固定 alpha × 教师的像素/通道平均绝对误差；C/S 的下降表示伪教师一致性。表格按像素汇总，训练则按两个视角平衡采样；逐视角质量门独立判定。',
        '',
        '| 列 | A 真实可见 | C 伪教师 | S 支持3131 | 旧2222 | 新909 | 全S3187 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    names={'filtered_A0':'过滤初值','L800':'共享低频800','ABC800':'共享多阶段800','matched_I0':'匹配独立颜色800','A0_midpoint':'中点积分初值','TRELLIS2':'TRELLIS.2零样本'}
    for key,label in names.items():
        vals=[t[key][region]['premult']['MAE']for region in('A','C','S','old2222','new909','full3187')]
        lines.append('| '+label+' | '+' | '.join(f'{v:.6f}'for v in vals)+' |')
    lines+=['','全 S 的 3187 个像素帧均保留：3131 支持、56 薄覆盖；原 D38416 为支持37404、薄覆盖929、孔洞83，所有列不变。薄覆盖56的低预乘误差不等价于材料正确，因此同时保存不透明教师误差与联合误差。','',
        '| 列 | 全S不透明误差 \|P−T\| | 全S预乘误差 | 全S联合误差 \|J−T\| |',
        '|---|---:|---:|---:|']
    for key in('filtered_A0','L800','ABC800','matched_I0','A0_midpoint','TRELLIS2'):
        d=t[key]['full3187'];lines.append('| '+names[key]+' | '+' | '.join(f"{d[k]['MAE']:.6f}"for k in('opaque','premult','joint'))+' |')
    lines+=['','## 预登记质量门','',
        '拟合门要求共享多阶段方案相对匹配独立颜色的每视角 A/C/S 误差增量不超过 1/255（0.003922）。','',
        '| 域 | 86增量 | 110增量 | 结果 |','|---|---:|---:|---|']
    fit=result['gates']['fit_noninferiority']['rows']
    for region in('A','C','S'):
        vals=[next(r['ABC_minus_matchedI0_premult_MAE']for r in fit if r['raw']==raw and r['region']==region)for raw in(86,110)]
        lines.append(f'| {region} | {vals[0]:.6f} | {vals[1]:.6f} | FAIL |')
    lines+=['','A 相对过滤初值的保留门为 PASS：86/110 误差均下降。','',
        '| 诊断帧 | 初值条纹指标 | 独立颜色 | 共享多阶段 | 正增量下降 | 50%门 |',
        '|---|---:|---:|---:|---:|---|']
    for row in result['gates']['diagnostic_stripe_positive_excess']['rows']:
        lines.append(f"| {row['raw']} | {row['filtered_A0_baseline']:.6f} | {row['matched_I0']:.6f} | {row['ABC']:.6f} | {100*row['excess_reduction']:.2f}% | PASS |")
    lines+=['','该相对下降未消除条纹：共享方案的原始指标仍高于过滤初值。共享多阶段相对低频控制的整体差异较小；完整逐视角增量见 JSON。','',
        '## 条纹乘积分解','',
        '仅在原冻结三像素模板且三邻道路 alpha 均大于0.9的诊断子集计算 c=P/alpha，不用于训练。',
        '',
        '`D²P = c₀D²alpha + alpha₀D²c + (alpha₊−alpha₀)(c₊−c₀) + (alpha₋−alpha₀)(c₋−c₀)`','',
        f"浮点64重建最大误差为 {result['stripe_product_decomposition']['reconstruction_max']:.3g}。共享方案的有效颜色二阶差分仍高于初值；不能仅用固定 alpha 下变亮解释剩余指标。各项 meanabs 不可相加解释为因果百分比，原质量门保持不变。",'',
        '## 来源、持久化与零样本先验','',
        '- 高频修正仅许可原观测来源；检索、生成与未知来源的高频许可为0。原始来源、known/unknown、观测颜色与 UV 台账保持。',
        '- 低有效核质量是数值统计分层，不是0.9训练资格门；S的3131分母含新增909，没有缩为旧2222。',
        '- 权威共享场保存低/高频参数与物理采样核，重载后重新查询再进入原生GS；行颜色是派生缓存。完整状态、Adam、随机状态、采样序列和下一步均经过真实短程。',
        '- TRELLIS.2为一次新推理，使用真实A与冻结C/S条件；同中点积分初值用于隔离采样器差异。S误差下降而真实A与C误差恶化，不能据此主张整体胜出。无独立真实隐藏道路/标线结构GT。',
        '',
        '## 更新与运行成本','',
        f"正式物理更新 {result['budget']['formal_physical_updates']}：共享分支1400（前缀200仅执行一次）＋独立颜色800。工程实际更新 {result['budget']['engineering_actual_updates']}：成功共享短程360、旧失败4、独立颜色短程4；合计实际优化器更新 {result['budget']['total_actual_optimizer_updates']}。TRELLIS.2新增推理1次、优化器更新0。",'',
        f"共享两分支1400更新与阶段导出计时 {result['runtime']['shared_pair1400_seconds']:.2f} 秒；独立颜色800更新与导出 {result['runtime']['matched_I0800_seconds']:.2f} 秒。两者计时均不含构造器；同步数不代表等GPU算力。",'',
        '## 证据','']
    lines+=['- `'+path+'`'for path in result['evidence_files']]
    return '\n'.join(lines)+'\n'


def main():
    q=P/'qa';metrics=read(q/'result_metrics.json');states=read(q/'field_stage_acceptance.json')
    fairness=read(q/'comparator_fairness.json');stripe=read(q/'stripe_product_decomposition.json')
    field=read(P/'formal_r1/results.json');fieldsmoke=read(P/'lifecycle_final_readonly_r1/verified_smoke_receipt.json')
    I0=read(P/'matched_I0/formal_r1/completed.json');I0smoke=read(P/'matched_I0/smoke_r1/completed.json')
    T=read(P/'trellis2/native_columns_r1/completed.json')
    fits=metrics['matched_I0_fit_gate'];stripes=metrics['diagnostic_stripe_comparison'];A_guard=metrics['real_A_initial_guard']
    mechanical=metrics['mechanical_status']=='pass_float_packets'and states['overall']=='pass'and fairness['overall']=='pass'and stripe['status']=='pass_identity_only'and T['completed']
    quality_ready=len(fits)==6 and len(stripes)==3 and len(A_guard)==2
    joint_quality=quality_ready and all(x['passed']for x in fits+stripes+A_guard)
    selected=['filtered_A0','L800','ABC800','matched_I0','A0_midpoint','TRELLIS2']
    table={name:metrics['stages'][name]['pooled']for name in selected}
    scope=read(P/'method_scope_r1/receipt.json');source=read(P/'source_kernel_impact_r2/receipt.json')
    # r2 source-only迁移复用旧A2；7+353是新实际调用，旧失败4另计，未重复记borrowed prefix2。
    shared_good=fieldsmoke['optimizer_updates'];shared_failed=fieldsmoke['prior_failed_attempt_updates']
    control_smoke=I0smoke['engineering_physical_updates'];engineering=shared_good+shared_failed+control_smoke
    formal=field['formal_physical_updates']+I0['optimizer_updates']
    checks=dict(all11_native_float_columns=metrics['mechanical_status']=='pass_float_packets',
        all_shared_field_states=states['overall']=='pass',same_initialization_scope_sampler_targets=fairness['overall']=='pass',
        shared_prefix_reuse_physical1400=field['formal_physical_updates']==1400,
        matched_I0_fresh800=I0['optimizer_updates']==800 and I0['resumed_from']is None,
        product_identity=stripe['status']=='pass_identity_only',TRELLIS_two_columns_zero_update=T['optimizer_updates']==0 and T['native_forward_calls']==30)
    mechanical=mechanical and all(checks.values())
    component={name:states['stages'][name]['details']for name in states['stages']}
    result=dict(schema='multiscale_world_material_final_acceptance.v1',campaign=str(P),frozen_UTC=datetime.now(timezone.utc).isoformat(),
        overall_status='MECHANICAL_PASS_QUALITY_FAIL'if mechanical and quality_ready and not joint_quality else'PASS_NUMERICAL_GATES_ONLY'if mechanical and joint_quality else'INCOMPLETE_OR_MECHANICAL_FAIL',
        mechanical_status='PASS'if mechanical else'FAIL',numerical_quality_status='PASS'if joint_quality else'FAIL'if quality_ready else'INCOMPLETE',
        mechanical_checks=checks,mechanical_check_counts=dict(float_source_and_alpha_K=len(metrics['mechanical_checks']),
            shared_state=sum(len(s['checks'])for s in states['stages'].values()),fairness_and_I0_state=len(fairness['checks']),product_identity=len(stripe['checks'])),
        gates=dict(fit_noninferiority=dict(status='PASS'if all(x['passed']for x in fits)else'FAIL',per_view_domain_tolerance=1/255,rows=fits),
            real_A_initial_guard=dict(status='PASS'if all(x['passed']for x in A_guard)else'FAIL',rows=A_guard),
            diagnostic_stripe_positive_excess=dict(status='PASS'if all(x['passed']for x in stripes)else'FAIL',max_fraction_of_matched_I0=.5,rows=stripes)),
        fixed_denominators=metrics['fixed_denominators'],source_counts=metrics['source_counts'],scientific_rows=scope['qualified_rows'],
        fixed_coverage=dict(original_D=dict(pixels=38416,gap=83,thin=929,supported=37404),
            fullS=dict(pixels=3187,gap=0,thin=56,supported=3131),all11_alpha_exact_parent=True,all11_joint_K77266_exact=True),
        material_table=table,per_view_metric_file=str(q/'result_metrics.json'),phase_table_csv=str(q/'result_metrics.csv'),
        shared_component_statistics=component,ABC_minus_L800=metrics['matched_L_strategy_deltas'],
        stripe_product_decomposition=dict(file=str(q/'stripe_product_decomposition.json'),csv=str(q/'stripe_product_decomposition.csv'),
            reconstruction_max=max(r['reconstruction_max']for r in stripe['rows']),
            interpretation='c_eff diagnostics retain a sizable color second-difference after normalization; term meanabs are nonadditive, never causal percentages; original quality gates unchanged'),
        sources=dict(A='real visible RGB',C='frozen2D hole pseudo',S='frozen2D shadow pseudo',q='binary1 inside each source-valid domain, not calibrated confidence',
            fixed_A0='oldS2 base_atlas F0 plus32 frozen separately tagged B32 pseudo nodes; learned corrections remain separate',
            unavailable_readonly_rows=source['unavailable_readonly_rows'],low_valid_mass='point9 is numeric stratum only, never a training eligibility cutoff',
            S_kernel_summary=next(r for r in source['region_rows']if r['region']=='S'),H_permission='only measuredsource1; retrieved/generated/unknown high permission0'),
        comparison_boundary=dict(equal_update_budget=True,equal_GPU_compute=False,
            each_method_effective_updates=dict(L800=800,ABC800=800,matched_I0=800),
            regularizers='shared L/H own priors differ from independentcolor effective-delta squared*.001; not a pure one-variable change',
            historical_I0='oldS2222 supervision only, excluded from matched gates',TRELLIS='new zero-shot prior conditioned on measured A + frozen C/S pseudo; compare with same-midpoint A0, not equal-training control'),
        budget=dict(scope='this multiscale campaign only; older capacity/activation campaigns excluded',formal_physical_updates=formal,
            shared_formal_physical_updates=field['formal_physical_updates'],common_prefix_reused_once=200,matched_I0_formal_updates=I0['optimizer_updates'],
            engineering_actual_updates=engineering,shared_successful_smoke_updates=shared_good,shared_prior_failed_attempt_updates=shared_failed,
            matched_I0_engineering_updates=control_smoke,total_actual_optimizer_updates=formal+engineering,
            TRELLIS_new_inference_calls=1,TRELLIS_optimizer_updates=0,TRELLIS_native_forward_calls=T['native_forward_calls'],
            shared_formal_native=dict(forward=field['native_calls_training_context']+field['additional_public_reload_calls'],backward=1400),
            matched_I0_formal_native=dict(forward=I0['native_forward_calls_in_context']+I0['additional_public_clone_forwards'],backward=I0['native_backward_calls']),
            matched_I0_smoke_native=dict(forward=I0smoke['native_forward_calls_in_context']+I0smoke['additional_public_clone_forwards'],backward=I0smoke['native_backward_calls']),
            native_count_boundary='do not sum overlapping lifecycle subpath counts; stage0/probes/failed-attempt calls remain in their own receipts'),
        runtime=dict(shared_pair1400_seconds=field['seconds'],matched_I0800_seconds=I0['elapsed_seconds'],
            scope='training/evaluation/export function timers, context constructors excluded; same800 updates does not mean equal GPU time',
            shared_real_smoke_seconds=fieldsmoke['elapsed_seconds'],matched_I0_smoke_seconds=I0smoke['elapsed_seconds']),
        views=dict(train=[86,110],diagnostic=[134,179,198],inherited_BUILD_frames=199,blind_test=False),
        strict_negative_capacity_certificate=False,real_marking_structure='no independent real structuralGT; ideal12cm targets are not used to score the real road',
        visual_verdict=None,human_verdict=None,production_promoted=False,GPU_keep_on=True,
        conclusions=['ABC reduces the fixed diagnostic stripe proxy relative to matched independentcolor but still exceeds filteredA0.',
            'ABC fails every registered A/C/S per-view noninferiority comparison; no joint quality acceptance or default promotion.',
            'ABC-L800 is small at this fixed budget. Finite-budget outcomes do not prove a shared-field representation limit.',
            'TRELLIS improves S pseudo consistency in its conditioned patch while degrading measured A; no overall superiority claim.'],
        evidence_files=[str(q/x)for x in ['result_metrics.json','result_metrics.csv','field_stage_acceptance.json','comparator_fairness.json','stripe_product_decomposition.json']],
        component_diagram_mermaid='flowchart LR\n A[A0 frozen world prior] --> F[shared low and high fields]\n G[fixed GS surface and Sigma] --> Q[physical footprint sampling]\n G --> R[native GS road and joint renderer]\n F --> Q\n Q --> R\n I[same ci0 independent RGB control] --> R\n T[new TRELLIS mesh UV prior] --> Q\n R --> E[fixed A C S and stripe evaluation]')
    files=result['evidence_files']+[str(P/'formal_r1/results.json'),str(P/'matched_I0/formal_r1/completed.json'),str(P/'trellis2/native_columns_r1/completed.json')]
    result['evidence_sha256']={path:sha(path)for path in files}
    (q/'final_acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (q/'final_acceptance.md').write_text(markdown(result),encoding='utf-8')
    owned=['road_chart_sampling.py','prepare_chart_plan_cpu.py','check_chart_against_aabb_cpu.py','qualify_filter_cpu.py',
        'matched_i0_training.py','prepare_matched_i0_cpu.py','run_matched_i0.py','qa_world_material_results.py',
        'audit_field_states_cpu.py','audit_comparator_fairness_cpu.py','decompose_native_stripes_cpu.py','assemble_world_material_acceptance_cpu.py']
    inventory=dict(schema='stage_QA_owned_code.v1',campaign=str(P),files=[dict(path=str(P/'code'/name),relative_path='code/'+name,sha256=sha(P/'code'/name))for name in owned],
        data_NPZ_in_publication=False,CPU_fixture_entry='CUDA_VISIBLE_DEVICES= drivingeditor/bin/python code/prepare_matched_i0_cpu.py --mode fixture',
        native_sources_frozen_at='matched_I0/smoke_r1_sources + formal_r1_sources; QA result readers remain separate',
        ownership_boundary='chart fix implemented here; independent geometry/numeric acceptance supplied by root and static AABB crosscheck, not self-certified')
    (q/'owned_source_inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(q/'final_acceptance.json'),overall=result['overall_status'],formal=formal,engineering=engineering,
        total_updates=formal+engineering,check_counts=result['mechanical_check_counts'])))


if __name__=='__main__':main()

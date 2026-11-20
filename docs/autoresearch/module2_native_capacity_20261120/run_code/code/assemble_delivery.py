"""合并固定阶段的已计算结果；不重算指标、不选checkpoint。"""
from pathlib import Path
import json,hashlib

ROOT=Path(__file__).resolve().parents[2]
O=ROOT/'outputs/road_native_capacity_20261120_r1'
def load(p):return json.loads((O/p).read_text('utf-8'))
def write(p,x):(O/p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def f(x,n=4):return 'N/A'if x is None else f'{x:.{n}f}'

if not(O/'evaluation/query_results.json').exists():(O/'evaluation/query_results.json').write_bytes((O/'evaluation/results.json').read_bytes())
q=load('evaluation/query_results.json');num=load('evaluation/numerical_results.json');b=load('evaluation_build/results.json')
for r in q['rows']+num['rows']:r['phase']='QUERY'
for r in b['rows']:r['phase']='BUILD';r['split']='BUILD_fit_all_qualified'
combined=dict(schema='native_capacity_delivery_phases.v1',rows=q['rows']+num['rows']+b['rows'],ideal_rows=q['ideal_rows'],
    phases=dict(QUERY='prior-conditioned spatial new-label holdout, not blind',BUILD='all-qualified teacher fitting, former held labels used'),
    source_files=['evaluation/query_results.json','evaluation/numerical_results.json','evaluation_build/results.json'],
    source_SHA256={str(p):hashlib.sha256((O/p).read_bytes()).hexdigest()for p in ['evaluation/query_results.json','evaluation/numerical_results.json','evaluation_build/results.json']},
    quality_PASS=False,human_verdict=None,production_promoted=False)
write('evaluation/results.json',combined)
if not(O/'previews/assets_query.json').exists():(O/'previews/assets_query.json').write_bytes((O/'previews/assets.json').read_bytes())
assets=load('previews/assets_query.json')
for extra in ['previews/assets_numeric.json','previews/assets_build.json']:
    a=load(extra)
    for raw,frame in a['frames'].items():assets['frames'][raw]['methods'].update(frame['methods'])
    assets.setdefault('additional_source_manifests',[]).append(dict(path=extra,source_manifest=a['source_manifest'],SHA256=a['source_manifest_sha256']))
write('previews/assets.json',assets)
main=load('formal_results.json');nr=load('numerical_results.json');br=load('build_results.json')
field=br['field'];bn=br['numerical']if'numerical'in br else br['numeric']
def row(arm,raw,source):return next(r for r in b['rows']if r['arm']==arm and r['raw']==raw and r['source']==source and r['scale']==1)
c=[row('G_BUILD_FIELD',r,'C')for r in [86,110]];s=[row('G_BUILD_FIELD',r,'S')for r in [86,110]]
opt=[]
for arm in ['G_RGB','G_DETAIL','G_ORACLE']:
    a=main['arms'][arm];e=a['extra'];opt.append(['留出对照',arm,str(e.get('logical_steps',e.get('accepted_steps'))),f(a['endpoint_common_DETAIL'],6),f(e.get('last_train_record',{}).get('relative_measured_FW_gap'),4)])
opt+=[['留出对照','有界数值补证','2000 FG',f(nr['extra']['original_FP32_objective'],6),f(nr['extra']['relative_training_FW_gap'],4)],
      ['全标签构建','共享材质','1200',f(field['final_postupdate_BUILD_DETAIL'],6),'N/A'],
      ['全标签构建','有界数值补证','2000 FG',f(bn['final_postupdate_BUILD_DETAIL'],6),f(bn['extra']['relative_training_FW_gap'],4)]]
cov=[]
for raw in [86,110]:
    r=next(x for x in q['coverage_rows']if x['arm']=='F0'and x['raw']==raw and x['source']=='C')
    high=next(z['pixels']for z in r['coverage']if z['alpha_threshold']==.95)
    cov.append([str(raw),str(r['expected_pre_alpha']),str(r['unmapped']),str(high),str(r['visibility'].get('high_road_alpha_but_joint_road_below05','N/A'))])
ideal={}
for r in q['ideal_rows']:
    ideal[str(r['raw'])]=[[str(r['qualified_pixels']),f(r['contrast_retention'],6),f(r['edge_F1_1px'],4),str(r['old_combined_detail_gate'])]]
summary=dict(findings=[
 dict(title='完整教师监督已补齐',text='中央彩色块曾落在刻意留出的世界块内。保留原对照后，用锁定方案完成全标签资产构建；新结果只作拟合验收。'),
 dict(title='细节仍未复现教师',text=f"全标签共享材质的孔洞σ1高频对应度为 {f(c[0]['HF_cosine'])} / {f(c[1]['HF_cosine'])}；阴影能量比为 {f(s[0]['HF_energy_ratio'],3)} / {f(s[1]['HF_energy_ratio'],3)}。更多能量不等于正确纹理。"),
 dict(title='尚无全局表示上限证据',text=f"全标签有界颜色控制目标为 {f(bn['final_postupdate_BUILD_DETAIL'],6)}，训练FW比例仍为 {bn['extra']['relative_training_FW_gap']:.1%}，未达到1%数值门；只读RGB范围限制已独立确认。")],
 scope='原生1920×1280；两帧教师，五视角重载。C/S是伪监督。资产构建使用全部合格标签；留出对照有历史预暴露。几何、覆盖、车辆、相机与生成模型保持冻结。',
 optimization=dict(heads=['阶段','参数化 / 求解','预算','同域细节目标','FW / 目标'],rows=opt,
 note='不同阶段的监督域和初值归一化数值不同，不能跨阶段直接排名。数值门未满足；固定预算结束后没有重启、扫参或按验证选终点。'),
 coverage=dict(heads=['帧','完整孔洞域','无固定表面交点','道路α≥.95','高覆盖但被静态层挡住'],rows=cov),
 persistence=['40份唯一原生公共packet覆盖五视角；来源与公共加载器的道路/联合RGB及alpha逐值一致。','保留车辆K区域变化为0；原观察来源台账与非颜色几何保持。','当前仍是研究候选，未替换默认场景。完整标线几何与真实隐藏道路没有GT。'],
 execution=['正式留出对照：两组共享材质各1200步，独立颜色4000步，理想标线1000步；主合同7400更新保留。','目标保持的数值补证：固定细节组起点，2000FG；全标签构建：固定共享材质1200步，再2000FG数值控制。没有调整学习率、频率或损失权重。','完整smoke、工程恢复、前向/反向与资产取回分项记录在计算账目。GPU保持开机。'],
 ideal_note='独立连续世界函数，12cm标线，固定合格范围。raw86对比度略低于.90门，不能据此宣称表示失败；raw110通过只说明存在该参考下的可行表现。',
 ideal_table=dict(heads=['合格像素','对比度保留','边缘F1','既定联合门'],rows_by_raw=ideal))
write('report_summary.json',summary)
table='| 帧 | 源 | 参数化 | RGB MAE | HF对应 | HF能量比 | HF相对RMSE |\n|---|---|---|---:|---:|---:|---:|\n'
for raw in [86,110]:
    for source in ['C','S']:
        for arm in ['F0','G_BUILD_FIELD','G_BUILD_NUMERIC']:
            r=row(arm,raw,source);table+=f"| {raw} | {source} | {arm} | {f(r['RGB_MAE'],5)} | {f(r['HF_cosine'])} | {f(r['HF_energy_ratio'],3)} | {f(r['HF_relative_RMSE'],3)} |\n"
report=f'''# 原生道路材质蒸馏与固定高斯容量

任务：`MODULE2-NATIVE-CAPACITY-20261120`。本波已完成，后续不追加实验，GPU保持开机。

![组件](architecture.svg)

保留原始留出对照、数值控制及全部负结果后，按同一固定配方完成全标签资产构建。中央彩色块的显著部分曾属于未给本轮标签的2m留出世界块，不能用该完整图直接判表示失败。BUILD现在合并全部合格教师标签；它是每街段拟合，不是留出验证。C/S始终为冻结伪教师，不是真实车下道路答案。

两帧教师原始分辨率1920×1280直接进入实际普通高斯光栅化损失。共享场保持12cm低频/2cm高频与GH31足迹读取，几何、尺度、不透明度、可写89887行及其余场景冻结。损失为同F0残差归一化的RGB、σ1/2/4带符号HF和Sobel；没有模型、频率、学习率或权重搜索。下表为全标签拟合域σ1结果，完整三尺度/尾部/保留区域数据见JSON和CSV。

{table}
已确认只读贡献构成RGB范围限制：原对照高覆盖孔洞训练域内，We/alpha<.05为4511/41490与6422/86720像素；逐像素颜色区间给出RGB MSE数值下界约.002981/.002903。该界不等于HF上限或语义残影检测，不能解释几乎全部可写的阴影域。完整监督和只读贡献定位见[固定范围图](readonly_floor_audit_r1/README.md)。

全标签共享材质目标{field['final_postupdate_BUILD_DETAIL']:.8f}，独立有界颜色控制给出{bn['final_postupdate_BUILD_DETAIL']:.8f}的更低可行点。但训练FW比例仍{bn['extra']['relative_training_FW_gap']:.2%}，未达预登记1%门，没有严格负容量证书，不能声称3DGS全局表示上限。原留出对照与全标签归一化域不同，不直接比较其目标数值。理想12cm控制的raw86/110对比度为.898992/.908429；采样工程资格不能当全局误差界。

所有候选均保存并通过五视角公共加载器重载，RGB/alpha/K变化验证通过；全覆盖、无表面与静态遮挡分母保持。道路改善不意味着联合画面的旧静态残影已清除。当前优先问题是可写范围/表面支持与静态交接，以及仍不足的真实细节对应；没有切换3D生成先验的正面证据。当前候选不晋升，human_verdict/problem_solved=null，failure_ledger_refs=[V82-F06]，failure_ledger_delta=updated:V82-F06。

[交互报告](index.html) · [分阶段数据](evaluation/results.json) · [计算账目](resource_ledger.json) · [主合同](protocol.json) · [全标签合同](build_protocol.json)。原始packet、教师与中间checkpoint保留远端；必要资产与最终优化状态已打包取回，完整场景仍依赖原静态/动态/环境检查点。
'''
(O/'report.md').write_text(report,'utf-8')
print(json.dumps(dict(completed=True,rows=len(combined['rows']),BUILD_field_C_cos=[r['HF_cosine']for r in c],BUILD_field_S_energy=[r['HF_energy_ratio']for r in s])))

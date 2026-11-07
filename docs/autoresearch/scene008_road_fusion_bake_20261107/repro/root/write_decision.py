"""以固定分母和实际native读出整理分项结论；不填写人工质量验收。"""
from pathlib import Path
import json
RUN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_road_fusion_bake_20261107_r1')
def read(p):return json.loads(Path(p).read_text())
def fmt(x):return 'unsupported' if x is None else f'{x:.6f}'
data=read(RUN/'root/science_digest_v1.json');AB=read(RUN/'red_review/full49_review_summary_v1.json');C=read(RUN/'red_review/full49_edgeblend_summary_v1.json')
ledger=read(RUN/'engineer/pipeline_compute_ledger_ABC.json');base=read(RUN/'results/full49/receipt.json');edge=read(RUN/'results/edgeblend49/receipt.json')
assert ledger['total_native_forwards']==2590 and ledger['total_native_backwards']==2528
assert base['forwards']==294 and edge['forwards']==196
assert C['frozen_checks']['receipt_all5geometry_equal_records']==98
methods={'direct':'A 未融合atlas','fusion':'B Poisson atlas','edgeblend':'C 固定10px混合'}
primary=[['状态／指标','前轮AA','A 未融合','B Poisson','C 10px混合']]
for state in ('keep','delete'):
    rows={r['method']:r for r in data['raw128_primary'] if r['state']==state}
    primary.append([state+' / RGB MAE',fmt(rows['direct']['road_MAE_baseline']),*[fmt(rows[m]['road_MAE']) for m in methods]])
    primary.append([state+' / near-x残差跳变',fmt(rows['direct']['near_baseline']),*[fmt(rows[m]['near']) for m in methods]])
seams=[['状态／边界','支持帧／49','配对bins','前轮AA','A 未融合','B Poisson','C 10px混合']]
for state in ('keep','delete'):
    for name in ('near_x','lateral_y','far_x'):
        rows={r['method']:r for r in data['all49seams'] if r['state']==state and r['edge']==name}
        r=rows['edgeblend'];seams.append([state+' / '+name,r['supported_views'],r['paired_bins'],fmt(r['baseline_equal_bin_mean']),*[fmt(rows[m]['candidate_equal_bin_mean']) for m in methods]])
protection=[['状态／原观测域','像素帧数','AA RGB MAE','C RGB MAE','C较差帧／49','最差单帧均值Δ']]
for r in C['all49_original_domain_metrics']:
    if r['domain'] in ('original_road','protected','a6_visible','a37_visible'):
        protection.append([r['state']+' / '+r['domain'],r['count'],fmt(r['baseline']),fmt(r['candidate']),r['worse_frames'],f"{r['delta_distribution']['max']:+.8f}"])
distribution=[['状态／边界','C改善bins','C回退bins','相等bins','总配对bins']]
for r in C['all49_world_seam']:
    if r['paired_bins']:distribution.append([r['state']+' / '+r['edge'],r['improved_bins'],r['worse_bins'],r['equal_bins'],r['paired_bins']])
compute=[['阶段','native前向','native反向','optimizer steps'],['固定mask贡献资格',80,80,0],['真实生命周期smoke／下一步重放',80,48,24],['A/B/C各400正式',2400,2400,1200],['两轮49帧双状态配对评价',490,0,0],['三份资产独立5帧×双状态重放',30,0,0],['合计',3080,2528,1224]]
protocol=[['组件','本轮实现／范围'],['道路源','190个非HELD FRONT原始道路见证；几何固定，不补新深度'],['共享chart','13358节点；103可靠外halo边界；同帧见证梯度；固定λ=.04一次Poisson'],['正式参数','2053个可监督父格RGB（6159标量）；8045父格保持初始字节；40392子格共享父色'],['正式预算','每组fresh相同基线；400 paired steps／10epochs／Adam .01；颜色[0,1]'],['C边缘权重','w=clip(EDT(M)/10,0,1)，C=O+w(B−O)；BUILD w均值.186558，没有w=1像素'],['C信息边界','真实可见RGB主导的启发式混合；完整C目标依视角有效mask，非纯worldatlas；推理不condition queryO'],['编辑一致性','同一三维资产始终启用；删除只移除a37；统一原生Gaussian排序'],['新旧层','没有旧静态剪枝、opacity退役或legacy RGB优化；未声称无关层已经全部接管']]
limitations=[
    'C有道路颜色与横向边界的实际改善，近端边界仍回退；原V44与前轮AA入口保留，未替用户接受整场景画质。far边界缺配对分母，不能记成0。',
    '横向均值改善不代表逐bin改善：DELETE lateral 147/277改善、107回退、23相等；near仍有107回退、77改善、64相等。',
    'C额外使用已有真实可见RGB做强边缘锚定；atlas平均权重约.1866，没有纯atlas内域。收益不能单独归因Poisson、预训练道路先验或未知纹理恢复。',
    'C的HELD9目标本身读入该query真实RGB，仅用于构造诊断；3D材质只在BUILD40拟合，native结果不condition queryRGB。既有场景此前已看过该clip，所有结果仍属同场景开发证据。',
    'raw128原蓝框1385道路像素包含545atlas替换＋840原RGB恒等目标，合法fit1071；near固定6bins配对35/33像素，目标含1个纯恒等bin与5个混合bin。该分母与全49世界patch分表。',
    'C车外保护均值改善，但KEEP7帧、DELETE4帧有回退，最差单帧均值Δ约+4.42e-5/+2.81e-5；不作逐像素无损保证。',
    '三组共294个材料状态的alpha、深度、group/newroad/oldroad贡献与各自AA重放逐值一致；车辆预乘颜色保持。固定71条红侧射线W变化为0，原6/7条严重chart-first缺失仍在，隐藏轮胎真值未知。',
    '原始A/B目标在有效性边界硬切，导致颜色均值下降但接缝回退；源atlas Poisson未自动解决投影mask衔接。C是看到此负证据后登记的唯一10px控制，未尝试其它带宽。',
    '一次Poisson解的60节点/131通道出现负值裁剪，最大方程残差由约2e-15变为.008007，原始解和裁剪收据保留；没有重调解算参数。',
    '完整checkpoint在恢复起点逐位一致；续步参数也一致，但少量Adam moment有约1e-14/1e-22量级原子累加差异，未冒称续步moment逐位相同。',
    '本轮仍是观测支持道路融合；学习型道路生成、严格空间盲化先验实验、R4决策泛化与MOG-2不透明首交渲染均未在本轮验证。']
status='''Scene008已完成[道路融合目标→持久三维写回](autoresearch/scene008_road_fusion_bake_20261107/index.html)。保留C作为“已观测道路外观／横向边界改进候选”，近端仍开放；原V44及前轮AA保留，human_verdict=null，整场景未采用。

使用190个非HELD FRONT实测道路见证构成共享chart，固定道路几何与原生场景。A未融合atlas、B固定Poisson、C固定10px真实RGB边缘混合，三组均fresh基线400paired steps、同2053父格RGB/6159标量、同mask/Adam.01；8045无监督父格原色不变，40392子格继承同一父色。C是A/B固定49帧回退后，依据投影有效mask硬切的额外梯度证据登记的唯一fallback，未扩带宽/权重网格，未执行legacy静态资格或退役。

raw128 DELETE原蓝框1385道路像素RGB MAE .029598→C .015966；near6bins却.009794→.011725。全49横向277bins、40帧支持，DELETE .016715→.009164，KEEP .015085→.008788；近端248bins、47帧支持，DELETE .009340→.009739、KEEP .008399→.008709仍回退；far全部unsupported。C横向DELETE 147bins改善、107回退、23相等，不由平均数代替逐域验收。保护均值下降但KEEP7/DELETE4帧有小回退，最坏均值Δ+4.42e-5/+2.81e-5。

C主要是观测RGB锚定：BUILD atlas权重均值.1866且无w=1，C目标依视角有效mask；HELD目标本身含queryO仅作诊断，不叫预测质量。3D拟合只用BUILD40，native颜色不condition queryO；继承场景已预曝，非独立泛化。A/B负例与C强控制均保留，不单独归因Poisson或学习道路先验。

三组294个材料状态的几何/alpha/分组贡献与AA逐值一致，车体预乘颜色保持；固定71红侧射线W不变，原6/7严重chart-first缺失未解决。三份完整40392行资产已导出，30次fresh独立重放的所有浮点字段/PNG逐字节一致；无编辑/删除共用同一资产，删除只关a37。

本轮3080F/2528B，含正式1200optimizer steps与单列24次smoke/续步工程更新；原native/车辆/几何/opacity/预训练网络更新0。完整恢复起点allstate exact；续步参数exact而少量Adam moment存在极小差异，实际收据保留。failure_ledger_refs=[V82-F13]，failure_ledger_delta=updated:V82-F13。R1盲化生成、完整MOG首交与R4泛化未新增验证。

下一步锁定近端边界的分层残差与跨帧对应，核实同一实测表面在新旧层间的材质衔接，再建立保护／接管账本。本轮不继续调带宽或扩大旧高斯关闭范围。源码、状态、资产、报告已归档取回核验；子代理仅sol6.1/max，AutoDL按用户要求保持开机。'''
decision=dict(schema='SEPARATE_ROAD_FUSION_BAKE_OUTCOMES_V1',selected_component='C_10PX_OBSERVED_RGB_ANCHORED_ROAD_APPEARANCE_CANDIDATE',
    whole_scene_adopted=False,human_verdict=None,
    outcomes=[dict(title='蓝圈：颜色与横向衔接改善，近端仍开放',body='C的raw128/delete道路RGB MAE 0.029598→0.015966；全49横向277bins均值0.016715→0.009164。近端248bins仍0.009340→0.009739，原框6bins也回退，未宣称全部接缝解决。'),
              dict(title='车体与车轮：几何遮挡保持，剩余缺失未修',body='三组材料状态的深度、alpha、分组贡献均与冻结AA逐值一致；固定71条车轮诊断射线W变化为0。C没有新增压轮胎，也没有修好原6/7条严重缺失。车辆资产与轨迹未更新。')],
    tables=[dict(title='raw128原蓝框：颜色与近端接缝分开看',rows=primary,note='同原1385道路像素与固定near-x6bins（配对35/33像素），encoded RGB[0,1]；不同于下表世界patch。候选推理不读这些目标图。'),
            dict(title='全49世界patch：固定边界、双状态',rows=seams,note='等配对bin均值；near47帧/248bins，lateral40帧/277bins；unsupported保留空值。A/B/C同40帧拟合，同190源；C额外用合法真实RGB锚定边缘。'),
            dict(title='C逐bin正反例数量',rows=distribution,note='不存在一个覆盖所有区域的总通过标签。完整分帧与最坏bin见独立review。'),
            dict(title='原观测域与车外保护',rows=protection,note='同一固定像素帧分母；删除后隐藏真值不在此。保护均值改善与逐帧回退并列。'),
            dict(title='方法与信息角色',rows=protocol,note='A/B只改变atlas融合算子；C为看到硬切负证据后登记的唯一观测主导边缘控制，增量归因分开。'),
            dict(title='实际计算账本',rows=compute,note='反向包含资格贡献读出；正式更新只到道路颜色。时间为worker窗口／渲染循环，非纯CUDA-event时长；所有smoke与必要恢复单列。')],
    limitations=limitations,downloads=[
        dict(label='C完整40392行道路资产与manifest',href='repro/engineer/canonical_edgeblend_v1/scene_manifest.json'),
        dict(label='C实际道路资产NPZ',href='repro/engineer/canonical_edgeblend_v1/new_road_canonical_candidate.npz'),
        dict(label='A/B全49分层独立review',href='evidence/red_review_full49_three_layer_review_v2.json'),
        dict(label='C全49分层独立review',href='evidence/red_review_full49_edgeblend_review_v1.json'),
        dict(label='目标硬切诊断',href='repro/targets/hardreplace_edge_audit_v1.json'),
        dict(label='后验单一10px控制合同',href='repro/contract/edgeblend_control_v1.json'),
        dict(label='原始输入／目标／完整状态重放说明',href='repro/README.md')],
    computation=dict(native_forwards=3080,native_backwards=2528,formal_optimizer_steps=1200,engineering_optimizer_steps=24,
        original_parameter_updates=0,geometry_updates=0,opacity_updates=0,pretrained_network_updates=0,
        engineering_ledger='engineer/pipeline_compute_ledger_ABC.json',root_evaluation_forwards=490),
    research_status=status,
    experiment_entry='- `scene008_road_fusion_bake_20261107_r1`：[共享道路atlas、融合目标、10px强观测锚定与持久3D写回](autoresearch/scene008_road_fusion_bake_20261107/index.html)。A/B接缝负例保留；C改善颜色与横向边界，近端仍回退。三组同预算、完整49帧分项评价、30次精确重放及全部计算/恢复账本见报告。failure_ledger_refs=[V82-F13]，failure_ledger_delta=updated:V82-F13。\n',
    failure_card='''共享atlas Poisson与固定预算材质回写让raw128颜色均值下降，却使固定近端与全49横向接缝更差。独立目标诊断显示投影有效mask的硬切会重新引入边缘；不能把世界chart连续当成目标图或最终渲染连续。

随后唯一固定10px观测RGB边缘控制C取得实际成像改善：raw128/delete原1385道路像素RGB .029598→.015966，全49/delete横向277bins .016715→.009164；近端248bins .009340→.009739仍回退，原框6bins .009794→.011725。C平均atlas权重仅.1866、无纯atlas内域，是观测主导的融合强控制，不能算学习型道路先验或Poisson独立成功。保护均值改善仍有逐帧负例。

道路几何/alpha/AA/车辆不变，固定71射线的车辆贡献变化0，原严重车地读出缺失保持。三份资产各10次重放全字段/PNG字节相同；近端新旧层兼容性仍开放，旧静态没有被盲剪或重新调色。见[完整分项证据](../../autoresearch/scene008_road_fusion_bake_20261107/index.html)；不更新原RCR-1限定去蓝采用决定。''',
    failure_ledger_refs=['V82-F13'],failure_ledger_delta='updated:V82-F13',AutoDL='KEEP_ON')
(RUN/'round_decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(result=decision['selected_component'],forwards=3080,backwards=2528,whole_scene_adopted=False),ensure_ascii=False))

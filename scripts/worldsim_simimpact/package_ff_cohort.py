"""同步六日志有限筛查、完整分母与F21，图册不删除前轮正反结果。"""
import json,shutil,tarfile,re
from pathlib import Path
W=Path(__file__).resolve().parent;S=W/'repo_stage';E=W/'evidence_ff_cohort_final';E.mkdir(exist_ok=True);O=W.parents[1]/'outputs/Simulation_Impact_Research';paths=[]
with tarfile.open(W/'ff_cohort_evidence_final.tar.gz') as t:t.extractall(E,filter='data')
with tarfile.open(W/'milestone8_ledgers_before.tar') as t:t.extractall(S,filter='data')
note='''# 当前：六日志原生感知矩阵完成，预选前车47/48保留完整匹配（2026-09-15）

WS-SIM-FF-COHORT-PERCEPTION-01 / 20260915-r1：82新增检测＋20复用完成102项。52目标，真实中心52/52、IoU40/52；四模型六/十二图已齐。补回原缺失后，48个元数据前车条件全部中心匹配、47个IoU匹配；没有原可靠目标在两固定置信度和两预算下持续丢失中心匹配。两新候选是0028右侧约15m行人，置信度0.74–0.81、中心误差0.34–0.48m，只有IoU/定位下降，未证明驾驶危害。本轮不做其局部网格扫参/训练。Ω十二图明显改善，正例和新增匹配完整保留。

累计222 CenterPoint；72几何前向、802策略/PDM、18 HUGSIM＋4 SplatAD闭环不变。六日志全部已曝光，每日志一时刻，当前扫描＋真实强度/九帧历史的额外信息控制，非完整传感器或独立确认。有限筛查收口；整体goal active、人工verdict=null；failure_ledger_delta=V74-H2-F21。下一次有意义推进须补独立来源与实际规划接口，不重跑这批或把低杠杆框误差继续细调成Hero。见docs/WORLDSIM_SIMULATION_SIX_LOG_PERCEPTION.md。以下为历史。

'''
# 历史打包器不再发布全局状态；结果保存在本轮报告/证据中。
rel='docs/research_failures/entries/V74-H2-F21.md'
(S/rel).write_text('''# V74-H2-F21：跨日志框精度下降不等于前车严重丢失或驾驶仿真危害

- task/run：WS-SIM-FF-COHORT-PERCEPTION-01 / 20260915-r1。六已曝光日志0004/0028/0045/0061/0073/0094，每日志一时刻；四日志首次做CenterPoint评价，但不是独立测试。
- 执行：补提36历史文件；82新增检测＋20复用=102矩阵项。四官方模型、六/十二图、full/fill_missing＋real；没有重跑几何模型、训练或新闭环。累计222 CenterPoint、72几何、802策略/PDM、18 HUGSIM＋4 SplatAD反馈闭环。
- 信息：已知标定、BUILD LiDAR尺度、当前真实强度、相同九帧真实历史为额外输入；只替换当前重建扫描。fill_missing另外恢复原来缺失回波。十二图用下一关键帧RGB，非在线当前帧因果输入。
- 完整分母：52可评价目标，真实输入两个置信度下中心52、IoU40。40个同时通过二者构成可靠基线子集，其他12保留完整分母；不能归为重建新增失败。
- 强正例：48个补回缺失的预选前车条件全部中心匹配，47保留IoU。0028 VGGT十二图唯一IoU失配未在六图重复。原可靠目标在score0.3/0.5与两个视图预算下均持续中心失配的数量为0。
- 残余：24模型－目标组合持续IoU失配，分布四日志，不等于24独立坏例。补回缺失后score0.5的IoU总数：VGGT32/31、Ω29/38、DVGT37/35、Pi3X28/27，真实40。每方法均有新增匹配；不忽略改善。
- 冻结新候选：按元数据前车、中心失配、IoU失配、scene/instance顺序选最多2新目标，最终0028两行人(Pi3X、DVGT)。真实当前框内11/8点、中部8/7；最近同类中心误差0.177→0.405/0.475m、0.033→0.341/0.357m，原检测均仍高置信度存在。IoU对小目标与对应关系敏感，最近同类框非核实轨迹身份。
- 停止判定：两目标横向约15m，不存在已验证的驾驶决策退化；0028既有真实策略基线亦有名义交叠。保留感知定位候选，不继续ROI/阈值/修复扫参或训练，不将其晋级严重几何因果Hero。
- 接口：当前TransFuser不读取CenterPoint。WorldEngine PDM读取目标但依赖nuPlan地图路线；BridgeSim/PKL只完成一手接口核查，未新增执行。PKL不是闭环或2026 SOTA。
- 结论范围：此前Ω局部几何因果作用保留，但没有独立重复或生成必要性；当前不支持普遍phantom→严重危害。有限批次关闭，整体goal active；人工verdict=null。

[报告与组件图](../../WORLDSIM_SIMULATION_SIX_LOG_PERCEPTION.md)。
''',encoding='utf-8');paths.append(rel)
base='docs/autoresearch/worldsim_simimpact/milestone8'
for p in E.rglob('*'):
    if not p.is_file():continue
    rel=base+'/evidence/'+p.relative_to(E).as_posix();dest=S/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);paths.append(rel)
payload={k:json.loads((E/f'{k}.json').read_text()) for k in ['decision','aggregate','scene_counts','candidates']}
payload['candidate_support']=json.loads((E/'candidate_support/support.json').read_text())
(O/'SIX_LOG_PERCEPTION_EVIDENCE.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
rel=base+'/evidence/report_evidence.json';(S/rel).write_text(json.dumps(payload,indent=2),encoding='utf-8');paths.append(rel)
shutil.copy2(W/'planner_interface_review.md',O/'PLANNER_INTERFACE_REVIEW.md')
rel='docs/WORLDSIM_SIMULATION_PLANNER_INTERFACE_REVIEW.md';shutil.copy2(W/'planner_interface_review.md',S/rel);paths.append(rel)
report=(O/'SIX_LOG_PERCEPTION_CLOSEOUT.md').read_text(encoding='utf-8')
for stem in ['F19_Six_Log_Perception_Matrix','F20_Candidate_Context_And_Limits']:
    for ext in ['png','pdf','svg']:report=report.replace(f'({stem}.{ext})',f'({base.removeprefix("docs/")}/figures/{stem}.svg)')
    rel=base+'/figures/'+stem+'.svg';(S/rel).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(O/(stem+'.svg'),S/rel);paths.append(rel)
report=report.replace('(SIX_LOG_PERCEPTION_EVIDENCE.json)','('+base.removeprefix('docs/')+'/evidence/report_evidence.json)').replace('(PLANNER_INTERFACE_REVIEW.md)','(WORLDSIM_SIMULATION_PLANNER_INTERFACE_REVIEW.md)').replace('(LOCAL_ASSET_CAUSAL_AUDIT.md)','(WORLDSIM_SIMULATION_LOCAL_ASSET_CAUSAL_AUDIT.md)')
report=report[:report.rfind('[完整数值]')]+f'[完整数值]({base.removeprefix("docs/")}/evidence/report_evidence.json) · [矩阵 SVG]({base.removeprefix("docs/")}/figures/F19_Six_Log_Perception_Matrix.svg) · [候选图 SVG]({base.removeprefix("docs/")}/figures/F20_Candidate_Context_And_Limits.svg)\n'
rel='docs/WORLDSIM_SIMULATION_SIX_LOG_PERCEPTION.md';(S/rel).write_text(report,encoding='utf-8');paths.append(rel)
for name in ['prepare_ff_cohort_perception.py','analyze_ff_cohort_perception.py','audit_ff_cohort_candidates.py','draw_ff_cohort_perception.py','draw_ff_cohort_candidates.py','finalize_ff_cohort_perception.py','package_ff_cohort.py']:
    rel='scripts/worldsim_simimpact/'+name;shutil.copy2(W/name,S/rel);paths.append(rel)
shutil.copy2(O/'SIX_LOG_PERCEPTION_CLOSEOUT.md',O/'STATUS.md')
p=O/'index.html';html=p.read_text(encoding='utf-8')
html=re.sub(r'<section>.*?</section>',lambda m:'' if any(x in m.group() for x in ['F19_Six_Log_Perception_Matrix','F20_Candidate_Context_And_Limits']) else m.group(),html,flags=re.S)
start=html.index('<main>');section=html.index('<section>',start)
header='''<main><span class="status">六日志矩阵完成 · 保留良好案例 · 未获得严重因果坏例</span><h1>重建误差怎样影响仿真：跨日志下游复核</h1><p>四官方模型、两种视图预算的原生感知矩阵已补齐：82次新增＋20次复用。52目标真实中心全部匹配；48个预选前车条件全部保留中心匹配、47个保留IoU。新筛出的路侧行人仍被高置信度检测，框精度下降尚无驾驶危害证据。</p><p>此前Ω局部网格修改能恢复车辆定位，但普通删除也有效；这一有限因果结果保留。不能据此主张所有SOTA普遍产生phantom并导致严重事故。累计222次CenterPoint；本轮没有新增训练、几何推理或闭环。</p><p><a href="SIX_LOG_PERCEPTION_CLOSEOUT.md">最新：六日志复核</a> · <a href="SIX_LOG_PERCEPTION_EVIDENCE.json">完整数值</a> · <a href="PLANNER_INTERFACE_REVIEW.md">规划接口边界</a> · <a href="LOCAL_ASSET_CAUSAL_AUDIT.md">局部资产因果证据</a></p><section><h2>四模型 × 六日志：完整分母与前车正例</h2><a href="F19_Six_Log_Perception_Matrix.png"><img src="F19_Six_Log_Perception_Matrix.png" alt="完整六日志冻结感知矩阵，含覆盖控制与前车正例"></a><p class="small"><a href="F19_Six_Log_Perception_Matrix.pdf">PDF</a> · <a href="F19_Six_Log_Perception_Matrix.svg">SVG</a></p></section><section><h2>把几何/检测局部误差放回真实驾驶场景</h2><p>两个冻结新候选都在侧方约15m。高置信度框的位置偏差可见，但不把小目标IoU失配改写成漏检或已验证的驾驶危害。</p><a href="F20_Candidate_Context_And_Limits.png"><img src="F20_Candidate_Context_And_Limits.png" alt="路侧行人真实RGB、全局BEV与检测框"></a><p class="small"><a href="F20_Candidate_Context_And_Limits.pdf">PDF</a> · <a href="F20_Candidate_Context_And_Limits.svg">SVG</a></p></section>'''
p.write_text(html[:start]+header+html[section:],encoding='utf-8')
for rel in paths:
    p=S/rel
    if p.suffix in ['.json','.md','.py','.svg']:p.write_text('\n'.join(line.rstrip() for line in p.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
with tarfile.open(W/'ff_cohort_milestone8.tar','w') as t:
    for rel in paths:t.add(S/rel,arcname=rel)
(W/'milestone8_paths.json').write_text(json.dumps(paths,indent=2));print('PACKAGED',len(paths),'files',sum((S/p).stat().st_size for p in paths),'bytes')

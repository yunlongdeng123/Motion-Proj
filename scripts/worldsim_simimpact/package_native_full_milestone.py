"""同步完整预算结果、F19 与可复审图册；只打包本轮修改。"""
import json,shutil,tarfile
from pathlib import Path
W=Path(__file__).resolve().parent;S=W/'repo_stage';O=W.parents[1]/'outputs/Simulation_Impact_Research';E=W/'evidence_milestone6'
E.mkdir(exist_ok=True)
with tarfile.open(W/'milestone6_evidence.tar.gz') as f:f.extractall(E,filter='data')
with tarfile.open(W/'milestone6_ledgers_before.tar') as f:f.extractall(S,filter='data')
shutil.copy2(W/'evidence_native_full/aggregate.json',O/'FULL_BUDGET_EVIDENCE.json')
shutil.copy2(O/'FULL_BUDGET_AND_PERCEPTION.md',O/'STATUS.md')
note='''# 当前：完整预算闭环与原生检测完成，局部强度恢复解释主要漏检（2026-09-15）

SplatAD 30001步完成；194 LiDAR／678相机留出、64策略/PDM与2次反馈闭环完成。LiDAR-only平均ADE差−0.014/+0.026m，两闭环无框交叠。冻结nuScenes CenterPoint真实机动车32/32，原生raw/median16/20；工程车真实8/8、原生0/1，局部真实扫描恢复8/8，而median仅恢复强度即7/8、位置恢复仅2/3。保留传感器感知坏例，不晋级纯几何或phantom Hero。四前馈模型两日志18次检测包含DVGT正例，不能主张普遍同样失效。全部CenterPoint130次，累计18 HUGSIM＋4 SplatAD反馈闭环、72几何前向、802策略/PDM。见docs/WORLDSIM_SIMULATION_FULL_BUDGET_PERCEPTION.md及F19。

本轮有限工程车诊断收口，不细切ROI或追末帧残余；不启动第二个SplatAD拟合。后续仅复核已有Ω车辆／Pi3X行人检测候选的真实支持，再判断实际局部资产修复是否必要。尚无前馈几何资产修复后的独立下游恢复。goal active；人工verdict=null；failure_ledger_delta=V74-H2-F19。以下为历史。

'''
paths=[]
for rel in ['AGENTS.md','docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md','docs/RESEARCH_FAILURES.md']:
    p=S/rel;p.write_text(note+p.read_text(encoding='utf-8'),encoding='utf-8',newline='\n');paths.append(rel)
rel='docs/research_failures/entries/V74-H2-F19.md';p=S/rel;p.parent.mkdir(parents=True,exist_ok=True)
p.write_text('''# V74-H2-F19：稳定仿真感知退化也不能自动归为几何，强度恢复提供替代解释

- task/run：WS-SIM-NATIVE-CLOSEDLOOP-01 / scene0004-step030000-full1；WS-SIM-NATIVE-DETECTOR-01、LOCAL-01及WS-SIM-FF-PERCEPTION-01 / 20260915-r1。已曝光nuScenes训练日志，仅发现。
- 完整参照：官方SplatAD 30001步、500万GS；194帧LiDAR／678相机完整留出评价。额外完整RGB+LiDAR、标定和演员轨迹，非前馈同预算排名。
- 反馈结果：64策略/PDM，两种读出各4s新位姿传感器闭环；ADE1.369/1.423m，间距+0.565/+0.570m，无标注框交叠。日志传感器时刻匹配后，LiDAR-only平均ADE差−0.014/+0.026m，RGB-only+0.265m；不能把全部轨迹差归为几何。
- 原生检测：冻结OpenMMLab CenterPoint十帧LiDAR，真实机动车32/32（5实例），raw16/32、median20/32；真实射线排列控制17/32、21/32。统一强度也损伤真实基线，不能作为几何充分证明。
- 重复候选：同一工程车真实8/8、raw0/8、median1/8；局部GT框+0.25m真实扫描替换都8/8。只位置恢复2/8、3/8；只强度恢复raw0/8、median7/8。固定八时刻、两个读出、三种局部操作，无ROI扫参。
- 关闭的解释：把完整局部传感器替换后的恢复直接解释成几何资产修复；把图像外观可用、点数量接近真实当成下游可靠保证；把某个读出的漏检当成普遍phantom危害。
- 信息边界：三种恢复是传感器数组操作；完整替换同时改变几何、强度、点数和顺序。位置吸附为近邻操作，可能多对一，不能证明所有几何修复无效。未改重建资产，未测规划/安全恢复。
- 四方法对照：两日志×真实及四模型full/fill_missing共18次检测，固定九帧真实历史及真实当前强度。DVGT0004 IoU匹配14/19高于真实13/19；其他行有框误差/少量漏检，不能概括普遍失败。Ω车辆/Pi3X行人仍是单帧候选，不是独立确认。
- 决策：有限工程车审计收口，不细切最后残余，不启动第二个SplatAD场景拟合。后续只复核具体前馈下游候选，再考虑局部资产修复；不预设first-return。累计130 CenterPoint，18 HUGSIM+4 SplatAD反馈闭环、72几何前向、802策略/PDM。goal active；人工verdict=null。

[报告与组件图](../../WORLDSIM_SIMULATION_FULL_BUDGET_PERCEPTION.md)。
''',encoding='utf-8');paths.append(rel)
rel='docs/WORLDSIM_SIMULATION_FULL_BUDGET_PERCEPTION.md';report=(O/'FULL_BUDGET_AND_PERCEPTION.md').read_text(encoding='utf-8')
figs=['F14_Full_Budget_Native_Loop','F15_Excavator_Perception_Candidate','F16_Feedforward_Perception_Control','F17_Local_Sensor_Recovery']
for name in figs:report=report.replace(name+'.png','autoresearch/worldsim_simimpact/milestone6/figures/'+name+'.svg')
report=report.replace('(FULL_BUDGET_EVIDENCE.json)','(autoresearch/worldsim_simimpact/milestone6/aggregate.json)').replace('；[图册](index.html)','')
(S/rel).write_text(report,encoding='utf-8');paths.append(rel)
base='docs/autoresearch/worldsim_simimpact/milestone6'
for source in E.rglob('*'):
    if not source.is_file():continue
    relative=source.relative_to(E)
    # 完整逐帧数据由仓库外封包保留；仓库保留注册、源代码和必要审计。
    if source.name=='summary.json' and relative.parts[0] in ['fixed_scans','logged_scans']:continue
    rel=base+'/evidence/'+relative.as_posix();dest=S/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);paths.append(rel)
for source,name in [(W/'evidence_native_full/aggregate.json','aggregate.json'),(W/'evidence_native_objects/instance_summary.json','instance_summary.json')]:
    rel=base+'/'+name;shutil.copy2(source,S/rel);paths.append(rel)
for name in figs:
    rel=base+'/figures/'+name+'.svg';p=S/rel;p.parent.mkdir(exist_ok=True);shutil.copy2(O/(name+'.svg'),p);paths.append(rel)
scripts=['native_splatad_bridge.py','export_native_detector_scans.py','probe_native_centerpoint.py','audit_native_detector_inputs.py','audit_native_detector_objects.py',
         'prepare_ff_detector_probe.py','prepare_native_local_sensor_audit.py','export_native_full_milestone.py','summarize_native_full.py',
         'draw_native_full_loop.py','draw_native_perception.py','draw_native_local_recovery.py','package_native_full_milestone.py']
for name in scripts:
    rel='scripts/worldsim_simimpact/'+name;shutil.copy2(W/name,S/rel);paths.append(rel)
rel='docs/autoresearch/worldsim_simimpact/milestone5/full_evaluation_registration.json';reg=json.loads((S/rel).read_text())
reg.update(status='completed',new_execution_count=64,completed_run='scene0004-step030000-full1',failure_ledger_delta='V74-H2-F19',result_reference='../milestone6/aggregate.json')
(S/rel).write_text(json.dumps(reg,indent=2)+'\n',encoding='utf-8');paths.append(rel)
# 图册顶部表示当前结果，历史条目保留原来的实验预算。
p=O/'index.html';html=p.read_text(encoding='utf-8');start=html.index('<main>');section=html.index('<section>',start)
header='''<main><span class="status">完整预算与局部恢复完成 · 几何因果仍未确认</span><h1>哪些重建错误真的降低仿真价值？</h1><p>找到重复的工程车漏检：真实8/8、SplatAD raw0/8、median1/8；局部真实扫描恢复后都是8/8，但median仅恢复强度也达到7/8。它是感知坏例，尚不能当作纯几何或phantom方法动机。完整预算反馈闭环无新增框交叠；四前馈模型对照保留DVGT改善的反例。</p><p>累计18次HUGSIM＋4次SplatAD反馈闭环、72次几何前向、802次策略/PDM，另有130次CenterPoint。</p><p><a href="FULL_BUDGET_AND_PERCEPTION.md">最新完整报告</a> · <a href="FULL_BUDGET_EVIDENCE.json">数值汇总</a> · <a href="DOWNSTREAM_FIRST_PROTOCOL.md">下游优先判据</a> · <a href="NATIVE_CLOSED_LOOP_PILOT.md">8k试跑历史</a></p>'''
descriptions=[('完整预算：RGB＋LiDAR实际反馈闭环','LiDAR单独替换影响仍小，完整输入偏差主要经RGB通道；两次4秒闭环无标注框交叠。'),
              ('完整场景中的稳定感知坏例','黄色框来自真实对象标注。挖掘机在重建LiDAR下重复漏检；RGB只作场景定位，检测器消费LiDAR。'),
              ('四前馈模型的两日志有限对照','固定九帧真实历史，只替换当前扫描；DVGT在0004改善，其他方法存在框匹配下降。非完整仿真排名。'),
              ('局部恢复没有自动证明几何因果','局部真实扫描恢复8/8；median只恢复强度达到7/8，位置恢复仅3/8。停止继续细切本候选。')]
new=''
for name,(title,body) in zip(figs,descriptions):new+=f'<section><h2>{title}</h2><p>{body}</p><a href="{name}.png"><img src="{name}.png" alt="{title}"></a><p class="small"><a href="{name}.pdf">PDF</a> · <a href="{name}.svg">可编辑SVG</a></p></section>'
html=html[:start]+header+new+html[section:]
html=html.replace('实际新位姿反馈：主要差距目前经 RGB 通道','历史8k试跑：实际新位姿反馈').replace('尚未确认局部几何因果，完整拟合仍在进行。','历史试跑结论；完整预算结果见上方。').replace('官方SplatAD相机与LiDAR场景正在拟合。此组件图表示执行路径，尚不代表新位姿闭环已经实测完成。','历史接入组件图；完整拟合与反馈结果见上方。')
html=html.replace('单RTX3090 · 无训练 · 未读旧AV2封存集质量 · 未关机 · 人工verdict留空','单RTX3090 · 已完成授权的SplatAD场景拟合 · 无新Hero训练 · 人工verdict留空')
p.write_text(html,encoding='utf-8')
for rel in paths:
    p=S/rel
    if p.suffix in ['.md','.py','.json','.svg']:
        text=p.read_text(encoding='utf-8');p.write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n',encoding='utf-8',newline='\n')
(W/'milestone6_paths.json').write_text(json.dumps(paths,indent=2))
with tarfile.open(W/'native_full_milestone6.tar','w') as f:
    for rel in paths:f.add(S/rel,arcname=rel)
print('PACKAGED',len(paths),'files')

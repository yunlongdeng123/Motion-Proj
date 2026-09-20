"""同步局部资产因果审计与F20；保留删除有效和Pi3X未恢复的结果。"""
import json,shutil,tarfile
from pathlib import Path
W=Path(__file__).resolve().parent;S=W/'repo_stage';D=W/'evidence_ff_local_asset_full';O=W.parents[1]/'outputs/Simulation_Impact_Research';paths=[]
with tarfile.open(W/'milestone7_ledgers_before.tar') as t:t.extractall(S,filter='data')
note='''# 当前：局部网格修改获得有限感知因果证据，普通删除同样有效（2026-09-15）

WS-SIM-FF-LOCAL-ASSET-01 / r2：两个冻结下游候选、两输入预算，10次CenterPoint。Ω车辆局部网格校正后重新投射完整扫描，中心误差2.073/1.954m→0.086/0.030m，IoU0.347/0.370→0.836/0.804；同面删除也恢复（0.089/0.086m）。支持当前固定真实历史适配器内的局部几何→传感器→感知作用，未证明生成式修复必要性。Pi3X行人距离改善但检测不稳定恢复，关闭本轮径向修复子命题。均未通过预注册超出删除控制的Hero准入。不能把2m中心匹配失配称作车辆完全消失，亦无独立日志/规划/安全证明。

全部CenterPoint累计140次；72几何前向、802策略/PDM、18 HUGSIM＋4 SplatAD闭环不变。见docs/WORLDSIM_SIMULATION_LOCAL_ASSET_CAUSAL_AUDIT.md和F20。两候选不继续扫参；下一步若推进应冻结新的时刻/日志及完整传感器信息范围，而非重复当前几何诊断。goal active；人工verdict=null；failure_ledger_delta=V74-H2-F20。以下为历史。

'''
# 历史打包器不再发布全局状态；结果保存在本轮报告/证据中。
rel='docs/research_failures/entries/V74-H2-F20.md';(S/rel).write_text('''# V74-H2-F20：局部几何能影响感知，但删除控制已恢复，距离改善也不保证检测改善

- task/run：WS-SIM-FF-LOCAL-ASSET-01 / local_asset_inputs_r2、local_asset_detection_r2。scene-0004已曝光train单帧，两输入预算非独立重复。
- 预先选择：CenterPoint score0.3下真实匹配、六图full/fill_missing均失配的Ω车辆和Pi3X行人；之后才检查几何。各有25/31真实回波。
- 实际资产：距目标原首交点0.35m内的顶点，按目标median(GT range-predicted range)做一次径向平移，保存完整网格并重新投射完整扫描；同一顶点集接触的面另做删除。原来缺失的束恢复固定，新缺失不恢复。
- Ω几何：六/十二图原目标MAE1.735/1.628m→0.116/0.092m；各改变23回波，其中1条在GT框外，无新增缺失。强度、时间、历史逐元素一致，原始射线顺序保留。
- Ω下游：原高置信度车辆中心误差2.073/1.954m，不是完全没有检测；校正0.086/0.030m，IoU0.836/0.804；同面删除0.089/0.086m，也恢复。因此支持当前适配器内局部几何的因果作用，不支持生成/补全的必要性。
- Pi3X：目标MAE0.642/0.673m→0.245/0.203m；六图没有恢复目标。十二图原中心误差0.838m、score0.642，校正0.314m但score0.328，IoU0.270，score0.5下丢失匹配。删除也未稳定改善。
- 关闭的路线：将几何指标下降等同于下游恢复；把中心2m阈值失配夸大为目标消失；以普通删除已解释的恢复直接立生成式Hero。Pi3X本次局部径向修复停止，不扫半径/阈值。
- 边界：当前帧强度和九帧真实历史、BUILD全局尺度均为额外信息；仅当前重建扫描被替换。局部oracle用GT框/距离，不是同预算方法。没有独立场景/时间、真实反馈规划或安全恢复；不支持普遍phantom/early危害。
- 执行：新增10检测，累计140 CenterPoint；既有72几何前向、802策略/PDM、18 HUGSIM+4 SplatAD闭环不变。首次准备点顺序在任何推理前修正，仅r2输入用于检测。goal active，人工verdict=null；Hero准入未通过。

[报告与组件图](../../WORLDSIM_SIMULATION_LOCAL_ASSET_CAUSAL_AUDIT.md)。
''',encoding='utf-8');paths.append(rel)
base='docs/autoresearch/worldsim_simimpact/milestone7'
for p in D.rglob('*'):
    if not p.is_file():continue
    rel=base+'/evidence/'+p.relative_to(D).as_posix();dest=S/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);paths.append(rel)
report=(O/'LOCAL_ASSET_CAUSAL_AUDIT.md').read_text(encoding='utf-8').replace('(F18_Local_Mesh_To_Perception.png)','(autoresearch/worldsim_simimpact/milestone7/figures/F18_Local_Mesh_To_Perception.svg)')
report=report[:report.rfind('[目标数值]')]+'[目标数值](autoresearch/worldsim_simimpact/milestone7/evidence/target_results.json) · [上一轮完整预算与传感器控制](WORLDSIM_SIMULATION_FULL_BUDGET_PERCEPTION.md)\n'
rel='docs/WORLDSIM_SIMULATION_LOCAL_ASSET_CAUSAL_AUDIT.md';(S/rel).write_text(report,encoding='utf-8');paths.append(rel)
rel=base+'/figures/F18_Local_Mesh_To_Perception.svg';(S/rel).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(O/'F18_Local_Mesh_To_Perception.svg',S/rel);paths.append(rel)
for name in ['audit_ff_detection_candidates.py','prepare_ff_local_asset_audit.py','export_ff_local_asset_context.py','summarize_ff_local_asset.py','audit_ff_local_asset_inputs.py','draw_ff_local_asset.py','package_ff_local_asset.py']:
    rel='scripts/worldsim_simimpact/'+name;shutil.copy2(W/name,S/rel);paths.append(rel)
shutil.copy2(D/'target_results.json',O/'LOCAL_ASSET_EVIDENCE.json');shutil.copy2(O/'LOCAL_ASSET_CAUSAL_AUDIT.md',O/'STATUS.md')
p=O/'index.html';html=p.read_text(encoding='utf-8');start=html.index('<main>');section=html.index('<section>',start)
header='''<main><span class="status">局部资产因果审计完成 · 删除控制同样有效</span><h1>重建几何如何改变仿真中的车辆定位？</h1><p>Ω车辆返回偏后约1.6–1.7m，冻结检测器以较高置信度把车放远约2m。修改局部网格并重投射后，中心误差降到3–9cm；同面删除也恢复。获得当前适配器内的局部几何因果证据，尚未证明生成式修复必要性、独立场景或闭环危害。</p><p>Pi3X行人的距离误差改善，却没有稳定恢复检测；SplatAD工程车的主要恢复可由强度解释。累计140次CenterPoint，其他执行计数不变。</p><p><a href="LOCAL_ASSET_CAUSAL_AUDIT.md">最新：局部网格与连续定位误差</a> · <a href="LOCAL_ASSET_EVIDENCE.json">目标数值</a> · <a href="FULL_BUDGET_AND_PERCEPTION.md">完整预算与传感器控制</a> · <a href="DOWNSTREAM_FIRST_PROTOCOL.md">研究判据</a></p><section><h2>局部几何 → 模拟LiDAR → 车辆定位</h2><p>真实场景、实际重投射回波和检测框；报告连续误差，避免把2m匹配阈值误写成目标完全消失。普通删除有效的结果同时展示。</p><a href="F18_Local_Mesh_To_Perception.png"><img src="F18_Local_Mesh_To_Perception.png" alt="局部资产变化与车辆定位恢复"></a><p class="small"><a href="F18_Local_Mesh_To_Perception.pdf">PDF</a> · <a href="F18_Local_Mesh_To_Perception.svg">SVG</a></p></section>'''
html=html[:start]+header+html[section:];p.write_text(html,encoding='utf-8')
for rel in paths:
    p=S/rel
    if p.suffix in ['.json','.md','.py','.svg']:
        p.write_text('\n'.join(line.rstrip() for line in p.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
p=O/'F18_Local_Mesh_To_Perception.svg';p.write_text('\n'.join(line.rstrip() for line in p.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
with tarfile.open(W/'ff_local_asset_milestone7.tar','w') as t:
    for rel in paths:t.add(S/rel,arcname=rel)
(W/'milestone7_paths.json').write_text(json.dumps(paths,indent=2));print('PACKAGED',len(paths),'files')

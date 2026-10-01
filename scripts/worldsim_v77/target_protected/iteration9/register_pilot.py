from pathlib import Path
import sys,shutil,json
sys.path.insert(0,str(Path(__file__).parent));from temporal_factory import T,read,dump
P=Path('/root/autodl-tmp/motion_proj_v77');O=T/'r10';R9=T/'r9';E=P/'docs/autoresearch/worldsim_v77/target_protected_20260929'
def copy(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True)
 if src.suffix=='.json':dump(dst,read(src))
 else:shutil.copy2(src,dst)
for n in ['plan.md','temporal_audit.json','real_temporal_audit.json','run.json','closeout.json','independent_data_reviews.json','label_fix.json']:
 if (R9/n).exists():copy(R9/n,E/'r9'/n)
for n in ['plan.md','run.json','admission_result.json','dataset_catalog.json','evaluation_plan.json']:copy(O/n,E/'r10'/n)
status=P/'docs/RESEARCH_STATUS.md';s=status.read_text();s=s.replace('当前63GB可用','开始时63GB可用');s+='\n\nr9实际22候选全220帧机器与三独立扫描检查，独立三关键帧QA18pass/2uncertain/2reject。P014人行区/P022双黄线拒绝；P002/P007扫过视觉证据不足不放行。4个蓝C活跃身份被旧绿色B说明误导，修正为只显示active protected实例后独立复核通过，原分与标签保留。r9覆盖缺额按预案训练0步，不再扫格。\n\n新登记`WS-V77-TARGET-PROTECTED-20260929/r10`有限过程pilot：50train/25scene/max3，20背景27单3密集；4扫过/3world、4显露、5静止A＋ego运动、37稳定控制。7旧合成＋4新过程（扫过仅1world）＋8真实DEV，四权重同输入。同80张量/160步/原loss，有限控制器只训练→评价→HTML，不自动追加/关机。密集训练仍仅1world；单秒过程/几何对应代理不能认证充分长时或精确纹理证据。参见[r10计划](autoresearch/worldsim_v77/target_protected_20260929/r10/plan.md)。\n';status.write_text(s)
idx=P/'docs/EXPERIMENTS.md';s=idx.read_text();s=s.replace('running：盘点＋有界数据工厂；同80张量160步候选控制，尚未训练','closed：实测时序缺口；22候选18pass，扫过覆盖缺额，r9训练0步')
tag='WS-V77-TARGET-PROTECTED-20260929/r10'
if tag not in s and tag.replace('/r10',' / r10') not in s:s=s.replace('|---|---|---|','|---|---|---|\n| '+tag.replace('/r10',' / r10')+' | frozen：50train/25scene，4扫过/3world；验证扫过1world，同80张量160步 | [计划](autoresearch/worldsim_v77/target_protected_20260929/r10/plan.md) |')
idx.write_text(s)
f=P/'docs/research_failures/entries/V77-F02.md';s=f.read_text();marker='## r9：时序过程覆盖的实际缺额'
if marker not in s:s+='''\n\n## r9：时序过程覆盖的实际缺额

`WS-V77-TARGET-PROTECTED-20260929/r9`实测r7训练68窗口/r8训练50例均0个B归一化遮挡中心跨度≥0.20的扫过例；r8已有12个世界静止A＋运动ego＋投影变化例，不能把该类说成完全缺失。原70审计13个该类，28个后车GT包络扫过代理；真实隐藏纹理证据UNKNOWN，A022固定窗ego静止但A移动，时序覆盖不能解释全部失败。

固定世界静止/固定世界速度轨迹替代逐帧随B固定偏移；一次旧锚点搜索＋唯一16-source扩展得到有限候选，未达8train/3val/2val-world扫过目标，r9按规则训练0步，不无限扫格。22候选实际全220帧X/Y/H与空间合同、三扫描静态占据复查；独立0/5/9实看18pass/2uncertain/2reject。P014人行区/P022跨双黄线拒绝，即使地图/GT/LiDAR无正证据也不放行；P002/P007数字扫过代理不足以视觉确认，不计合格。

旧标注以来源顺序标绿B/蓝C，实际active保护车有时为蓝C，造成4例语义疑点。修成只标active protected实例，独立再核对四例通过；训练Y/X/H/几何不改，原分与旧标签保留。QA标签不清是工程问题，不能据此否定合成输入或模型。

随后单独冻结r10有限过程pilot（非充分覆盖）：50train/25world，实际4扫过/3world、4显露、5静止A+ego、37稳定控制；新扫过验证仅1独立world。原架构80张量160步，不改loss/seed/模块。GT cuboid表面格只给近似几何时序证据，不把其当真实纹理已见；一秒、共享DEV形状、密集train1world和无真实隐藏GT边界保留。模型结果尚待新控制，原/r7/r8不覆盖。failure_ledger_delta: updated V77-F02（数据时序缺口、有限来源缺额、审核身份标注修正）；不新增ID。
''';f.write_text(s)
print('registered r10; r9 frozen no training')

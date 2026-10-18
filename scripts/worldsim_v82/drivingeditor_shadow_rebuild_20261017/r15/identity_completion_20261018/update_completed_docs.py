from pathlib import Path
import shutil,os,json
W=Path(__file__).parent;P=W.parent/'r15_closure_run/publish_repo';R=W.parent/'r15_repair_4k'
O=W.parents[1]/'outputs/r15_repair_4k_20261018'
D=Path('docs/autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15')
def long(p):return Path('\\\\?\\'+str(p.resolve())) if os.name=='nt' else p
def cp(src,dst):
    dst=long(dst);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(long(src),dst)
for f in O.iterdir():
    if f.is_file():cp(f,P/D/'repair_4k_20261018'/f.name)
cp(R/'build_report.py',P/'scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/repair_4k_20261018/build_report.py')
progress=json.loads((R/'live_progress.json').read_text(encoding='utf-8-sig'));step=progress['global_step']
status='''# 当前研究状态

更新：2026-10-18。当前 task=WS-V82-R15-REPAIR-4K-20261018，并交付 WS-V82-R15-IDENTITY-COMPARISON-20261018 已完成结果；failure_ledger_refs=[V82-F06]，failure_ledger_delta=none。无subagent、R16/R17；原45K暂停保留。

旧GN11 30K＋五实例30K已完成，a5/a26/a18/a6/a37都有真实独立owner，全49帧245身份帧的独立RGB、alpha、全景可见alpha与删除完成。历史旧3的147身份帧与原冻结R15的245身份帧也已完成，总637身份帧；本地全PNG矩阵及31视频交付，Git保存视频、固定锚帧和完整三值支撑证据。[旧基线完成对照与组件](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/identity_20261018/baseline_complete.html) · [旧5/原R15三维中心与真实支撑](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/identity_20261018/tristate/index.html)。

以同一真实车框、物理米坐标及1e-5m容差，旧5/原R15的框外中心：a6 3.73%/24.09%，a37 6.35%/17.53%，a5 1.97%/4.22%，a26 2.45%/6.85%，a18 34.97%/46.44%。旧5无增密且初始化不同；a6最大σ约3.29m，在raw105/113/121可见大面积下部支撑、亮白/蓝色条带。a37两版删除均残留车形；旧5不是干净资产真值，框外率低也不能证明资产干净。旧5没有三值证据/概率，绿色只表示硬owner。原始历史旧3检查点仍只有三个owner，不能把新5归属倒推到它。

R15修复真实GPU自检已通过：2步训练、实际阴影分块差0、严格恢复、46帧完整当前证据审计；smoke模型未用于正式初始化。控制器PID20154已自动开始从相同原测量/原三值mask重新训练4000步，最近读取STEP/4000，尚无修复后质量结论。[修复组件、合同与GPU记录](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/repair_4k_20261018/index.html)。

修复使用Actor不可重置来源锚点、累计0.25m、真实框+0.10m容许域（保留初始异常包络）、σ增长1.5倍、2%每次/4倍总增密预算；当前证据审计在1500/2500/3500，source-tag不挡候选，反证撤销确定状态，重复真实3D才改owner，真实道路/大支撑可否决整点转车。FP64因子阴影接入模型/恢复。原30K学习率、loss、SGCM、容量保留，外部池不重放旧459点补入。组合修复不能归因单项，需检查保体、a6碎片与a37外观残影。26项CPU针对测试通过；4项旧reference失败在原版相同复现，已披露。

下一步等待新4K三次归属审计和训练完成，执行五身份全49帧独立/删除评估，再交付新旧图、视频、三值与保体/残影。原GN11、旧3、旧5、原R15检查点和原始输入均保留，用户其他dirty不动。同一重建窗口，无独立编辑GT，human_verdict=null，problem_solved=null。

GPU仍用于新4K；旧两队列完成不是关机条件。新训练/评估、报告保存及v82推送全部完成，确认所有worker和会启动任务的controller结束后，按用户授权关闭AutoDL并记录SSH关闭观察。
'''.replace('STEP',str(step))
(P/'docs/RESEARCH_STATUS.md').write_text(status,encoding='utf-8')
p=P/'docs/EXPERIMENTS.md';s=p.read_text(encoding='utf-8')
old='历史旧3、R15冻结身份渲染完成；补齐旧5训练进行中；三值谱系不变与a6监督覆盖缺口，GPU支撑投影待队列'
new='旧5完整30K及五身份49帧完成，三版637身份帧/31视频；旧5与原R15三维中心、统一框统计、三值真实支撑投影完成'
s=s.replace(old,new)
s=s.replace('[三值交互、组件与证据](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/identity_20261018/tristate/index.html)、[执行范围]', '[旧5完整对照与视频](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/identity_20261018/baseline_complete.html)、[三值交互](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/identity_20261018/tristate/index.html)、[执行范围]')
p.write_text(s,encoding='utf-8')
p=P/'docs/research_failures/entries/V82-F06.md';s=p.read_text(encoding='utf-8')
if '旧五身份30K完成对照' not in s:s+='''

## 2026-10-18：旧五身份30K完成对照

task/run：`WS-V82-R15-IDENTITY-COMPARISON-20261018/owner5_train30k`。[组件、同一a6/a37锚帧与全49帧视频](../../autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/identity_20261018/baseline_complete.html)。旧5实际30000步，49×5真实owner独立/删除完成；原旧3、冻结原R15和旧5合计637身份帧，31视频。本地全PNG保留；GPU三值支撑投影490身份/阶段帧也已完成，参数更新0。

共同真实车框，旧归一化坐标按实际scale0.1还原米；1e-5m边界容差避免FP32舍入：旧5/原R15框外中心a6=3.73%/24.09%，a37=6.35%/17.53%，a5=1.97%/4.22%，a26=2.45%/6.85%，a18=34.97%/46.44%。旧固定拓扑、种子规模不同，不是配对单因果实验。旧5 a6最大主轴σ3.29m，同raw105/113/121独立图有明显下部大支撑和亮白/蓝色条带；a37旧5删除有明亮车形残留，原R15关阴影删除仍有暗车形。少量大尺度点说明较低框外中心比例不能证明干净资产。旧5是补齐归属的控制扩展，不把该owner属性倒推历史旧3；旧5无三值概率，viewer绿色只表示硬owner。

新R15真实GPU自检已通过并启动独立修复4K；此处比较冻结原R154K，不抢先宣布新修复质量通过。human_verdict和problem_solved仍null。
'''
p.write_text(s,encoding='utf-8')
print('current snapshot and completed baseline evidence updated')

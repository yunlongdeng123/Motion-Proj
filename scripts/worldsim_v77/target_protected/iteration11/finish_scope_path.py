"""保存正反证据与交付；合成误差下降不触发电源操作。"""
from pathlib import Path
import sys,shutil,json
sys.path.insert(0,str(Path(__file__).parent));from process_sources import T,read,dump
P=Path('/root/autodl-tmp/motion_proj_v77')

def main():
 for run in ['r14','r15']:
  b=T/run/'docs_before_closeout';b.mkdir(exist_ok=True)
  for rel in ['docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md','docs/research_failures/entries/V77-F02.md']:
   if not (b/Path(rel).name).exists():shutil.copy2(P/rel,b/Path(rel).name)
  assert read(T/run/'delivery_validation.json')['success']
 r14={'stage':'closed_no_real_cross_case_benefit','training_steps':160,'evaluation_windows':95,'fresh_windows':19,'exact_input_reused_windows':76,'same_160_sample_order_verified':True,'all_finite_gradients':True,'module_tensors':80,'module_parameters':49574080,'real_cross_case_benefit_demonstrated':False,'positive_evidence':'new process GT scene-equal H MAE -19.28%, B MAE -18.94% vs r10','negative_evidence':'fixed f5 real eight DEV retain regeneration/smear/protected damage; no video pass-rate claim','failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'updated V77-F02: equal-size temporal module finite GT gain did not transfer under current data/budget','human_verdict':None,'final_used':False,'shutdown_performed':False}
 r15=read(T/'r15/coverage_closeout.json')|{'stage':'closed_valid_new_data_but_validation_coverage_shortage','failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'updated V77-F02: path-relative operator produced six sweep train/4world, no sweep val; data valid, no training','shutdown_performed':False}
 dump(T/'r14/closeout.json',r14);dump(T/'r15/closeout.json',r15)
 reports={
 'r14':'''# r14：等参数量时间自注意力控制

真实 DELETE 尚无跨例稳定收益，本轮不推广、不关机。时间更新改善了有限的新过程GT，但不能替代目标删净与真实保护对象保持。

```mermaid
flowchart LR
 D[同50真实Y与遮洞条件] --> S[r10空间self attention80]
 D --> T[r14时间self attention80]
 S --> E[同11GT与8真实DEV]
 T --> E
 E --> H[原生与写回视频／逐帧HTML]
```

唯一变量为空间→时间self attention；两臂均80张量49,574,080参数，原DriveEditor架构及初始化、官方106目标encoder、160步、320×576、AdamW 1e-5/wd.01、seed6201、原loss和50train/25world相同。160次case/window顺序实际核对一致，所有loss/梯度有限，首次80张量均有有限非零梯度；峰值10.529GiB allocated/11.115GiB reserved。没有叠加r13 CFG1、加loss权重、更多参数或继续训练r10。

推理同seed42/25steps、10帧576×1024、默认CFG1.2→2.0、previous=false。旧四臂76窗逐帧RGB/H/Y匹配才复用，19窗新r14；95/95完成。旧合成7case/4world与新过程4case/3world分别统计，真实8例是已曝光DEV，无真实去车GT，final未用。

| scene等权MAE | r7 | r8 | r10 | r14 | r14对r10 |
|---|---:|---:|---:|---:|---:|
| 新过程洞H／4case/3world | .065898 | .068389 | .079612 | .064263 | −19.28% |
| 新过程保护B／3case/2world | .099487 | .099700 | .111563 | .090436 | −18.94% |
| 旧合成洞H／7case/4world | .056969 | .056371 | .056924 | .056356 | −1.00% |
| 旧合成保护B／4case/2world | .097119 | .101757 | .096497 | .101244 | +4.92% |

19例固定f5均实际查看，人工空、时序结论空。P006道路额外棕黄色车在r14固定帧未再出现，整窗H/B对r10分别−62.7%/−50.8%，但后方保护白车仍有错误浅色补片。P012仍有车身融合；P016仍有灰轮／腿状残留，不能说背景闭合。真实A022仍补黑车，A041仍车形涂抹，A013客车前部箱状，A048停车车辆拉伸；其它真实例缺明确稳定增量，未知隐藏身份不判恢复。像素降低不是实体通过率。

当前一秒窗口、仅4扫过train/3world、扫过val仅1world、dense train1world、旧开发形状和GT/LiDAR辅助限制仍在。可以拒绝“这批数据和预算的时间更新已经解决真实任务”，不能否定时间注意力或整体路线。保留权重、原始PNG、全部正反对照；不追加同配方步数。

[19例七列视频与逐帧评分](C:/Users/dengyunlong/Documents/Codex/2026-09-26/xia/outputs/v77-target-protected-r14/index.html)。228视频实际解码2280帧、1330引用图片、19固定帧对照与JS语法通过；浏览器同步播放尚未验证。全部原生另链、人工0/1/2空。task WS-V77-TARGET-PROTECTED-20260929/r14，wm-3090-1001，failure_ledger_refs [V77-F02]。
''',
 'r15':'''# r15：沿合法虚拟路径制造相对遮挡变化

12个新候选全部达到独立AI2，保留作为下一批数据；覆盖目标未达，训练0。

```mermaid
flowchart LR
 Y[真实Y／相机／已有合法虚拟路径] --> F[沿A朝向拟合相对速度]
 F --> G[全帧空间／实际SAM2交叠]
 G --> C[X影响全被H擦除]
 C --> Q[独立AI＋人工逐帧页]
 Q --> K[6扫过train／0扫过val，训练0]
```

冻结46来源/32world，每来源最多4旧合法锚点、两端目标±.4拟合、相对速度delta限.25–8m/s；无输出选源或seed网格。原地面2.5m支持、道路、.3m间距、ego禁入、尺寸/截边、深度和85%保护遮挡上限未放宽；新增速度朝向一致性检查。全部46处理完成，12实际新候选：train11（6扫过/4world、5显露）、val1显露/scene0289。目标8扫过train、3扫过val、2隔离valworld未达，不训练这一不足数据。

120帧磁盘Y/X/H重载：Y与真实RGB重解码完全一致，全部X影响在H内，masked-X=masked-Y、ego洞像素0；时序几何与实际掩罩合同通过。只有实际活跃保护车标B/C，速度标注是相对旧路径delta而非绝对世界速度。多种边缘/膨胀示意全擦除，不用车漆假或灰贴片作为拒绝理由。

独立gpt-6-sol xhigh，无fast；逐例0/5/9四栏，疑点另看原尺寸标注，12pass/AI2、0uncertain/0reject。这是输入质量准入而非模型效果、人类通过或视频全时序认证。T009尽管面积近似稳定，遮挡从车身右下转左下；T010扫离、T007显露变化均有三帧视觉支持。

[12例四列全十帧审核](C:/Users/dengyunlong/Documents/Codex/2026-09-26/xia/outputs/v77-target-protected-r15/data_review.html)。48视频实际解码480帧、480图片引用、12三帧联系图、JS语法通过；人工全空。保持原规则和全部拒绝，不无限扩同来源拟合；下一步更长真实曝光窗口补过程和隔离验证覆盖。task WS-V77-TARGET-PROTECTED-20260929/r15，wm-3090-1001，failure_ledger_refs [V77-F02]。
'''}
 for run,report in reports.items():
  (T/run/'report.md').write_text(report);E=P/f'docs/autoresearch/worldsim_v77/target_protected_20260929/{run}';E.mkdir(exist_ok=True)
  names={'r14':['closeout.json','results_summary.json','decision.json','assistant_effect_reviews.json','delivery_validation.json'],'r15':['closeout.json','coverage_closeout.json','independent_data_reviews.json','technical_checks.json','delivery_validation.json']}[run]
  for name in names:shutil.copy2(T/run/name,E/name)
  if run=='r14':shutil.copy2(T/run/'training/scope_control.json',E/'scope_control.json')
  filename={'r14':'TARGET_PROTECTED_TEMPORAL_SCOPE_R14.md','r15':'TARGET_PROTECTED_PATH_PROCESS_R15.md'}[run];(P/'docs/v77'/filename).write_text(report)
 idx=P/'docs/EXPERIMENTS.md';lines=idx.read_text().splitlines()
 for i,l in enumerate(lines):
  if l.startswith('| WS-V77-TARGET-PROTECTED-20260929 / r14 |'):lines[i]='| WS-V77-TARGET-PROTECTED-20260929 / r14 | 同80张量/160步时间更新；新GT改善，真实8DEV无稳定收益，228视频验收 | [报告](v77/TARGET_PROTECTED_TEMPORAL_SCOPE_R14.md) |'
  if l.startswith('| WS-V77-TARGET-PROTECTED-20260929 / r15 |'):lines[i]='| WS-V77-TARGET-PROTECTED-20260929 / r15 | 相对合法路径，12独立AI2；6train扫过/4world、val扫过0，训练0 | [报告](v77/TARGET_PROTECTED_PATH_PROCESS_R15.md) |'
 idx.write_text('\n'.join(lines)+'\n')
 f=P/'docs/research_failures/entries/V77-F02.md';txt=f.read_text()
 if '## r14：同数量时间模块控制' not in txt:
  txt+='''\n\n## r14：同数量时间模块控制

同r10的50train/25world、80张量49.57M、160步和实际抽样顺序，唯一空间→时间self attention。新过程GT洞/保护车scene等权MAE对r10 −19.28%/−18.94%，但旧保护车+4.92%；真实8DEV固定f5关键失败仍在，不推广、不关机。P006额外车固定帧改善不等于真实任务跨例收益；P016灰轮状残留仍在。95窗完整，228视频解码，human空，final未用。不宣布时序模块整体无效、不加相同训练步数。[报告与图](../../v77/TARGET_PROTECTED_TEMPORAL_SCOPE_R14.md)。failure_ledger_delta: updated V77-F02。

## r15：合法路径的相对速度产生新数据但验证仍缺扫过

46来源/32world固定小预算拟合，全空间/实际mask关卡得到12候选、120帧合同通过，独立gpt-6-sol xhigh无fast全部AI2。6扫过train/4world、5train显露、1val显露，val扫过0，因此未达8train/3val/2valworld、训练0。合法数据保留，不能把覆盖不足说成输入质量失败或模型失败；停止同来源无限搜索，下一步延长实际曝光窗口，不放宽物理门槛。[报告与图](../../v77/TARGET_PROTECTED_PATH_PROCESS_R15.md)。failure_ledger_delta: updated V77-F02。
'''
  f.write_text(txt)
 (P/'docs/RESEARCH_STATUS.md').write_text('''# 当前研究状态

2026-10-01，v77，wm-3090-1001。r13采样控制无稳定真实收益；r14等参数量时间模块完成同160步与19新窗、95总窗，实际抽样顺序一致。新过程GT洞/保护MAE对r10下降19.28%/18.94%，旧保护+4.92%，真实8曝光DEV固定f5关键失败仍在；不推广、不能以GT下降代替真实任务。

r15有限路径相对速度工厂46来源完成，12独立AI2候选全部保留；6扫过train/4world，val扫过0，未达覆盖目标，训练0。r14七列19例与r15四列12例HTML均已本地实际解码/引用校验，人工空，单帧不判视频时序。

下一步延长为约3秒真实连续曝光并补隔离验证过程。数据与采样合同要先登记、全帧物理/mask及独立AI2后才能训练；不追加旧同配方步数，不按模型输出换验证例。真实跨例收益关机条件仍false，不关机。全部旧/新模型、拒绝与监督保留，无新自动化，final未用。

参见[r14](v77/TARGET_PROTECTED_TEMPORAL_SCOPE_R14.md)、[r15](v77/TARGET_PROTECTED_PATH_PROCESS_R15.md)，failure_ledger_refs [V77-F02]。
''')
 print('closed r14 and r15; real benefit false, no shutdown')
if __name__=='__main__':main()

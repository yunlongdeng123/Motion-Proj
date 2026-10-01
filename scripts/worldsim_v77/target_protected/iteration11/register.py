"""先登记过程来源控制和用户条件关机要求，不自动触发电源操作。"""
from pathlib import Path
import sys,shutil
sys.path.insert(0,str(Path(__file__).parent));from process_sources import O,T,read,dump
P=Path('/root/autodl-tmp/motion_proj_v77');E=P/'docs/autoresearch/worldsim_v77/target_protected_20260929/r11'
PLAN='''# r11：过程优先的来源与端点轨迹控制

task WS-V77-TARGET-PROTECTED-20260929 / r11，wm-3090-1001，v77；failure_ledger_refs [V77-F02]。先数据准入，再决定模型试验，不按下游结果换数据。

```mermaid
flowchart LR
 S[未进入r9工厂的真实窗口] --> P[真实投影变化优先选来源]
 P --> A[地面中点＋相机端点拟合世界速度]
 A --> Q[全帧空间／mask／独立AI2]
 Q --> D[固定训练与隔离过程验证]
 D --> F[原结构DriveEditor 同80张量160步]
 F --> R[固定真实DELETE与补景视频对照]
 R --> H[有跨例真实收益才允许收工关机]
```

r10新过程退化，不推广。新控制针对来源不足：旧工厂排序用ego+B世界位移，可能优先挑到两者同速、图像相对稳定的窗口。此次在未搜索的真实短窗里按B实际投影变化和ego路径选32个不同世界来源，近静止ego＋运动B优先；新未训练世界和已训练世界各最多16。既有验证世界隔离，真实final不使用。

每个来源只有9个中点（距B朝相机6/8/10m，横向0/±1.75m），速度由相机端点归一化B投影中心差拟合目标±0.5扫过，限绝对8m/s；同时保留世界静止。只把端点作为规划代理，最终用真实SAM2交叠测实际过程，不能把预测扫过当成功。保持真实Y、已有mesh、道路/地面/0.3m间距/ego禁入/遮挡深度/mask完整覆盖/连续性门槛；不放宽分割或空间规则。

全帧机器＋三独立扫描检查、独立gpt-6-sol xhigh无fast查看0/5/9，只有AI2可训练，人工留空。目标约50train/≥20world、单world≤3，8扫过train/≥3world、3扫过val/≥2隔离world。拒绝/待定均保留。如果这次32来源仍不足，不启动相同160步训练；明确记录缺额，下一步只改有证据的入口，不扩大旧池网格。

若准入，通过后从原权重同80空间attention/160步/320×576/seed6201/原loss开始；推理同576×1024、10帧、seed42、25steps、previous=false，原/r7/r8/r10和r11对照，旧结果只在输入逐帧相同才复用。GT合成任务按过程/证据支持分开；真实无去车GT，不编造MAE或身份。

用户本轮明确授权完成后关机，但必须真实删除＋补景有收益，否则继续。验收优先固定A022/A013/A048等可观察真实DEV：跨至少两个不同scene有明确去目标/保后车或邻车改善、无新的严重误删，主agent与独立实际视频帧复核一致，不能用合成MAE、训练loss或单帧局部改善替代。固定8真实DEV均保留、曝光属性明示；未知隐藏身份不能判成已恢复。人工0/1/2仍由用户填写。

没有满足上述真实任务条件就不关机；即使满足，也需交付HTML、保存全产物、提交推送并核对没有其它作业/可启动控制器后才执行。无自动关机脚本；不是宣称能保证模型一定成功。
'''
def main():
 O.mkdir(exist_ok=True);E.mkdir(parents=True,exist_ok=True)
 if (O/'run.json').exists():print('already registered');return
 b=O/'docs_before_r11';b.mkdir(exist_ok=True)
 for rel in ['docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md','docs/research_failures/entries/V77-F02.md']:shutil.copy2(P/rel,b/Path(rel).name)
 run={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r11','host':'wm-3090-1001','stage':'registered_source_process_control','failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'pending','training_steps':0,'source_budget':32,'human_verdict':None,'final_used':False,'power_authorization':{'source':'user latest request','condition':'demonstrated cross-case real DELETE plus completion benefit, delivery saved and pushed, no other active or launching jobs','satisfied':False},'automatic_shutdown':False}
 dump(O/'run.json',run);dump(E/'run.json',run);(O/'plan.md').write_text(PLAN);(E/'plan.md').write_text(PLAN)
 (P/'docs/RESEARCH_STATUS.md').write_text('''# 当前研究状态

2026-10-01，v77，默认wm-3090-1001。用户授权自主继续到真实DELETE＋补景有收益，再交付保存并关机；未达到时不关机。r10四臂已交付但不推广，不能用合成误差下降视作完成。

当前唯一新run WS-V77-TARGET-PROTECTED-20260929/r11：按实际B投影变化、ego视差和近静止ego＋运动B优先选择未进入r9工厂的新窗口，固定32来源和相机端点速度拟合。原80张量/160步/原loss不改，尚未准入数据或训练，不重复旧进程。保持全部r7/r8/r10数据、权重、拒绝和人工评分；final未用。

过程覆盖及物理、mask、独立AI2共同准入后才训练；覆盖不足就停本次具体无效搜索并改有证据的入口。真实任务验收看固定DEV跨例目标删净/保护车保留及视频连续性，合成GT误差只作辅助。人类verdict不代填，未知身份保持未知。

参见[r11预案与组件图](autoresearch/worldsim_v77/target_protected_20260929/r11/plan.md)、[r10反例与结果](v77/TARGET_PROTECTED_TEMPORAL_CONTROL_R10.md)。failure_ledger_refs [V77-F02]。无新定时任务或自动电源控制。
''')
 idx=P/'docs/EXPERIMENTS.md';lines=idx.read_text().splitlines();anchor=next(i for i,l in enumerate(lines) if l.startswith('|---'));lines.insert(anchor+1,'| WS-V77-TARGET-PROTECTED-20260929 / r11 | 过程优先新来源32世界，端点拟合速度；准入后同80张量160步，真实收益前不关机 | [预案](autoresearch/worldsim_v77/target_protected_20260929/r11/plan.md) |');idx.write_text('\n'.join(lines)+'\n')
 print('registered r11 and conditional power scope')
if __name__=='__main__':main()

"""记录r11零产出边界并登记真实轨迹时间平移控制。"""
from pathlib import Path
import sys,shutil
from collections import Counter
sys.path.insert(0,str(Path(__file__).parent));from process_sources import T,read,dump
P=Path('/root/autodl-tmp/motion_proj_v77');O=T/'r12';E=P/'docs/autoresearch/worldsim_v77/target_protected_20260929'
def main():
 O.mkdir(exist_ok=True)
 if (O/'run.json').exists():print('already registered');return
 rows=[read(p) for p in (T/'r11/planned').glob('*.json')];assert len(rows)==32 and sum(len(r['candidates']) for r in rows)==0
 summary={'sources':32,'candidates':0,'training_steps':0,'rejections':dict(sum((Counter(r['rejects']) for r in rows),Counter())),'stage':'closed_source_endpoint_placement_zero_yield','failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'updated V77-F02; no new ID','result_is_model_negation':False,'power_condition_satisfied':False,'human_verdict':None}
 dump(T/'r11/closeout.json',summary);dump(E/'r11/closeout.json',summary)
 old=read(T/'r11/run.json');old.update(stage=summary['stage'],failure_ledger_delta=summary['failure_ledger_delta']);dump(T/'r11/run.json',old);dump(E/'r11/run.json',old)
 b=O/'docs_before_r12';b.mkdir(exist_ok=True)
 for rel in ['docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md','docs/research_failures/entries/V77-F02.md']:shutil.copy2(P/rel,b/Path(rel).name)
 plan='''# r12：真实轨迹时间平移遮挡

task WS-V77-TARGET-PROTECTED-20260929/r12，wm-3090-1001，v77，failure_ledger_refs [V77-F02]。

```mermaid
flowchart LR
 Y[当前真实视频Y] --> C[当前相机＋障碍＋保护mask]
 A[同场景真实车辆轨迹] --> T[时间平移±0.5/±1秒]
 T --> C
 C --> Q[道路地面／间距／遮挡／显露全帧检查]
 Q --> I[独立AI2准入]
 I --> F[同架构同模块同预算微调]
 F --> R[真实DELETE＋补景固定视频验收]
```

r11过程来源32世界均无候选，不训练。端点拟合仍依赖人工放置中点，地图/地面/ego/邻车约束未通过；不能因此否定模型。r12复用相同冻结32来源，但完全更换轨迹产生机制：在同世界8真实keyframe上下文内，原始车辆轨迹时间平移±0.5/±1秒。无需猜新的道路中点或世界速度，GT轨迹没有支撑即拒绝；不能外推。

Y保持当前真实RGB，不用生成图片监督；A为当前相机投影下的平移真实轨迹及已有mesh。A位置对当前Y中所有GT actor重新验证0.3m距离、地图、LiDAR支持、ego禁入和正确前后深度。GT车底离真实拟合地面>0.35m拒绝，小偏差统一贴地且保存修正。每帧位置/yaw/尺度连续、hole覆盖全部合成影响，不同时换网络、loss或模块。

最多32来源，每个真实car track四次固定时间偏移，各过程最多2候选/source。不放宽既有空间、分割或连续性门槛；技术通过后仍需独立gpt-6-sol xhigh无fast看0/5/9。目标沿用8扫过train/≥3世界，3扫过val/≥2全旧训练隔离世界，约50train/≥20世界。覆盖不足时不启动相同训练或无限枚举偏移。

满足数据准入再同80张量/160步/官方106encoder/原loss/seed6201控制；GT验证和真实DELETE DEV均冻结、无final。时间平移是训练遮挡制造，不是把未来GT暴露给推理条件；Y仅监督。真实被隐藏实体身份没有RGB GT时保持未知。

真实任务收益条件继承r11：固定真实DEV跨至少两个scene明确删净/保保护车收益，无新严重误删，实际视频帧主/独立检查一致。满足并交付保存推送、确认无其它作业后才关机；目前条件false，不自动关机。
'''
 (O/'plan.md').write_text(plan);(E/'r12').mkdir(exist_ok=True);(E/'r12/plan.md').write_text(plan)
 run={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r12','host':'wm-3090-1001','stage':'registered_real_track_shift','source_budget':32,'time_shifts_s':[-1.,-.5,.5,1.],'training_steps':0,'failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'pending','human_verdict':None,'final_used':False,'power_authorization':old['power_authorization'],'automatic_shutdown':False}
 dump(O/'run.json',run);dump(E/'r12/run.json',run);dump(O/'source_selection.json',read(T/'r11/source_selection.json'))
 (P/'docs/RESEARCH_STATUS.md').write_text('''# 当前研究状态

2026-10-01，v77，默认wm-3090-1001。当前r12真实同世界轨迹时间平移数据控制；r11的32过程来源/端点拟合0候选，训练0，已停止该具体放置搜索。r10已交付、真实无稳定收益，不推广。

r12复用冻结32来源，真实8keyframe支持内±0.5/±1秒轨迹，不猜道路位置/速度、不外推；当前相机、真实Y、现有mask与mesh不变。全部当前障碍间距、道路/地面/ego/深度/连续性/覆盖规则重新检查，独立AI2后才准入。训练与真实任务评测未启动。

用户条件关机授权有效：真实DELETE＋补景跨例视频有收益并交付保存推送、确认没有其他作业后才关机；目前不满足，继续迭代。人工评分留空，final未用，旧全部数据/权重保留，无新定时或自动关机。

参见[r12预案与图](autoresearch/worldsim_v77/target_protected_20260929/r12/plan.md)、[r11零产出](autoresearch/worldsim_v77/target_protected_20260929/r11/closeout.json)。failure_ledger_refs [V77-F02]。
''')
 failure=P/'docs/research_failures/entries/V77-F02.md';s=failure.read_text();s+='\n\n## r11：过程来源与端点拟合仍未产生合法训练遮挡\n\n32个新世界来源选择依据真实投影变化，全部0候选，训练0；保留固定来源、完整拒绝和端点候选。主要受道路/地面、空间间距、ego与未核实邻车限制，不改变质量门槛，不宣布模型失败。该具体中点放置搜索停止；r12改为真实同世界轨迹时间平移，在实际GT时间支撑内制造A，并对当前Y重查所有几何/遮罩/过程。真实收益关机条件未满足，不关机。failure_ledger_delta: updated V77-F02，不新增ID。\n';failure.write_text(s)
 idx=P/'docs/EXPERIMENTS.md';lines=idx.read_text().splitlines()
 for i,l in enumerate(lines):
  if l.startswith('| WS-V77-TARGET-PROTECTED-20260929 / r11 |'):lines[i]='| WS-V77-TARGET-PROTECTED-20260929 / r11 | 新过程来源32世界，端点拟合0候选；保留拒绝、训练0，停止中点放置搜索 | [零产出](autoresearch/worldsim_v77/target_protected_20260929/r11/closeout.json) |'
 at=next(i for i,l in enumerate(lines) if l.startswith('|---'));lines.insert(at+1,'| WS-V77-TARGET-PROTECTED-20260929 / r12 | 同世界真实轨迹时间平移，固定32来源与四偏移；物理/过程独立质量准入后才训练 | [预案](autoresearch/worldsim_v77/target_protected_20260929/r12/plan.md) |');idx.write_text('\n'.join(lines)+'\n')
 print('r11 stopped without training; r12 registered',summary)
if __name__=='__main__':main()

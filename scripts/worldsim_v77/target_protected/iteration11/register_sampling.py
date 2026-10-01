"""数据入口停止后，登记一个同权重、同输入的采样强控制。"""
from pathlib import Path
import sys,shutil
sys.path.insert(0,str(Path(__file__).parent));from process_sources import T,read,dump
P=Path('/root/autodl-tmp/motion_proj_v77');O=T/'r13';E=P/'docs/autoresearch/worldsim_v77/target_protected_20260929'
PLAN='''# r13：采样条件外推的单一强控制

task WS-V77-TARGET-PROTECTED-20260929/r13，wm-3090-1001；failure_ledger_refs [V77-F02]。

```mermaid
flowchart LR
 X[冻结真实DELETE RGB＋H] --> R[相同有效r7权重]
 R --> B[原线性CFG 1.2到2.0]
 R --> C[固定CFG 1.0 仅条件预测]
 B --> V[8真实DEV＋2已知GT对照]
 C --> V
 V --> Q[实际十帧图像／视频检查]
```

r11/r12两类数据入口都0合格候选，不训练；保存全部拒绝和工程证据。地面raw支持恢复不能解决所有放置/观测缺口，不继续枚举同来源/偏移。本轮先做一个廉价强控制，检查已有适配权重在采样条件外推中的错误是否放大：同有效r7参数、相同输入/删除范围、seed42、25steps、previous=false，唯一修改2D guider从官方线性1.2到2.0改为固定1.0。x_u+1*(x_c-x_u)=x_c，未改变architecture或训练、不改3D guider。

不预先认定CFG有错，不保证改善，不扫参数。固定8真实曝光DEV全跑，加P006/P012两个已知Y保护车/额外实体哨兵。原/r7/r8/r10输出保留。只有跨例真实去目标＋补景收益且无新严重误删，经实际10帧主/独立复核，才能考虑推广与满足用户条件关机。单帧或合成MAE不能替代真实验收。失败则关闭该采样策略，不试1.1/1.3等网格，继续有依据的数据或更新范围控制。

真实隐藏身份缺GT保持未知，人工0/1/2为空，不声称final或三秒长视频结论。目前关机条件false；仍需交付并提交推送、核实无其它作业才能关机。
'''
def main():
 O.mkdir(exist_ok=True)
 if (O/'run.json').exists():print('already registered');return
 assert read(T/'r12/closeout.json')['candidates']==0
 b=O/'docs_before_r13';b.mkdir(exist_ok=True)
 for rel in ['docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md','docs/research_failures/entries/V77-F02.md']:
  target=b/Path(rel).name
  if not target.exists():shutil.copy2(P/rel,target)
 cases=[c for c in read(T/'r10/evaluation_plan.json')['cases'] if c['kind']=='real_development' or c['eval_id'] in ['temporal_P006','temporal_P012']];assert len(cases)==10
 dump(O/'evaluation_plan.json',{'cases':cases,'arms':['r7_default_reused','r7_cfg1'],'seed':42,'steps':25,'frames':10,'scope':'single preregistered sampling operator; no model/data/architecture change, no grid or seed selection','real_actor_free_GT':None,'human_verdict':None,'final_used':False})
 r={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r13','host':'wm-3090-1001','stage':'registered_single_guidance_control','training_steps':0,'fresh_inference_windows':10,'failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'pending','power_authorization':read(T/'r12/run.json')['power_authorization'],'human_verdict':None,'final_used':False,'automatic_shutdown':False};dump(O/'run.json',r);dump(E/'r13/run.json',r)
 for p in [O/'plan.md',E/'r13/plan.md']:p.write_text(PLAN)
 (P/'docs/RESEARCH_STATUS.md').write_text('''# 当前研究状态

2026-10-01，v77，wm-3090-1001。r11新过程来源/端点拟合32来源0候选，r12真实轨迹时间平移32来源及raw地面支持恢复后仍0候选，均训练0，停止对应具体搜索，不放宽空间/质量条件。

当前r13单一采样强控制：有效r7权重与固定8真实DEV＋2GT哨兵不变，2D CFG原线性1.2到2.0对固定1.0；同seed42/25steps/previous=false，3D guider不改，不做参数网格。该诊断不预设机制成立。

用户授权真实DELETE＋补景有跨例收益，完成交付保存推送并确认无其它作业后才关机；目前条件未满足，不关机。全部旧模型、拒绝、分数保留，人工空，final未用，无新自动化。

参见[r13预案与图](autoresearch/worldsim_v77/target_protected_20260929/r13/plan.md)、[r12零产出](autoresearch/worldsim_v77/target_protected_20260929/r12/closeout.json)。failure_ledger_refs [V77-F02]。
''')
 f=P/'docs/research_failures/entries/V77-F02.md';s=f.read_text();marker='## r12：真实轨迹时间平移仍缺物理与观测准入'
 if marker not in s:
  s+='\n\n'+marker+'\n\n同32来源四个真实时间偏移0候选，训练0。真实GT时间支撑、地面支持/高度、当前车辆距离和视野联合限制产量。观测oracle发现部分真实车辆脚点被旧GT框整体去点误拒，恢复原平面±0.08m raw近地返回使部分支持恢复，2.5m阈值不变，但并未产生合格候选；原与修复产物均保存。停止同来源时间偏移枚举，不把零造数说成模型失败。r13另登记同有效r7参数的单一CFG外推控制，未更换网络/seed/模型。真实收益与关机条件false。failure_ledger_delta: updated V77-F02，不新增ID。\n';f.write_text(s)
 i=P/'docs/EXPERIMENTS.md';lines=i.read_text().splitlines()
 for n,l in enumerate(lines):
  if l.startswith('| WS-V77-TARGET-PROTECTED-20260929 / r12 |'):lines[n]='| WS-V77-TARGET-PROTECTED-20260929 / r12 | 同世界真实轨迹平移32来源0候选，raw地面修复未改善产量，训练0并停止该搜索 | [记录](autoresearch/worldsim_v77/target_protected_20260929/r12/closeout.json) |'
 at=next(n for n,l in enumerate(lines) if l.startswith('|---'));lines.insert(at+1,'| WS-V77-TARGET-PROTECTED-20260929 / r13 | 同r7参数单一CFG1控制，固定8真实DEV＋2GT哨兵，不改架构/数据/seed | [预案](autoresearch/worldsim_v77/target_protected_20260929/r13/plan.md) |');i.write_text('\n'.join(lines)+'\n')
 print('registered 10 fixed windows, no training, no parameter grid')
if __name__=='__main__':main()

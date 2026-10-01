"""在采样结果前冻结同数量时间参数控制，不按新输出改数据或seed。"""
from pathlib import Path
import sys,copy,shutil
sys.path.insert(0,str(Path(__file__).parent));from process_sources import T,read,dump
P=Path('/root/autodl-tmp/motion_proj_v77');O=T/'r14';E=P/'docs/autoresearch/worldsim_v77/target_protected_20260929/r14'
PLAN='''# r14：同数量时间自注意力控制

task WS-V77-TARGET-PROTECTED-20260929/r14，wm-3090-1001，failure_ledger_refs [V77-F02]。

```mermaid
flowchart LR
 D[冻结r10真实Y／遮洞50train] --> S[空间self attention80 r10]
 D --> T[时间self attention80 r14]
 S --> E[同11GT＋8真实DEV评测]
 T --> E
 E --> V[原生／局部写回十帧对照]
```

源码及实际checkpoint键确认r7/r8/r10只更新空间transformer_blocks.attn1，未更新时间time_stack.attn1。本轮不预设模块错误；用相同50训练例/25world及11GT/8真实冻结验证，唯一更换更新张量的时间位置。两组均80张量、49,574,080参数，保持原模型初始化、官方106目标encoder、160steps、320×576、AdamW1e-5/0.01、seed6201、原StandardDiffusionLoss。既有空间和条件模块冻结，不同时扩范围、加权loss或加训练步数；原结构不变。

r10训练集合及原QA完整保留，本轮不声称解决过程覆盖缺额：4个扫过train/3world，新扫过val仅1world，dense train1world、一秒窗口仍限制结论。r11/r12零候选数据不训练、不混入集合，final未用。两组实际训练采样顺序/数据窗口要核对。

推理同默认CFG线性1.2到2.0、原seed42/25steps/previous=false/576×1024，r13 CFG1不叠加。原/r7/r8/r10的19例结果在每帧X/H/Y一致时复用，r14新跑19窗。报告分开旧合成/新过程GT与真实无GT任务；loss或GT MAE下降不能替代真实DELETE跨例收益。

同一160步只做一次；若GT和真实都无收益，停止这个更新位置，不临时继续加步数或扩参数救结果。原权重/所有拒绝/优化器/反例保留。实际视频帧主/独立复核跨至少两scene真实去目标及保护车收益、无新严重误删，才扩为3秒真实视频确认。完成真实收益验收、HTML交付、保存推送及确认无其他任务后才关机；目前条件false，人工空。
'''
def main():
 O.mkdir(exist_ok=True);E.mkdir(parents=True,exist_ok=True)
 if (O/'run.json').exists():print('already registered');return
 scope=read(T/'r13/equal_size_module_scope.json');assert scope['spatial']['parameters']==scope['temporal']['parameters']==49574080 and scope['temporal']['tensors']==80
 old=read(T/'r10/dataset_catalog.json');catalog=copy.deepcopy(old);catalog.update(scope_control='exact same cases and roles as r10; only parameter update location changes');dump(O/'dataset_catalog.json',catalog)
 adm=read(T/'r10/admission_result.json');assert adm['ready'];dump(O/'admission_result.json',adm)
 plan=copy.deepcopy(read(T/'r10/evaluation_plan.json'));plan['arms']=['base','r7','r8','r10','r14'];plan.update(scope_control=True,selection_before_r14_outputs=True);dump(O/'evaluation_plan.json',plan)
 r={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r14','host':'wm-3090-1001','stage':'frozen_equal_size_temporal_scope_control','operator':'temporal_self_attention_only','module_tensor_count':80,'module_parameters':49574080,'training_steps':160,'dataset_exact_same_as':'r10','failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'pending','power_authorization':read(T/'r13/run.json')['power_authorization'],'automatic_shutdown':False,'human_verdict':None,'final_used':False};dump(O/'run.json',r);dump(E/'run.json',r);dump(E/'module_scope.json',scope)
 for p in [O/'plan.md',E/'plan.md']:p.write_text(PLAN)
 b=O/'docs_before_r14';b.mkdir(exist_ok=True)
 for rel in ['docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md','docs/research_failures/entries/V77-F02.md']:shutil.copy2(P/rel,b/Path(rel).name)
 idx=P/'docs/EXPERIMENTS.md';lines=idx.read_text().splitlines();at=next(n for n,l in enumerate(lines) if l.startswith('|---'));lines.insert(at+1,'| WS-V77-TARGET-PROTECTED-20260929 / r14 | 同r10数据、同80张量/49.57M/160步，只更新时序self attention；真实收益门槛不变 | [预案](autoresearch/worldsim_v77/target_protected_20260929/r14/plan.md) |');idx.write_text('\n'.join(lines)+'\n')
 print('frozen equal size temporal scope control; GPU launch waits for r13 exit')
if __name__=='__main__':main()

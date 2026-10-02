"""登记用户授权的Temporal reveal与简单actor-state条件阶段。"""
from pathlib import Path
import json, shutil

P = Path('/root/autodl-tmp/motion_proj_v77')
T = Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')
O = T/'r21'


def dump(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n')


def main():
    O.mkdir(exist_ok=True)
    E = P/'docs/autoresearch/worldsim_v77/target_protected_20260929/r21'
    E.mkdir(parents=True, exist_ok=True)
    if (O/'run.json').exists():
        raise RuntimeError('r21已登记，检查实际状态后继续，禁止覆盖')
    b = O/'before_changes';b.mkdir()
    for rel in ['docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md','scripts/worldsim_v77/repair_drive.py','scripts/worldsim_v77/delete_audit/mask_batch.py']:
        dst=b/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(P/rel,dst)
    plan = '''# r21：入口修复与整窗支持，进入简单条件之前

task WS-V77-TARGET-PROTECTED-20260929/r21，2026-10-02，wm-3090-1001，failure_ledger_refs [V77-F02]。

用户最新授权恢复auto research：工程bug直接修，先验证Temporal reveal＋projected actor-state条件，条件有效后才加surfel。原架构不再完全不变：保留原9通道及骨干，后续新增空间残差与独立受区域约束的身份分支。研究目标是实际DELETE中保护车与无车背景两侧的跨场景收益。

```mermaid
flowchart LR
 R[真实RGB＋SAM目标] --> M[完整目标H与alpha／空mask]
 Y[真实视频Y] --> X[合成遮挡X＋H]
 X --> C[先擦除再缩放]
 M --> C
 C --> L[单次20或30帧DriveEditor]
 Y -.仅监督.-> L
 C --> A[后续合法可见actor状态]
 A --> P[O/N/U/Q＋身份／几何条件]
 P -.简单Adapter待验证.-> L
 L --> V[删净／保护身份／不新增车]
```

本run只修入口和验证单次模型长窗，不做正式微调。保存旧70例与旧权重；从已有SAM另建v2 mask，无GT硬裁，目标alpha全1、洞外全0，空目标帧零写回。有可见保护实例时冲突必须拒绝；仅GT邻车投影交叠只记疑点，不当分割真值。先复核8个已曝光真实DEV的输入修复，正式模型比较时统一使用同一个新入口。

初始化FP32修复继承r19。按Y/H/实际遮后条件去重；旧目录不改，得到新的目录与重复映射。新训练/验证同时隔离背景scene和mask来源；旧跨split共享mesh只能作开发探针，不能冒称新独立评测。

冻结长窗工程探针：r16/L001，真实30帧，320×576。先30帧训练前向/反向（无optimizer.step）与3采样步推理，只核对shape、梯度、显存和单次整窗；若显存OOM则固定退到20帧，同类对照统一长度/分辨率。不能拆窗后声称长证据进入同一次调用。3步不是效果评测。最多两种长度，不扫分辨率/seed。训练仅原sigma加权去噪latent损失，未启用新区域损失。

后续主方案保留用户原文，不在本run偷做surfel或多分支搜索。先50条工厂质量样本，达标后300–500条/至少30scene；60% reveal、25%密集已知背景、15%普通背景。合法条件必须在遮洞之后构造；O/N/U由真实观测支持，N不能设成1−O。轨迹/GT几何辅助须明示；无真实证据的区域保持未知。固定A无条件/B几何存在状态/C加身份，D仅在C收益成立后登记。

资源先预算，不删除原始输入、关键checkpoint和失败证据。用户允许清不重要旧物，仍先确认可重建与归属。原关机条件沿用，未取得真实任务两侧收益不自动关机。人工分数空，final未用。
'''
    run={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r21','stage':'registered_engineering_and_whole_window', 'training_optimizer_steps':0,'length_probe_order':[30,20],'probe_size':[320,576],'probe_inference_steps':3,'seed':42,'source':'r16/L001','surfel_started':False,'human_verdict':None,'automatic_shutdown':False,'failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'pending'}
    for r in [O,E]:
        dump(r/'run.json',run);(r/'plan.md').write_text(plan)
    shutil.copy2(O/'user_plan.txt',E/'user_plan.txt')
    status='''# 当前研究状态

2026-10-02，v77，wm-3090-1001。用户新授权恢复研究，主线为Temporal reveal＋带身份/几何/可信度的projected actor-state。先验证简单条件价值，完整surfel不启动。当前唯一run r21先修GT硬裁漏车、写回和空mask，生成独立v2入口并按真实模型任务去重；继承r19的FP32初始化修复。旧70例、r7/r14、r18暂停权重与失败产物均保留。

r21先以r16/L001在320×576验证单次30帧训练前后向（优化0步）及3步工程推理，OOM才统一退20帧。随后新数据严格隔离背景和mask来源，只从遮后窗口构建合法条件；50条质量pilot通过后再扩量和A/B/C训练对照。未知不能当空背景，隐藏GT仅监督，受保护身份独立记录。用户允许清可重建的不重要旧物，当前约51GB可用暂不清理。

人工14条排名已保存，A061局部真实收益与P019幻觉均保留；关机完成条件尚未满足，无新自动化。r18不自动续跑，final未用。参见[r21预案](autoresearch/worldsim_v77/target_protected_20260929/r21/plan.md)，failure_ledger_refs [V77-F02]。
'''
    (P/'docs/RESEARCH_STATUS.md').write_text(status)
    p=P/'docs/EXPERIMENTS.md';ls=p.read_text().splitlines();i=next(i for i,l in enumerate(ls) if l.startswith('|---'))
    ls.insert(i+1,'| WS-V77-TARGET-PROTECTED-20260929 / r21 | 入口修复、去重与单次20/30帧工程探针；简单actor-state先于surfel | [预案](autoresearch/worldsim_v77/target_protected_20260929/r21/plan.md) |');p.write_text('\n'.join(ls)+'\n')
    print('REGISTERED_R21')


if __name__=='__main__':main()

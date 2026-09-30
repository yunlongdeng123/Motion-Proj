"""记录最新自主AI准入/微调授权，保留历史人工评分与训练边界。"""
import argparse,json
from pathlib import Path
def main(root,repo):
    if (root/'run_config.json').exists():
        print('ALREADY_REGISTERED: preserve current run and research status');return
    cfg={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r6','date':'2026-10-01',
         'authorization':'用户允许自主CPU/GPU造数及infra迭代，固定标准AI自评2分后微调原DriveEditor并评估',
         'quality_scale':{'0':'空间/洞覆盖/身份/次序等硬失败','1':'质量有疑点或不充分，不能训练','2':'五项输入合同合格、独立QA通过，可作为技术训练候选'},
         'score_source':'assistant technical self-review; not human verdict','human_verdict':None,
         'stop_rule':'工程bug可修；不按生成结果放宽数据标准或强改失败评分；明确无收益则保留反例并停止该控制',
         'architecture_change':False,'initial_GPU':'RTX3090 24GB','seed':42,
         'experiment_roles':'real nuScenes train Y supervised; past val audit cases only development evaluation; isolated final scenes untouched'}
    root.mkdir(parents=True,exist_ok=True);(root/'run_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
    (repo/'docs/RESEARCH_STATUS.md').write_text(f'''# 当前研究状态

2026-10-01，v77；`WS-V77-TARGET-PROTECTED-20260929/r6`。用户最新明确解除CPU/GPU限制，授权自主造数/infra迭代、固定标准AI自评2分后微调DriveEditor并比较效果。训练准入从此前人工全检前置改为本次授权的技术AI2分；历史human verdict保持null，不把AI改成人工分。

已启动r4四个新保护实例的冻结SAM2。沿用真实Y、同一位姿与五项输入要求，不以GT包络认证实例mask，不强行把1分/失败升级为2分。新mask独立检查后调用r5精确关卡。既有r2/r3技术候选将按同标准归档，拒绝/待定保留。

后续先固定训练集与开发评测，验证实际训练数据条件、反向与24GB显存，再运行有界小规模微调和原权重对照。DriveEditor架构不变，Ω/GLB路线暂不加入；val只作已曝光开发比较，隔离final未用。当前训练步数0，尚无微调效果结论。

run `{root}`。failure_ledger_refs: [V77-F02]；failure_ledger_delta: pending。
''');print('REGISTERED_R6')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);a=p.parse_args();main(a.root,a.repo)

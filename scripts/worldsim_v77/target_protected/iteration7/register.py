"""r7先登记有界因果控制，结果不能改变数据或预算。"""
from pathlib import Path
import json, shutil
REPO=Path('/root/autodl-tmp/motion_proj_v77')
TASK=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')
ROOT=TASK/'r7'
def main():
    ROOT.mkdir(exist_ok=True)
    if (ROOT/'run_config.json').exists():print('already registered');return
    backup=Path('/root/autodl-tmp/codex_backups/WS-V77-TARGET-PROTECTED-20260929/r7-start')
    for rel in ['docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md','docs/research_failures/entries/V77-F02.md']:
        dst=backup/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/rel,dst)
    config={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r7','date':'2026-10-01',
            'purpose':'区分r6无改善的工程/数据监督/更新范围，原架构不变',
            'arms':[{'id':'native_spatial','size':[576,1024],'steps':160,'modules':'same r6 spatial self attention 80 tensors'},
                    {'id':'native_contextual','size':[576,1024],'steps':160,'modules':'existing main spatial+temporal self/cross attention and input/output conv/null embedding; only if memory feasible'}],
            'fixed':'r6 same catalog/split, case order/seed6201, AdamW lr1e-5, unchanged official loss, seed42/25step inference',
            'evaluation':'four frozen synthetic holdouts + one train background/one train protected, chosen by dataset id before training',
            'stop':'OOM ends that arm, preserve evidence; no sampling seed/threshold/loss grid; no promotion based on latent loss',
            'data_change':False,'new_architecture':False,'human_verdict':None,'failure_ledger_refs':['V77-F02'],
            'limits':'holdout4cases one scene; controlled update range bundles existing modules; not individual module attribution; real-development comparison follows only if synthetic improvement'}
    (ROOT/'run_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    catalog=json.loads((TASK/'r6/dataset_catalog.json').read_text())
    for arm in config['arms']:
        dest=ROOT/arm['id'];dest.mkdir();shutil.copy2(TASK/'r6/dataset_catalog.json',dest/'dataset_catalog.json')
    chosen=[]
    for kind in ('background','single_actor'):
        chosen.append(min((c for c in catalog['cases'] if c['split']=='train' and c['type']==kind),key=lambda c:c['dataset_id']))
    cases=[dict(c,eval_id='holdout_'+c['case_id'],role='heldout',frames=list(range(10))) for c in catalog['cases'] if c['split']=='validation']
    cases += [dict(c,eval_id='train_'+c['dataset_id'].replace('/','_'),role='train_diagnostic',frames=list(range(10))) for c in chosen]
    (ROOT/'evaluation_plan.json').write_text(json.dumps({'cases':cases,'seed':42,'steps':25,'native_resolution':[576,1024]},ensure_ascii=False,indent=2)+'\n')
    (REPO/'docs/RESEARCH_STATUS.md').write_text('''# 当前研究状态

2026-10-01，v77；`WS-V77-TARGET-PROTECTED-20260929/r7`排查r6无改善。原r6及默认权重保留，冻结52例catalog和48/4 split。源码发现原blank分支mask_fuse=0导致全画面均匀loss；r6仅80个空间self-attention，范围比官方窄；训练/推理分辨率不同。以上先是候选原因，不先判数据或模块。

本轮先全帧数据分布/权重审计，再登记两个同数据/seed/loss/160步控制：native_spatial仅修正训练至576×1024；native_contextual再扩既有条件/时空attention与入口/出口。若显存不可行停止对应臂，不换尺寸救科学比较。四个既有单场景留出+两个提前固定训练例作效果诊断，不用final。仅合成有收益后加真实已曝光开发例。

GPU允许，无电源/自动化操作。run `/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929/r7`。failure_ledger_refs: [V77-F02]；failure_ledger_delta: pending。
''')
    print('registered r7')
if __name__=='__main__':main()

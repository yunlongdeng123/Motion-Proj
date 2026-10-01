"""r8数据覆盖控制：冻结训练配方，先形成合格数据和隔离评测再训练。"""
from pathlib import Path
import json,shutil
R=Path('/root/autodl-tmp/motion_proj_v77')
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')
O=T/'r8'
def main():
    O.mkdir(exist_ok=True)
    if (O/'run_config.json').exists():print('already registered');return
    backup=Path('/root/autodl-tmp/codex_backups/WS-V77-TARGET-PROTECTED-20260929/r8-start')
    for name in ['docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md','docs/research_failures/entries/V77-F02.md']:
        p=backup/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/name,p)
    cfg={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r8','date':'2026-10-01',
         'purpose':'只改变技术合格训练数据覆盖，评估合成恢复和真实DELETE开发迁移',
         'target_training_cases':50,'minimum_training_receiver_scenes':20,'max_cases_per_receiver_scene':3,
         'target_types':{'background':16,'single_actor':20,'dense_actors':14},
         'source_pool':'available official nuScenes train; first use already materialized continuous windows, no final val/test',
         'data_method':'same metric placement, real LiDAR/map, exact SAM2 silhouette; no generated GT; full synthetic RGB erased before conditions',
         'template_extension':'receiver track may supply a duplicate synthetic A silhouette; original B remains genuine Y. A moved to distinct noncolliding metric pose, same geometry/view gates. Template RGB fully erased. This changes data only, not model inputs/architecture.',
         'quality':'all-frame five-check technical gates plus gpt-6-sol xhigh fixed-frame per-case independent QA; no human score assignment',
         'evaluation':'freeze synthetic scene groups before training; exact same conditions original/r7/new; real A022/A041/A013/A048 and preselected additional well-observed development failures',
         'training':{'start':'original DriveEditor + official106 target encoder, NOT continue r7','modules':'same80 spatial self attention','steps':160,'size':[320,576],'lr':1e-5,'seed':6201,'loss':'official unchanged','optimizer':'AdamW','weight_decay':.01,'clip':1},
         'inference':{'size':[576,1024],'frames':10,'seed':42,'steps':25},
         'stop':'no training before >=20 scenes and all three types qualified; retain shortages/rejects. One frozen new-data arm, no steps/loss/modules grids; report synthetic and real separately.',
         'human_verdict':None,'failure_ledger_refs':['V77-F02'],'power_operations':False}
    (O/'run_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
    (R/'docs/RESEARCH_STATUS.md').write_text('''# 当前研究状态

2026-10-01，v77；`WS-V77-TARGET-PROTECTED-20260929/r8`按最新用户要求扩训练数据覆盖：约50例、至少20个receiver场景、单scene最多3例、背景/单保护车/密集三类。先复用已落盘真实train短窗，再检查来源缺额。真实Y不改，精确SAM2轮廓、米制放置、LiDAR/map/ego/depth/连续性及完整覆盖门槛保持；允许receiver轮廓作为人工复制A模板，RGB全部清除，属数据改变。

训练配方冻结r7有效初始化、同80空间attention、320×576/160步、原loss/seed6201，从原权重开始；不延续r7训练、不扩大模块或加权。合成多场景隔离和已曝光真实DELETE开发两套评测在训练前冻结，三臂原/r7/新数据同条件。合格覆盖不足停止训练，保留缺额；人工评分空、final不用。无电源/自动化操作。

failure_ledger_refs: [V77-F02]；failure_ledger_delta: pending。
''')
    print('registered r8')
if __name__=='__main__':main()

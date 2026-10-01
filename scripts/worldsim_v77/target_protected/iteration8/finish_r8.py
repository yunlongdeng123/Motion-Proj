"""有界r8控制收口；不启动后续训练，不作电源操作。"""
from pathlib import Path
import json, shutil

ROOT=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929/r8')
REPO=Path('/root/autodl-tmp/motion_proj_v77')
DEST=REPO/'docs/autoresearch/worldsim_v77/target_protected_20260929/r8'

def read(path):
    return json.loads(path.read_text())

def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def main():
    result=read(ROOT/'results_summary.json');state=read(ROOT/'evaluation/state.json')
    delivery=read(ROOT/'delivery_validation.json');reviews=read(ROOT/'assistant_effect_reviews.json')
    assert result['all_inference_complete'] and state['all_three_arms_complete'] and state['completed_count']==45
    assert result['same_modules_budget']['all_fixed_recipe_fields_equal']
    assert delivery['success'] and delivery['videos_checked']==392 and delivery['len_images_checked']==3470
    assert len(reviews['cases'])==15 and all(c['reviewed_frames']==[5] and c['human_verdict'] is None for c in reviews['cases'])
    assert read(ROOT/'training/state.json')['steps']==160
    closeout={
        'task':'WS-V77-TARGET-PROTECTED-20260929','run':'r8',
        'stage':'complete_training_evaluation_review_delivery',
        'training':{'cases':50,'receiver_scenes':25,'max_cases_per_scene':3,'types':{'background':24,'single_actor':23,'dense_actors':3},'steps':160,'trainable_tensors':80,'same_recipe_as_valid_r7':True,'from_original_not_r7':True},
        'quality':{'candidates':68,'frames_reloaded':680,'independent_fixed_frame':{'pass':61,'uncertain':5,'reject':2},'reviewer_model':'gpt-6-sol','reasoning_effort':'xhigh','fast':False,'admitted_train':50,'admitted_validation':7,'qualified_unused':4,'input_AI2_not_effect_AI2':True},
        'evaluation':{'synthetic_cases':7,'synthetic_receiver_scenes':4,'protected_cases':4,'protected_receiver_scenes':2,'real_exposed_DEV_cases':8,'arms':['base','r7','new_data'],'windows':45,'frames_per_window':10,'seed':42,'sampling_steps':25,'native_resolution':[576,1024],'real_actor_free_GT':None},
        'synthetic_metrics':result['synthetic_metrics'],
        'real_observation':'All eight fixed-f5 comparisons show no clear improvement over valid r7; A022 regenerated vehicle, A013/A048 structural damage. Other hidden identities unverified. Not temporal/human pass rate.',
        'decision':{'stable_extra_benefit_over_r7':False,'r8_promoted':False,'module_failure_proven':False,'default_original_preserved':True,'valid_r7_preserved':True,'stop_repeat_same_configuration':True,'next_control':'Match training-hole geometry/area/border to real inference contract with physical source checks; keep modules,160steps,loss and frozen evaluations. Then isolated protected loss weighting; module expansion later.','next_control_executed':False},
        'limitations':['Boston daytime official-train sources; original pretraining overlap unknown','shared two exposed DEV mesh shapes across train/validation','dense train only3cases/1scene','protected evaluation only4cases/2scenes','GT/map/LiDAR-assisted synthesis, sparse static evidence not empty-space proof','one-second windows; fixed-frame review not temporal certification','real DEV contains small/truncated/rain cases, not sufficient well-observed-only audit','data sources and silhouette factory both changed, not causal scene-count-only experiment','training vs real mask distribution differs, causal hypothesis unproven'],
        'local_delivery':'C:/Users/dengyunlong/Documents/Codex/2026-09-26/xia/outputs/v77-target-protected-r8/index.html',
        'local_data_review':'C:/Users/dengyunlong/Documents/Codex/2026-09-26/xia/outputs/v77-target-protected-r8/data_review.html',
        'delivery':delivery,'full_remote_run':str(ROOT),
        'preserved':['all source RGB and failed/candidate proposals','real Y/X/H/B masks and geometric QA','original/r7/r8 native PNGs and composites','all160-step training checkpoints and optimizer snapshots','all old baselines and user scores'],
        'lightweight_git_evidence_is_full_backup':False,'human_verdict':None,'final_used':False,'power_operation':None,'new_automation':None,
        'failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'updated V77-F02: no stable data-control increment, protected damage, unproven input distribution gap; no new failure ID'
    }
    dump(ROOT/'closeout.json',closeout)
    controller=read(ROOT/'controller_state.json');controller.update(stage=closeout['stage'],all_inference_complete=True,local_delivery_verified=True,followup_training_started=False,human_verdict=None);dump(ROOT/'controller_state.json',controller)
    DEST.mkdir(parents=True,exist_ok=True)
    for src,name in [(ROOT/'results_summary.json','results_summary.json'),(ROOT/'evaluation/state.json','evaluation_state.json'),(ROOT/'controller_state.json','controller_state.json'),(ROOT/'assistant_effect_reviews.json','assistant_effect_reviews.json'),(ROOT/'delivery_validation.json','delivery_validation.json'),(ROOT/'closeout.json','closeout.json')]:
        # Windows交付核验使用CRLF；Git轻量JSON保持UTF-8/LF，不改变任何字段。
        dump(DEST/name,read(src))
    (DEST/'review_link.md').write_text('''# r8 数据覆盖控制审核入口

[模型五列逐帧审核](C:/Users/dengyunlong/Documents/Codex/2026-09-26/xia/outputs/v77-target-protected-r8/index.html) · [68候选数据四列审核](C:/Users/dengyunlong/Documents/Codex/2026-09-26/xia/outputs/v77-target-protected-r8/data_review.html)

实际392视频/3920帧解码、3470引用JPG、HTML脚本语法通过；未实际验证浏览器同步播放。人工逐帧0/1/2为空，可导出CSV/JSON；助手只看固定f5，数据技术分与模型效果粗分独立。

完整run：`/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929/r8`。`evaluation/`保存45窗原生PNG/局部写回；`review/`保存HTML/视频/十帧图；`data_review/`保存68候选全帧；`assets/`为已有DEV形状投影模板，非新生成器。

新数据权重：`training/attention_step_0160.safetensors`，同目录有40/80/120/160快照及优化器。有效r7：同task的`r7/encoder_fixed_lowres/training/attention_step_0160.safetensors`。原模型：`/root/autodl-tmp/external/worldsim_v75_downstream_bench/DriveEditor/checkpoints/model.safetensors`。新训练从原权重+官方106encoder恢复开始，不延续r7。

`dataset_catalog.json`冻结50train/25scene、7val/4scene；`evaluation_plan.json`冻结15case/三臂。原模型三个旧真实窗口仅在RGB/H/窗口/seed/步数/previous条件相同后复用，不复用r6错误训练权重。所有真实例曝光DEV、无去车GT；final未使用。

r8相对有效r7洞内scene等权MAE仅−1.05%，保护车MAE+4.78%；固定f5未见明确真实迁移收益。本轮不推广r8，也不据此否定Target+Protected路线或证明模块错误。见[报告与组件图](../../../../v77/TARGET_PROTECTED_DATA_CONTROL_R8.md)、[收口](closeout.json)、[输入分布](input_distribution.json)。轻量Git记录不是完整备份。
''')
    print('R8_CLOSEOUT',closeout['stage'])

if __name__=='__main__':main()

"""同一失败卡记录r45中止与r46修复；正式结论仅在实际结果齐全时写入。"""
from common import *
import argparse, shutil
from importlib.metadata import version


def main(finish=False):
    assert O.name=='r46'
    audit=read(O/'time_fix_audit.json');plan=read(O/'manifest.json')
    E=REPO/'docs/autoresearch/worldsim_v77/target_protected_20260929/r46';E.mkdir(parents=True,exist_ok=True)
    for name in ['time_fix_audit.json','correction_plan.json','preflight.json','condition_inventory.json']:
        shutil.copy2(O/name,E/name)
    first=[]
    for cid in ['A022','A013','A007']:
        c=next(c for c in plan['cases'] if c['case_id']==cid)
        a=read(T/'r45/conditions'/cid/'result.json');b=read(O/'conditions'/cid/'result.json')
        first.append({'case_id':cid,'camera_minus_sample_ms':(c['frames'][0]['timestamp']-c['frames'][0]['sample_timestamp'])/1000,
            'old_actors':next(x for x in audit['cases'] if x['case_id']==cid)['frames'][0]['old_actors'],
            'new_actors':len(c['frames'][0]['actors']),'old_hole':a['statistics'][0],'new_hole':b['statistics'][0],
            'old_lidar_sources':len(a['lidar']),'new_lidar_sources':len(b['lidar'])})
    dump(E/'first_frame_regression.json',{'cases':first,'SDK_version':version('nuscenes-devkit'),'SDK_method':'NuScenes.get_boxes'})
    qa=read(O/'visual_review.json') if finish else {}
    result=read(O/'results_summary.json') if finish else {}
    if finish:
        state=read(O/'training/state.json');idle=read(O/'gpu_finished.json')
        assert state['stage']=='complete' and state['steps']==160 and idle['gpu_jobs_remaining']==0
        assert len(qa['cases'])==12 and read(O/'gpu_delivery_validation.json')['decoded_frames']==960
        for source,dst in [('training/state.json','training_state.json'),('training/config.json','training_config.json'),
            ('training/zero_init.json','zero_init.json'),('training/steps.jsonl','training_steps.jsonl'),
            ('results_summary.json','results_summary.json'),('visual_review.json','visual_review.json'),
            ('gpu_delivery_validation.json','gpu_delivery_validation.json'),('gpu_finished.json','gpu_finished.json')]:
            shutil.copy2(O/source,E/dst)
    run={k:v for k,v in plan.items() if k!='cases'}
    run.update(phase='GPU_complete_stopped' if finish else 'SDK_time_fix_retraining',
        cases=[{k:c[k] for k in ['case_id','split','scene','type']} for c in plan['cases']],
        changed_train_cases=audit['changed_train_cases'],new_inference_windows=25 if finish else 0,
        reused_original_windows=11,comparison_windows=36 if finish else 0,
        failure_ledger_refs=['V77-F02'],failure_ledger_delta='updated V77-F02: keyframe time semantics and downstream corrected experiment',
        human_verdict=None,result=qa.get('conclusion'))
    dump(E/'run.json',run)
    report='''# r46：修复相机关键帧条件时间，再验证小Adapter

task `WS-V77-TARGET-PROTECTED-20260929/r46`，parent r45，failure refs `V77-F02`。

```mermaid
flowchart LR
 S[sample_data与关联标注] --> K[官方get_boxes · 关键帧直接取框 / 中间帧插值]
 K --> C[保留车O / 实测背景N / 未知U / 可信度Q]
 C --> A[157888参数Adapter]
 X[固定RGB与mask] --> D[冻结DriveEditor]
 A --> D
 Y[合成真实Y · 只监督] -. loss .-> A
 D --> V[原模型 / 全未知 / 实际条件 · 固定DELETE对照]
```

## 工程证据

旧路径只按相机时间查选定轨迹的插值范围，关键帧稍早于第一份sample标注时返回空。
相机sample_data关键帧有自己的关联sample；官方SDK直接取该sample全部标注，非关键帧在当前/前一个sample之间插值并处理新出现的实例。
本轮直接调用安装版本SDK的 `NuScenes.get_boxes()`；只加载所需记录，补齐SDK初始化时会生成的sample→anns反向索引。

旧r45训练已160步，但评测因工程错误停在32/36窗；所有旧条件、权重、原生输出和日志保留。
不再将其条件对照作为完整条件下的负结果。

| case | 相机减标注时间(ms) | 旧/新首帧actor数 | 旧/新LiDAR来源数 | 新首帧洞内O/N/U像素 |
|---|---:|---:|---:|---|
'''
    report+='\n'.join(f"| {c['case_id']} | {c['camera_minus_sample_ms']:.3f} | {c['old_actors']}/{c['new_actors']} | {c['old_lidar_sources']}/{c['new_lidar_sources']} | {c['new_hole']['O_H']} / {c['new_hole']['N_H']} / {c['new_hole']['U_H']} |" for c in first)
    report+='''

A022首帧恢复了画面其他区域的背景返回，但其删除洞仍全部U；这是洞内没有可靠返回的证据边界，不是再次丢失关键帧标注。不能为了变蓝而将未知刷成背景。

## 范围与验证

24例240帧按同一SDK规则重建。逐张比较O/N/U/Q：11/12训练例发生变化，10/12评测例发生变化。
因此从原始DriveEditor重训160步，只更新相同Adapter，保持数据case、RGB、H、alpha、相机、seed、优化器与预算。
“训练无全灰帧”只排除了整帧缺失，不能证明训练条件正确，故不复用旧Adapter当修复后权重。

CPU条件排他覆盖、Q未知为0、隐藏X/Y不进入图像条件等240帧合同通过。
三例实际首帧回归恢复关联标注并排除错误全画面U，正向LiDAR来源1→2。
11个已完成原模型关闭分支窗口不依赖条件，也不依赖新Adapter参数；逐项核对输入后复用。
其余25窗重新推理，总对照仍为12例×3臂36窗，seed42/25steps/10帧。所有case是已曝光DEV，人工判定为空。
'''
    if finish:
        report+='\n## 修复后的结果\n\n'+qa['conclusion']+'\n\n'
        report+='| 合成case | 原模型洞MAE | 全未知洞MAE | 实际条件洞MAE |\n|---|---:|---:|---:|\n'
        report+='\n'.join(f"| {c['case_id']} | {c['metrics']['adapter_off']['hole_MAE']:.6f} | {c['metrics']['all_unknown']['hole_MAE']:.6f} | {c['metrics']['conditioned']['hole_MAE']:.6f} |" for c in result['cases'] if c['kind']=='synthetic')
        protected_cases=[c['case_id'] for c in result['cases'] if c['kind']=='synthetic' and c['metrics']['adapter_off']['protected_hole_MAE'] is not None]
        report+='\n\n4个合成case等权均值：`'+str(result['synthetic_case_equal_MAE'])+'`；保护区域仅计'+str(len(protected_cases))+'例（'+', '.join(protected_cases)+'），case等权均值：`'+str(result['synthetic_protected_case_equal_MAE'])+'`。合成GT指标不能替代真实DELETE人工通过率。\n\n'
        changes=[c['output_change_checks']['conditioned_vs_all_unknown']['hole_rgb_MAE'] for c in result['cases']]
        report+=f'实际条件与全未知的原生输出在{sum(v>0 for v in changes)}/{len(changes)}例不完全相同，洞内RGB差异MAE范围{min(changes):.6f}–{max(changes):.6f}（0–1）。实际网络零初始化等价、分支梯度与主干冻结检查通过；这些只排除条件通路完全未接或输出完全不变，不能证明模型已充分利用条件。此处差异不是对真实GT的恢复误差。\n\n'
        report+=qa['review_scope']+'\n\n| case | 助手观察（非人工verdict） |\n|---|---|\n'
        report+='\n'.join(f"| {c['case_id']} | {c['observation']} |" for c in qa['cases'])
        report+='\n\n新知识：'+qa['new_knowledge']+'\n\n96视频960帧实际解码，保留原生和写回输出。GPU进程退出，按最新要求停下通知；没有追加一轮或自动关机。\n'
    else:
        report+='\n当前修复与CPU回归完成；修复后的训练/推理尚未完成，模型收益待验证。\n'
    report+='\n代码 `scripts/worldsim_v77/target_protected/iteration13/`；通过 `V77_ONUQ_RUN_ID=r46` 选择本轮路径。旧r45不被覆盖。\n本地结果入口 `outputs/v77-onuq-r46/index.html`。failure_ledger_delta: updated V77-F02。\n'
    (REPO/'docs/v77/TARGET_PROTECTED_ONUQ_TIME_FIX_R46.md').write_text(report)
    status='''# 当前研究状态

更新：2026-10-03。分支 `research/worldsim-v7.7-target-protected-editing`，主机 wm-3090-1001。

用户要求先修关键帧时间边界bug，再做小条件分支实验。当前唯一run `WS-V77-TARGET-PROTECTED-20260929/r46`。
统一直接使用官方NuScenes.get_boxes：关键帧取关联sample，非关键帧前后sample插值。旧r45保留160步权重、全部条件及32窗结果，因条件工程错误中止，不能作为完整条件的负结果。

24例240帧重建与CPU合同通过；11/12训练条件、10/12评测条件实际变化，因此同预算重训160步、主干冻结。复用11个不依赖条件的原模型窗口，其余25窗重跑，总比较36窗。没有扩数据/结构/训练步数，人工verdict空，全部是DEV。
'''
    status+=('\n'+qa['conclusion']+'\n\nGPU训练/推理已结束，无后续自动实验；按最新要求停止并通知，未自动关机。本地审核 `outputs/v77-onuq-r46/index.html`。\n' if finish else '\n修复后训练正在运行；尚无新收益结论。GPU结束就停下通知，不自动开新一轮。\n')
    status+='\n细节与组件图见 [r46报告](v77/TARGET_PROTECTED_ONUQ_TIME_FIX_R46.md)，沿用 [V77-F02](research_failures/entries/V77-F02.md)。N仍稀疏，A022首帧洞内无背景返回仍应U。\n'
    (REPO/'docs/RESEARCH_STATUS.md').write_text(status)
    p=REPO/'docs/EXPERIMENTS.md';lines=p.read_text().splitlines()
    lines=[('| WS-V77-TARGET-PROTECTED-20260929 / r45 | 160步、32窗；时间语义bug中止，保留旧结果 | [报告](v77/TARGET_PROTECTED_ONUQ_ADAPTER_R45.md) |' if '20260929 / r45' in s else s) for s in lines]
    lines=[s for s in lines if '20260929 / r46' not in s]
    i=next(i for i,s in enumerate(lines) if s.startswith('|---'))
    lines.insert(i+1,'| WS-V77-TARGET-PROTECTED-20260929 / r46 | SDK时间修复；'+('160步、25新窗＋11复用，GPU已停' if finish else '240帧CPU回归通过，重训中')+' | [报告](v77/TARGET_PROTECTED_ONUQ_TIME_FIX_R46.md) |')
    p.write_text('\n'.join(lines)+'\n')
    p=REPO/'docs/research_failures/entries/V77-F02.md';body=p.read_text();marker='\n## r46：相机关键帧时间语义修复\n'
    if marker in body:body=body[:body.index(marker)]
    body+=marker+'\n旧时间插值在A022/A013/A007首帧丢弃已有关键帧标注，首个LiDAR来源被跳过。已统一接官方get_boxes。没有全灰训练帧不代表训练条件正确，实际11/12训练和10/12评测条件变化；同160步重训，旧r45的32窗不作完整条件负结果。A022修复后首帧洞内仍U是缺返回边界，不伪造N。\n\n'
    body+=(qa['conclusion']+'\n\n'+qa['new_knowledge']+'\n\n' if finish else '修复后240帧CPU合同与三例首帧回归通过，模型结果待实际GPU验证。\n\n')
    body+='[证据与组件图](../../v77/TARGET_PROTECTED_ONUQ_TIME_FIX_R46.md)。failure_ledger_delta: updated V77-F02。\n'
    p.write_text(body)
    print('R46_RECORDED','complete' if finish else 'running')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--finish',action='store_true');a=p.parse_args();main(a.finish)

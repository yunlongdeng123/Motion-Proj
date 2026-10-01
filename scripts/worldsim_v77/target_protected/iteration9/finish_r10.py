"""有限过程试验交付收口；不追加训练或执行电源操作。"""
from pathlib import Path
import json,shutil
from collections import defaultdict
import numpy as np

P=Path('/root/autodl-tmp/motion_proj_v77')
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')
O=T/'r10';E=P/'docs/autoresearch/worldsim_v77/target_protected_20260929/r10'
def read(p):return json.loads(p.read_text())
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def f(v):return 'N/A' if v is None else f'{v:.5f}'
def change(v):return 'N/A' if v is None else f'{v:+.2%}'

def main():
    s=read(O/'results_summary.json');delivery=read(O/'delivery_validation.json');reviews=read(O/'assistant_effect_reviews.json');decision=read(O/'decision.json')
    state=read(O/'evaluation/state.json');assert state['all_four_arms_complete'] and state['completed_count']==76
    assert s['same_recipe']['all_fixed_recipe_fields_equal'] and s['training']['steps']==160
    assert delivery['success'] and delivery['videos_checked']==278 and delivery['len_images_checked']==2020
    assert len(reviews['cases'])==19 and all(c['human_verdict'] is None for c in reviews['cases'])
    assert read(O/'controller_state.json')['stage']=='complete'
    qa=read(T/'r9/independent_data_reviews.json');census=read(O/'training_process_census.json');dist=read(O/'process_distribution.json');evidence=read(O/'evidence_conditioned_metrics.json')
    close={'task':'WS-V77-TARGET-PROTECTED-20260929','run':'r10','host':'wm-3090-1001','stage':'complete_training_evaluation_review_delivery',
        'data':s['data'],'training':s['training'],'same_recipe':s['same_recipe'],'census':{k:v for k,v in census.items() if k not in ['cases','sampled_case_counts']},
        'evaluation':{'cases':19,'arms':['base','r7','r8','r10'],'windows':76,'fresh':31,'exact_input_reuse':45,'frames_per_window':10,'seed':42,'steps':25,'real_actor_free_GT':None},
        'decision':decision,'delivery':delivery,'human_verdict':None,'final_used':False,'power_operation':None,'new_automation':None,
        'failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'updated V77-F02: actual temporal coverage gap, bounded factory shortage, limited process transfer result; no new ID',
        'full_remote_run':str(O),'lightweight_git_evidence_is_full_backup':False,
        'preserved':['r7/r8/all original weights and scores','all candidate inputs and rejected placements','native and written-back frames','160-step checkpoints and optimizer snapshots','pre-fix labels and independent prior scores'],
        'limitations':['new sweep validation only one independent world','dense training only one world','one-second windows','GT cuboid geometric evidence proxy is not texture correspondence','Boston daytime, two exposed mesh shapes, official pretraining overlap unknown','data process and background/protected type proportions changed together','real exposed DEV without actor-free GT; fixed frame reviews do not certify video temporal quality']}
    dump(O/'closeout.json',close)
    for root,stage,steps in [(T/'r9','closed_bounded_factory_coverage_shortage',0),(O,close['stage'],160)]:
        registered=read(root/'run.json');backup=root/'run_initial_registration.json'
        if not backup.exists():dump(backup,registered)
        registered.update(stage=stage,actual_training_steps=steps,failure_ledger_delta='updated V77-F02; no new ID')
        dump(root/'run.json',registered)
        dump(P/'docs/autoresearch/worldsim_v77/target_protected_20260929'/root.name/'run.json',registered)
    report='''# v77：运动—遮挡—显露过程控制 r9/r10

task `WS-V77-TARGET-PROTECTED-20260929`，r9数据盘点与工厂、r10有限过程微调；默认主机wm-3090-1001，分支v77。failure_ledger_refs [V77-F02]。人工结论为空，未使用final，原/r7/r8和所有反例保留。

## 本轮结论

'''+decision['summary']+'''

```mermaid
flowchart LR
 BAD[真实DELETE badcase] --> AUD[运动/遮挡/显露盘点]
 OLD[r7/r8实际训练窗口] --> AUD
 AUD --> G[固定世界轨迹A + 真实Y/B]
 G --> Q[全帧空间/遮罩 + 独立AI2]
 Q --> D[新过程 + 稳定控制 50train]
 D --> FT[DriveEditor原结构 80张量160步]
 Y[真实nuScenes Y 只作监督] --> FT
 FT --> E[原/r7/r8/r10 同输入类型评测]
 E --> HTML[同步视频 + 逐帧人工打分]
```

## 缺口是否实际存在

| 实际训练窗口 | 例/世界 | 静止A＋运动ego＋图像变化 | 遮挡扫过B | 显露变化 |
|---|---:|---:|---:|---:|
| r7 | 68 / 9 | 0 | 0 | 1 |
| r8 | 50 / 25 | 12 | 0 | 4 |
| r10 | 50 / 25 | {static} | {sweep} | {visibility} |

上述过程可以重叠，选择类别并非实际过程计数。世界静止直径≤0.20m、ego路径≥0.50m、图像中心变化≥8px；B归一化遮挡中心跨度≥0.20、至少3帧有效重叠；显露变化为B遮挡比例范围≥0.15。是可复核的操作判据，不是完整过程语义。

70个已曝光真实审计例中56个有完整目标几何，14个缺track，不填运动类别；56例中13个静止A＋运动ego且图像变化，28个后车GT包络扫过代理。真实B车身遮挡/隐藏纹理未逐实例认证，不能把包络代理当SAM2像素证据。真实全段约3秒与训练一秒分母分开；固定8例DEV另用同10帧窗口盘点。A022固定窗ego几乎静止、A明显运动，时序缺口不能统一解释所有失败。

## 数据与质量

工厂改为世界静止或固定世界速度轨迹，不再随B逐帧加固定偏移。真实相机、GT朝向、地图、LiDAR地面、0.3m间距、底部ego禁入、A前于活跃B、完整合成影响先擦除后缩放等门槛保留。Y来自真实nuScenes；模型看到遮洞条件，不看到完整synthetic-X车漆。

r9先复用旧合法锚点，再一次冻结16来源扩展；原8train/3val/2val-world扫过目标未达，按预案训练0步。未无限扩大位置网格。选22例全220帧重读X/Y/H并复查三次独立静态扫描；稀疏静态扫描零正证据不能认证空地。

独立gpt-6-sol xhigh、无fast检查0/5/9：18通过、2不确定、2拒绝。P014人行区、P022跨双黄线拒绝；P002/P007视觉扫过证据不足未训练。4例旧B/C标签与实际活跃实例错配，修显示标签后定向复核通过，Y/X/H/几何未改，原标签和先前分数保留。技术AI2不等于人工通过。

r10另行预注册有限pilot，50train/25world、最多3例/scene；20背景/27单保护/3密集。选择类别4扫过（3world）、4显露、5静止A+ego、37稳定控制。新过程验证4例/3world，其中扫过仅1world；旧合成7例/4world、真实8例DEV全部冻结。新增验证world未进入r7/r8/r10训练，预训练重叠未知。密集train仍只1world，两种已曝光DEV形状、Boston白天、一秒窗口不变。

实际160步采样：{step_counts}。扫过4例分别采样3/3/3/4次；频率不是梯度贡献。新增过程与背景/保护类别比例同时变化，不能把结果解释为单一因素的因果证明。

## 同预算四臂评测

80空间attention张量、49,574,080参数、320×576、160步、AdamW 1e-5/0.01、seed6201，官方原StandardDiffusionLoss，严格恢复官方106目标encoder，0missing/0unexpected；80有限非零梯度。新权重从原模型开始，不继续训练r8。10帧输入、原网络结构不改，不同时加权loss或扩模块。训练峰值分配10.48GiB、保留11.05GiB。

19例×4权重=76窗，31新采样，45旧PNG逐帧核对X/H/Y及seed42/25steps/previous=false后复用。采样576×1024。同一原模型Engine重新加载各臂，恰好替换80键，避免权重污染。报告原生输出MAE；局部写回只恢复原范围，洞外原像素保持不能当神经保护能力。

| 组别 / 指标（scene等权MAE，越低越好） | 原 | r7 | r8 | r10 | r10对r8 | case / scene |
|---|---:|---:|---:|---:|---:|---:|
'''
    flags=census['actual_case_process_flags'];report=report.replace('{static}',str(flags['static_A_moving_ego_image_change'])).replace('{sweep}',str(flags['sweep_over_any_B'])).replace('{visibility}',str(flags['visibility_transition_any_B'])).replace('{step_counts}',str(census['actual_step_family_counts']))
    for name,g in s['groups'].items():
        for key,v in g.items():report+='| '+name+' / '+key+' | '+' | '.join(f(v['macro_scene'][a]) for a in ['base','r7','r8','r10'])+' | '+change(v['r10_vs_r8']['scene_relative_change'])+f' | {v["cases"]} / {v["scenes"]} |\n'
    report+='\n| 新过程 / 指标 | 原 | r7 | r8 | r10 | case / scene |\n|---|---:|---:|---:|---:|---:|\n'
    for name,g in s['process_groups'].items():
        if name=='r8_validation_control':continue
        for key,v in g.items():report+='| '+name+' / '+key+' | '+' | '.join(f(v['macro_scene'][a]) for a in ['base','r7','r8','r10'])+f' | {v["cases"]} / {v["scenes"]} |\n'
    report+='''
旧合成、新过程分别统计，不合并掩盖退化。单一场景的扫过胜负不能推到充分泛化。像素恢复误差不是车辆身份或视频通过率；真实DEV没有actor-free GT。

## 其他帧证据与剩余分布差异

按真实Y的SAM2 B像素与定向GT cuboid16格交点，分别保存其他帧几何支持、没有支持、无法对应的洞内像素及误差；不是准确纹理对应。纯背景证据未测量，真实隐藏纹理未知，不把未知写为“从没见过”。

{evidence_table}

P006的f5明确在真实GT道路上生成额外金/棕色车，整10帧洞内MAE对r8上升131.66%、洞内B上升55.47%。这不是把未知隐藏车存在误判为目标再生，而是已知Y的额外实体反例。其65.5%洞内B像素有其他帧GT框格几何支持，但该支持区域误差也上升；不能只用“所有帧完全不可见”解释此例，准确纹理对应仍未认证。P012像素误差略降，固定f5仍有合并车身，量化小收益不认证两车身份恢复。

| 分布 | r8 train | r10 train | 固定真实DEV8 |
|---|---:|---:|---:|
'''
    ev_table='| 新过程例 | 洞内B其他帧几何支持像素比例 | 支持区MAE r8→r10 | 无支持区MAE r8→r10 |\n|---|---:|---:|---:|\n'
    for c in evidence['cases']:
        if c['suite']!='new_temporal_validation':continue
        total=sum(r['protected_hidden'] for r in c['frames']);support=sum(r['geometric_other_frame_support'] for r in c['frames'])
        ev_table+='| '+c['eval_id']+f' | {support/total:.1%} | '+f(c['arms']['r8']['geometric_other_frame_support'])+'→'+f(c['arms']['r10']['geometric_other_frame_support'])+' | '+f(c['arms']['r8']['no_geometric_other_frame_support'])+'→'+f(c['arms']['r10']['no_geometric_other_frame_support'])+' |\n'
    report=report.replace('{evidence_table}',ev_table)
    ds=dist['summary'];rows=[('洞面积均值','mean_hole_canvas_fraction'),('触边帧比例','mean_border_frame_fraction')]
    for label,key in rows:report+='| '+label+' | '+' | '.join(f'{ds[k][key]:.2%}' for k in ['r8','r10','real_fixed_DEV8'])+' |\n'
    for label,key in [('洞中心路径速度中位数(px/s)','hole_centroid_speed_proxy_pxps'),('洞尺度max/min中位数','area_max_min_ratio'),('洞框填充率中位数','hole_bbox_fill')]:report+='| '+label+' | '+' | '.join(f'{ds[k]["quantiles"][key]["median"]:.3f}' for k in ['r8','r10','real_fixed_DEV8'])+' |\n'
    report+='\n'+'\n'.join('- '+x for x in decision['next_control_requirements'])+'\n\n'+decision['interpretation_limits']+'\n'
    report+='''
## 审核与工程记录

[19例六列模型视频/逐帧打分](C:/Users/dengyunlong/Documents/Codex/2026-09-26/xia/outputs/v77-target-protected-r10/index.html) · [22候选四列数据全帧](C:/Users/dengyunlong/Documents/Codex/2026-09-26/xia/outputs/v77-target-protected-r10/data_review.html)。每例原生视频另链。人工0/1/2全空，助手效果观察仅固定f5，不认证时序。

实际278视频/2780帧解码、2020引用JPG及HTML脚本语法验证通过；浏览器同步播放未实测。4项过程语义测试通过、代码编译通过。期间统计脚本的NumPy int64 JSON序列化错误已修复，未改变数据或训练；训练/推理无额外重跑。

完整run保留在`/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929/r10`；新权重`training/attention_step_0160.safetensors`，所有40/80/120/160和优化器快照保留。Git轻量JSON不是完整资产备份。当前资源未要求删数据或扩盘；无关机、定时或追加训练操作。

failure_ledger_delta: updated V77-F02；没有新增ID。原/r7/r8仍保留，本轮结果不改变用户历史评分。
'''
    (P/'docs/v77/TARGET_PROTECTED_TEMPORAL_CONTROL_R10.md').write_text(report)
    (O/'report.md').write_text(report);shutil.copy2(O/'report.md',O/'review/report.md')
    status=f'''# 当前研究状态

2026-10-01，v77；默认wm-3090-1001。`WS-V77-TARGET-PROTECTED-20260929/r9`过程盘点/有界工厂收口（覆盖缺额、训练0）；`r10`有限过程同80张量/160步训练、四臂76窗、19例模型与22候选HTML交付完成。不重复启动，final未用，无电源/定时操作。

{decision['summary']}

实际r7/r8均无达到阈值的B扫过；r8已有12个静止A＋运动ego。r10训练50/25world，实际11静止A＋ego、4扫过、12显露（可重叠）；新扫过val仅1world、dense train仅1world、一秒窗口。其他帧证据仅GT框表面几何代理，真实隐藏GT未知，不把统计等同因果或视频通过率。

下一有界控制：{decision['next_control']} 尚未启动；保留全部输入/拒绝/旧基线/新权重。参见[报告与组件图](v77/TARGET_PROTECTED_TEMPORAL_CONTROL_R10.md)、[指标](autoresearch/worldsim_v77/target_protected_20260929/r10/results_summary.json)、[审核](autoresearch/worldsim_v77/target_protected_20260929/r10/review_link.md)。failure_ledger_refs: [V77-F02]。
'''
    (P/'docs/RESEARCH_STATUS.md').write_text(status)
    idx=P/'docs/EXPERIMENTS.md';lines=idx.read_text().splitlines()
    for i,l in enumerate(lines):
        if l.startswith('| WS-V77-TARGET-PROTECTED-20260929 / r10 |'):lines[i]='| WS-V77-TARGET-PROTECTED-20260929 / r10 | 50train/25world，4扫过/3world，同80张量160步；19例四臂76窗与278视频交付；有限过程结果不推广 | [报告](v77/TARGET_PROTECTED_TEMPORAL_CONTROL_R10.md) · [结果](autoresearch/worldsim_v77/target_protected_20260929/r10/results_summary.json) · [审核](autoresearch/worldsim_v77/target_protected_20260929/r10/review_link.md) |'
    idx.write_text('\n'.join(lines).rstrip()+'\n')
    failure=P/'docs/research_failures/entries/V77-F02.md';text=failure.read_text();marker='## r10：有限遮挡过程同预算迁移控制'
    assert marker not in text,'closeout already registered; do not append twice'
    old=s['groups']['r8_frozen_synthetic']['protected_inside_hole'];new=s['groups']['new_temporal_validation']['protected_inside_hole']
    text+='\n\n'+marker+'\n\n'+decision['summary']+'\n\n'+f'保持原架构/80空间张量/160步/原loss；50train/25world，实际4扫过/3world，13训练步采样，新扫过val仅1world。四臂76窗完成，旧GT保护车r10对r8 {change(old["r10_vs_r8"]["scene_relative_change"])}，新过程GT保护车 {change(new["r10_vs_r8"]["scene_relative_change"])}，两组分开报告。真实无去车GT，固定f5观察不是视频通过率。\n\n'+decision['interpretation_limits']+'\n\n'+f'下一控制尚未执行：{decision["next_control"]}。洞均值1.82%对真实5.60%、零触边对48.75%、投影中心速度与尺度变化差异仍开放，不宣布模块错误。278视频实际解码与2020引用帧通过，人工留空，全部反例/原权重保留。[报告](../../v77/TARGET_PROTECTED_TEMPORAL_CONTROL_R10.md)。failure_ledger_delta: updated V77-F02，不新增ID。\n'
    failure.write_text(text)
    controller=read(O/'controller_state.json');controller.update(stage=close['stage'],local_delivery_verified=True,followup_training_started=False,human_verdict=None);dump(O/'controller_state.json',controller)
    for name in ['results_summary.json','closeout.json','decision.json','assistant_effect_reviews.json','delivery_validation.json','training_process_census.json','process_distribution.json','evidence_conditioned_metrics.json','controller_state.json']:
        dump(E/name,read(O/name))
    dump(E/'evaluation_state.json',state)
    (E/'review_link.md').write_text('''# r10 审核入口

[19例六列模型视频](C:/Users/dengyunlong/Documents/Codex/2026-09-26/xia/outputs/v77-target-protected-r10/index.html) · [22候选四列数据](C:/Users/dengyunlong/Documents/Codex/2026-09-26/xia/outputs/v77-target-protected-r10/data_review.html)

逐帧人工0/1/2为空，可导出CSV/JSON；19例固定f5助手观察与独立输入QA分开。完整run保留在`/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929/r10`；新权重`training/attention_step_0160.safetensors`，不覆盖原/r7/r8。全部原生PNG/局部写回、拒绝/待定和优化器保存。278视频实际解码、2020引用图片与JS语法通过；未实测浏览器同步播放。GT表面格是几何对应代理、真实隐藏GT未知，扫过验证只有1个world；轻量Git索引不是资产备份。见[报告与图](../../../../v77/TARGET_PROTECTED_TEMPORAL_CONTROL_R10.md)。
''')
    print('R10_CLOSEOUT',decision['summary'])

if __name__=='__main__':main()

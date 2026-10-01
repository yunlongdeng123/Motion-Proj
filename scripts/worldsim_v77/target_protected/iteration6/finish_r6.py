"""按文件职责收口r6，归档轻量证据，保留完整外部数据与权重。"""
from pathlib import Path
import argparse,json,shutil
REPO=Path('/root/autodl-tmp/motion_proj_v77')
def read(p):return json.loads(p.read_text())
def main(root):
    s=read(root/'pilot_summary.json');v=read(root/'delivery_validation.json');e=read(root/'evaluation/state.json')
    assert s['training']['steps']==160 and e['stage']=='complete' and len(e['completed'])==18
    assert v['actual_decoded_videos']==215 and v['actual_decoded_frames']==2750
    assert any(r['eval_id']=='A041_w10' for r in s['assistant_review']['cases'])
    dest=REPO/'docs/autoresearch/worldsim_v77/target_protected_20260929/r6';dest.mkdir(parents=True,exist_ok=True)
    items=['run_config.json','dataset_catalog.json','exact_admission.json','exact_mask_diagnostics.json','new_synthetic_reviews.json','assistant_effect_reviews.json','evaluation_input_audit.json','evaluation_plan_pre_input_fix.json','evaluation_plan.json','pilot_summary.json','delivery_validation.json']
    for name in items:shutil.copy2(root/name,dest/name)
    for name in ['config.json','state.json','input_contract.json','backward_probe.json','validation_base.json','validation_finetuned.json','checkpoint_manifest.json']:
        shutil.copy2(root/'training'/name,dest/('training_'+name))
    shutil.copy2(root/'evaluation/state.json',dest/'evaluation_state.json')
    decision={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r6','state':'complete_bounded_pilot','assistant_data_score':2,'human_verdict':None,'data_cases':52,'train_cases':48,'validation_cases':4,'training_steps':160,'evaluation_tasks':8,'evaluation_windows_including_retained_empty':9,'neural_sampling_windows':18,'model_promoted':False,'default_checkpoint':'original DriveEditor','failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'updated V77-F02: true mask gate attrition, loss/output quality mismatch, empty-window evaluation guard','next':'first exclude train/inference resolution mismatch via one fixed native-resolution control; not executed in r6','power_action_authorized_this_round':False}
    (root/'closeout.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n');shutil.copy2(root/'closeout.json',dest/'closeout.json')
    report=f'''# Target + Protected Actors r6：技术准入、原结构微调与固定效果对照

2026-10-01；`WS-V77-TARGET-PROTECTED-20260929/r6`。完成有界160步微调；**此配方未获稳定DELETE收益，原DriveEditor仍为默认**。数据AI技术2分和模型效果评分分开，人工verdict全部null。

```mermaid
flowchart LR
    Y[真实 nuScenes train Y] --> Q[五项几何与mask质检]
    A[真实供体与米制轨迹 A] --> Q
    Q --> X[合成X与精确洞H]
    X --> C[先清零X再缩放的条件]
    Y --> T[真实恢复目标与含噪latent]
    C --> D[原DriveEditor 9通道结构]
    T --> D
    D --> W[更新现有80个空间attention张量]
    W --> V[同mask seed窗口的原与微调视频对照]
```

## 数据与身份准入

SAM2新分割4个保护实例，独立gpt-6-sol xhigh非fast检查通过；实际像素遮挡门槛只允许C012，其余C010/C014/C018因`single_visibility_ratio`退出。C010实际遮挡9–12%，证明GT包络预案不能当实例像素准入。阈值、位姿和来源未因结果改动。C012实际10帧合成通过工程验证及独立抽帧2分。

按照用户2026-10-01最新授权，以技术AI2分准入：r2先前五项重评27例+r3先前24例+本轮C012一例，共**52case / 10个receiver scene**，并非52场景或全部新造。29纯背景、23单保护车，密集多车仍0。拒绝、待定、D006/D009空间硬失败与原分数保留。720帧重新解码，X-Y影响都在H内、masked-X=masked-Y。独立视觉仍是抽帧，工程是全帧，不宣称人工全检。

receiver与donor scene入图，整连通分量分割48train / 4validation，双来源scene均无跨集合重叠。4个留出case共享一个receiver scene，属于有限开发验证。过去70例val只作已曝光真实DELETE开发对照，final未用。

## 实际训练合同与资源

完整synthetic-X不送条件；先在原尺寸清零H，再缩放清零后的浮点条件。Y只作为扩散恢复目标及标准含噪target latent，不作为干净条件。实际合同及4个新回归测试验证：改洞内X不改任何batch字段，改洞内Y只改jpg目标，可见上下文进入条件，洞外合成影响被拒绝。protected标签仅作准入/评价，不增加输入通道。

原官方完整已训练`model.safetensors`初始化，CLIP由此恢复；**r7核对训练日志发现106个目标encoder权重缺失，0 missing只属于推理加载，旧训练不能用于数据/模块归因**。沿用原StandardDiffusionLoss。AdamW 1e-5、weight_decay .01、梯度clip1、seed6201，batch1、10帧、68训练窗口、160步。仅更新主分支空间self-attention Q/K/V/out的80张量、49,574,080参数，其余权重冻结；不是官方8卡全参数训练或LoRA。冻结权重bf16、训练参数fp32，nonreentrant checkpoint只改计算重算API，架构/公式不变。

训练320×576，推理576×1024；这个差异明确保留，尚未做原分辨率控制，不能称为已证实失败原因。真实反向80张量有有限非零梯度，训练峰值10.48GiB，训练与验证阶段约534秒（不含提取/初始化），生成峰值约21.86GiB。权重合并4077键保持一致，只有80张量更新，完整权重约12.06GB；原权重不改写。

CLIP train配置试图联网初始化的失败已保留，随后用和本地sample一致的version=null及完整checkpoint恢复解决。CPU编码使用已有imageio-ffmpeg；这些是工程错误，不是模型科学否定。

## 效果与分母

固定留出4case×2个噪声draw：latent去噪loss **{s['validation_loss_base']:.6f} → {s['validation_loss_finetuned']:.6f}（{s['validation_loss_change_percent']:.2f}%）**。然而洞内真实保护车平均MAE **{s['protected_MAE_base']:.5f} → {s['protected_MAE_finetuned']:.5f}（+{s['protected_MAE_change_percent']:.2f}%）**，4/4例变差；洞整体MAE3/4变差。每帧等权平均，不把洞外原像素直接写回算恢复收益。

四个合成例独立第5帧粗评原模型1分、微调0分：硬边有所软化，但真实后车主体更模糊。A022车形减少却变大片模糊；A013/A048真实后车仍受损。没有真实clean GT，不能用“出现车”本身判幻觉，抽帧也不认证时序。

A041首窗0–9帧的洞全为空。原助手该帧2分只说明原图未变，**不计DELETE成功**。入口检查后，仅按输入mask最早连续10帧非空规则选择10–19帧，原与微调同条件补跑；旧空窗和旧评分保留并注明N/A。8个有效任务对照+1个保留空窗，18次实际单窗采样（含2次空窗）；这次修正不是根据模型效果挑窗口。

全部对照采用seed42、25采样步、10Hz单秒窗口、同mask/alpha写回，原生整帧输出另留链接。没有训练集逐例参数或多套生成器。单场景留出变差不否定整个数据路线，但足以停止推广本轮权重。

## 工程与交付

8项训练合同/准入回归通过；完成状态拒绝重复训练、评测按已完成窗口恢复。训练增加未来optimizer/RNG快照入口；r6实际训练早于该后续infra补充，没有旧optimizer快照，不能声称已验证本轮中断恢复。完整权重及attention补丁都保留。静态链接与215段H264完整解码、2750帧通过；浏览器播放未检查，不绕过既有file URL工具限制。

本地`outputs/v77-target-protected-r6/index.html`：真实/GT、条件洞、原权重、微调四列，真实目标带冻结实例core包络黄框，人工分数留空，可导出CSV。`data_review.html`提供52例真实Y/遮洞条件/synthetic-X视频和准入依据。组件图、负结果、原生输出和空窗都保留。

远端root：`{root}`。完整微调权重：`training/model_step_0160.safetensors`；补丁：`training/attention_step_0160.safetensors`。本轮无电源操作授权，没有关机或定时任务。

下一轮先以同数据/同更新范围做训练与推理分辨率一致的有界控制，排除工程变量后再决定保护区域监督；密集多车与新场景覆盖仍需造数。当前未执行该控制，不靠重复同配方步数或低loss宣布有效。failure_ledger_refs: [V77-F02]；failure_ledger_delta: updated V77-F02。
'''
    (REPO/'docs/v77/TARGET_PROTECTED_FINETUNE_R6.md').write_text(report)
    (REPO/'docs/RESEARCH_STATUS.md').write_text(f'''# 当前研究状态

2026-10-01，v77；`WS-V77-TARGET-PROTECTED-20260929/r6`已完成有界数据准入与微调对照。用户授权GPU/CPU自主迭代及技术AI2分后训练，历史人工分保持null。52case/10个receiver scene，48train/4validation；29背景、23单保护车、密集0。新的4例精确mask仅C012通过，原阈值未放宽。

原DriveEditor结构不变，160步更新既有80个空间attention张量。latent loss0.544→0.165，但4个共享一个场景的合成留出例保护车MAE均变差（均值+{s['protected_MAE_change_percent']:.1f}%），视频变模糊；本轮权重不升为默认，原权重保留。A041空首窗不计删除成功，按输入非空规则补跑有效10–19帧，两权重相同条件，保留旧空窗。8个有效任务+1个空窗全部在本地r6报告，人工verdict留空；215视频/2750帧解码通过。

下一步优先原分辨率训练/推理一致性控制，同数据同更新范围，尚未执行；之后才判断保护区监督是否需要变化。密集类及来源场景多样性仍欠缺，不把这次单场景负结果否定整个路线。GPU当前允许；无本轮关机或自动化授权，作业结束不关机。

见[报告与组件图](v77/TARGET_PROTECTED_FINETUNE_R6.md)、[证据](autoresearch/worldsim_v77/target_protected_20260929/r6/pilot_summary.json)。run `{root}`。failure_ledger_refs: [V77-F02]；failure_ledger_delta: updated V77-F02。
''')
    index=REPO/'docs/EXPERIMENTS.md';text=index.read_text();row='| WS-V77-TARGET-PROTECTED-20260929 / r6 | 52技术AI2例/48train+4val；原结构80张量160步；4留出保护车MAE均变差，原模型保留；8有效任务+1空窗 | [报告](v77/TARGET_PROTECTED_FINETUNE_R6.md) · [证据](autoresearch/worldsim_v77/target_protected_20260929/r6/pilot_summary.json) |\n'
    if '/ r6 |' not in text:text=text.replace('|---|---|---|\n','|---|---|---|\n'+row,1);index.write_text(text)
    card=REPO/'docs/research_failures/entries/V77-F02.md';text=card.read_text()
    if '## r6：数据技术准入与微调负对照' not in text:
        text+=f'''\n## r6：数据技术准入与微调负对照

`WS-V77-TARGET-PROTECTED-20260929/r6`。真实SAM2 mask只让4个GT预案中的C012通过；C010实际遮挡9–12%，门槛30%，证明包络预案不能代替精确像素。未修改位姿或阈值。依用户最新AI2训练授权，已有独立五项技术候选51例+新C012形成52case/10场景，48train/4val，密集0，人工评分及拒绝/待定全部保留。

实际原结构80张量160步微调，全部梯度有限非零，训练峰值10.48GiB。隐含X像素不进入任何条件，真实Y作扩散监督；8项回归通过。固定latent loss降低69.58%，但4个同receiver场景的留出例洞内保护车MAE全变差，平均+{s['protected_MAE_change_percent']:.2f}%，抽帧真实后车更糊。真实A022/A013/A048仍失败或改变错误形态。保留原模型默认，微调权重不推广；不据此宣布整条Target+Protected路线无效。

A041原首窗全部空mask，画面未变不计DELETE成功；独立旧2分以图像观察保留、任务有效分N/A。仅按输入最早连续非空10帧规则补跑10–19帧，两权重同条件；8有效任务+1原空窗留证。训练/推理320×576与576×1024不同是未控制变量，尚未证明因果；后续先有界原分辨率控制，不无限重复同配方。

见[报告](../../v77/TARGET_PROTECTED_FINETUNE_R6.md)、[完整轻量证据](../../autoresearch/worldsim_v77/target_protected_20260929/r6/closeout.json)。failure_ledger_delta: updated V77-F02（精确输入门槛、loss与编辑质量错配、空窗评测入口），不新增失败ID。
''';card.write_text(text)
    print(json.dumps(decision,ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)

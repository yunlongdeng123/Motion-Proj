"""记录CPU停止点；不启动任何模型、不更改人工评分。"""
import argparse
import json
from pathlib import Path
import shutil


def main(root,repo):
    s=json.loads((root/'cpu_closeout.json').read_text())
    assert s['state']=='cpu_complete_awaiting_gpu'
    evidence=repo/'docs/autoresearch/worldsim_v77/target_protected_20260929'
    evidence.mkdir(parents=True,exist_ok=True)
    for name in ['cpu_closeout.json','segmentation_queue.json','source_geometry_validation.json',
                 'subagent_source_reviews.json','source_rgb_missing.json','delivery_validation.json']:
        # 这些是本轮生成的JSON证据；仓库副本统一LF，run内原件保留。
        (evidence/name).write_bytes((root/name).read_bytes().replace(b'\r\n',b'\n'))
    r=s['receiver'];d=s['donor'];inv=s['inventory']
    report=f'''# v77 Target + Protected Actors：CPU素材准备与GPU停止点

`{s['task_id']}/{s['run_id']}`；2026-09-29。已完成用户70例评分归档、质量规则、真实来源提取和独立抽帧；目前停止在SAM2批量推理之前。**合格合成对为0，训练为0；约50例仍是下一步合成目标。** 不能把本页来源通过数当成数据工厂已通过数。

```mermaid
flowchart LR
 Y[nuScenes train 真实视频 Y] --> Q[CPU 时序 / 几何 / 来源审核]
 D[真实 donor track 候选 A] --> Q
 Q --> S[SAM2实例分割 · 等待GPU]
 S --> X[连续3D放置 / 合成输入X]
 Y --> G[真实GT保持不变]
 X --> R[规则检查 + subagent逐例抽帧]
 G --> R
 R --> H[通过例 · 人工逐帧全检]
 H --> T[未来原DriveEditor微调 · 架构不变]
```

## 当前可复核数量

| 项目 | 数量与边界 |
|---|---|
| 初选 train scene / 短片 | {s['selected_source_scenes']}，每scene一个来源窗 |
| 实际曝光时序门槛拒绝 | {s['sampling_rejected']}，保留原清单，不复制帧补齐 |
| 通过时序元数据的窗口 | {s['continuous_rgb_sources']} |
| RGB齐备并独立实际看图 | {s['available_rgb_sources']} / {s['independent_reviewed']} |
| RGB未在指定分片找到 | {len(s['source_rgb_missing'])}，记录路径并排除，不写视觉结论 |
| 可读来源中全部30帧几何通过 | {s['geometry_pass_among_available']} |
| Receiver 来源 pass / reject / uncertain | {r.get('pass',0)} / {r.get('reject',0)} / {r.get('uncertain',0)} |
| Donor 来源 pass / reject / uncertain | {d.get('pass',0)} / {d.get('reject',0)} / {d.get('uncertain',0)} |
| 待SAM2主实例队列 | {s['gpu_segmentation_jobs']}段、{s['gpu_frames']}帧；两角色取并集，资格不互相升级 |
| 成功提取真实文件 | {inv['all_files']}：JPEG {inv['jpeg_files']}，传感器二进制 {inv['sensor_binary_files']} |
| 原始文件实际大小 | {inv['actual_bytes']/1024**3:.2f} GiB；JPEG实解码，二进制仅完整提取/存在性，尚未认证地面 |

source ID保留初选编号，中间缺号有拒绝记录。S011/S012/S046存在标注间隔超过600ms，硬拒绝不能被某张图清楚覆盖。RGB缺失主要来自旧log→archive近似索引，不能据此声称整个公共数据集没有这些RGB；本轮不继续无边界扫描其他大压缩包。全部缺失路径见`source_rgb_missing.json`及远端`extract_state.json`。

## 独立质量检查与限制

同一个subagent实际查看每段首/中/末和最差清晰度帧，需要时补看原始crop。高可见性GT中仍发现行人、链条、杆件和植被遮挡，因此GT visibility与bbox不能认证可剪切donor。拒绝/待定不进入相应角色队列。抽帧通过不证明30帧无闪烁；后续需要逐帧mask与轨迹统计，再由用户逐帧全检。

素材来自Boston少数日志，重复车辆/近似外观较多。此次优先两个已有公共分片，目的是跑通小批量数据制作，不代表整个nuScenes train分布，也不是泛化评测。未来训练/验证需同时隔离receiver和donor，优先按log隔离，不能只按生成后的case ID随机切分。

三类合成仍需真实mask确认：纯背景不得误盖真实actor；单actor须保留真实B的可见证据；密集车列须分别确认B/C。当前只有主要实例完成来源审核，次要绿框候选尚未自动放行。SAM2后要检查视角、真实地面支持、完整context中的碰撞/遮挡、alpha与模型洞下的protected可见性，不能为凑50例放宽质量。

## 监督与模型接口

真实Y只作监督，X只新增待删A。原DriveEditor架构/条件字段不变；protected信息先用于质量/监督/评价，不能暗中加入网络条件分支。模型条件只能由X及合法可得输入构造。A的完整影响域若都被H遮住，则masked X与masked Y相同，不能声称donor边缘多样性仍提供可见训练信号。

4项CPU数据合同测试通过：洞覆盖下条件等价、条件接口不读取Y、合成影响域不能逃出删除洞、人工所有帧明确通过才准入。这不是实际训练loader或GPU前向认证。GPU入口已核对官方SAM2参数与双向传播，保留官方默认后处理，断点续跑直接核对来源字段和30张mask可解码。尚未执行。

## 页面与下一步

本地`outputs/v77-target-protected/index.html`为**来源预审页**，三列是真实RGB、GT定位、主要实例crop。按真实曝光时间播放，逐帧评分、备注、导入/导出。评分scope为source-only，不会转成合成样本已批准。资源与JS语法核验见交付记录；未声称浏览器实际播放验证。

下一步用户开GPU后，先运行来源SAM2队列并核查mask，再构造约50例X/Y。实际合成通过规则与独立逐例抽帧后，才交付真实GT / 合成输入 / target+protected / 实际模型输入四列的正式逐帧全检页面。现在不启动DriveEditor推理或训练，不开旧自动化，不执行关机。

远端完整run：`{root}`。数据盘仍有约149GiB，当前试产不需要扩容；扩大训练集前按实际每对占用再估算。

[质量协议](TARGET_PROTECTED_QUALITY.md) · [输入与监督合同](TARGET_PROTECTED_CONDITION.md) · [来源决定和数量](../autoresearch/worldsim_v77/target_protected_20260929/cpu_closeout.json) · [独立逐例记录](../autoresearch/worldsim_v77/target_protected_20260929/subagent_source_reviews.json) · [此前用户评审](DELETE_AUDIT_USER_REVIEW.md)。failure_ledger_refs: [V77-F02]；failure_ledger_delta: updated V77-F02（来源质量/工程边界，不是新增模型实验）。
'''
    (repo/'docs/v77/TARGET_PROTECTED_CPU.md').write_text(report)
    (root/'CPU_PREPARATION.md').write_text(report)
    status=f'''# 当前研究状态

2026-09-29，v77；唯一任务`{s['task_id']}/{s['run_id']}`。CPU素材准备收口，按用户要求**等待用户开启GPU后再运行SAM2**。当前没有模型推理/训练，没有旧定时任务或关机授权在执行。

70例用户CSV已原样归档并推送，human_score 0/1/2/未填=26/26/15/3；两列差异不补写。下一阶段聚焦well-observed Target + Protected Actors，真实nuScenes train视频作Y，只添加遮挡A构成X；原DriveEditor架构不变。旧DEV/audit不用于新训练，隔离final保持不动。

64个初选来源经时序、文件与几何检查，{s['available_rgb_sources']}段RGB可读且独立subagent全部抽帧。Receiver来源{r.get('pass',0)}通过，donor来源{d.get('pass',0)}通过，{s['gpu_segmentation_jobs']}段进入SAM2来源队列；拒绝与待定保留。2段RGB在索引指向的分片缺失并排除；3段标注间隔不合格。来源主要集中Boston少数日志，外观重复，不声称泛化。

**尚无合格合成对，约50例是下一步目标。** 本地`outputs/v77-target-protected/index.html`是来源预审，不是最终合成审核。GPU分割后完成放置/遮挡/边缘/模糊/模型条件检查，独立subagent每例抽帧，仅合格例进入用户逐帧全检；人工verdict仍为空。次要actor尚未自动准入。

远端run：`{root}`。CPU配额0.5核/2GiB，磁盘可用约149GiB，本批无需扩容。4项CPU合同测试、页面资源和JS语法核验已完成，GPU调用尚未验证。见[CPU结果与组件图](v77/TARGET_PROTECTED_CPU.md)、[质量协议](v77/TARGET_PROTECTED_QUALITY.md)、[接口合同](v77/TARGET_PROTECTED_CONDITION.md)。failure_ledger_refs: [V77-F02]；failure_ledger_delta: updated V77-F02。
'''
    (repo/'docs/RESEARCH_STATUS.md').write_text(status)
    p=repo/'docs/EXPERIMENTS.md';lines=p.read_text().splitlines()
    for i,line in enumerate(lines):
        if line.startswith('| WS-V77-TARGET-PROTECTED-20260929 / r1 |'):
            lines[i]=f"| WS-V77-TARGET-PROTECTED-20260929 / r1 | CPU来源{s['available_rgb_sources']}段独立抽帧，receiver {r.get('pass',0)}/donor {d.get('pass',0)}通过；{s['gpu_segmentation_jobs']}段待SAM2，合格合成对0，GPU前停止 | [CPU结果](v77/TARGET_PROTECTED_CPU.md) · [质量协议](v77/TARGET_PROTECTED_QUALITY.md) |"
            break
    else:raise AssertionError('missing experiment registration')
    p.write_text('\n'.join(lines)+'\n')
    p=repo/'docs/research_failures/entries/V77-F02.md';body=p.read_text()
    heading='## 2026-09-29 Target + Protected Actors 来源质量边界'
    if heading not in body:
        p.write_text(body.rstrip()+f'''\n\n{heading}

同一任务CPU试产：{s['available_rgb_sources']}段实际RGB全部由独立subagent抽帧，receiver {r.get('pass',0)}、donor {d.get('pass',0)}来源通过。GT高可见性仍含行人/链条/杆件遮挡；S011/S012/S046标注间隔超过600ms，几何硬拒绝。两段RGB未从旧索引定位分片找到，不写视觉结论。不能从来源清晰或bbox有效推到实例mask、合成质量或补景模型有效。Boston少数日志的重复车形也不能按scene数宣称外观多样性。

下一步仍是用户指定的真实Y/人工遮挡X路线，原DriveEditor架构不变；当前0个合格合成对、0次模型调用。GPU来源mask后才能判断三类合成是否满足protected可见性和真实地面/遮挡。见[CPU结果与边界](../../v77/TARGET_PROTECTED_CPU.md)。保留原始源数据、拒绝、待定和旧模型结果；这是来源质量与工程证据，尚未否定或确认微调路线。
''')
    print('CPU_CLOSEOUT_RECORDED')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True)
    a=p.parse_args();main(a.root,a.repo)

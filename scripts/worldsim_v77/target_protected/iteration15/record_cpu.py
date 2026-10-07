"""登记本轮CPU准备；不从旧关机/训练记录触发任务。"""
from common import *
import subprocess


def main():
    assert read(O/'preflight.json')['CPU_ready']
    assert not (O/'training').exists() and not (O/'diagnostics').exists(), '已有GPU工作，禁止重置CPU阶段'
    plan=read(O/'manifest.json')
    # 首次CPU登记纠正从parent沿用的scope描述，不改变任一数组/模型/seed。
    scope='四既有train / 两合成DEV / 五真实DEV；两个可靠参考的模块探针；不是最终泛化评测'
    if plan['scope']!=scope:
        dump(O/'manifest.before_scope_fix.json',plan)
        plan['scope']=scope;dump(O/'manifest.json',plan)
    report='''# r48：短循环检查、训练与真实DELETE验证

task `WS-V77-TARGET-PROTECTED-20260929/r48`，parent r47，failure refs `V77-F02`。2026-10-07，wm-3090-1001。当前仅CPU准备；训练0步、模型采样0窗，等待用户开GPU。

```mermaid
flowchart LR
 R[现有邻帧RGB · 先剔除A] --> V[冻结官方RGB VAE]
 V --> C[RGB交叉注意力]
 G[LiDAR / pose / tracks] --> B[BEV + 2D几何编码]
 X[目标RGB + 固定H + 同一指令] --> D[DriveEditor · 主干冻结]
 C --> A[r47小条件分支 · 继续训练]
 B --> A
 D --> A
 A --> T[64步 → 验证 → 最多128步]
 T --> N[原生DELETE → 固定α写回]
 N --> Y[真实DEV + 已知GT对照]
```

## 人工记录与本轮问题

新版 `打分记录.xlsx` 5sheet完整只读归档到run的user_review；16条r46/r47真实记录分版本、逐单元格保存，源文件未改。r47已填5例均1分；A013、A007、A022空白保留null。A034/A061保护车局部结构改善但仍有薄膜，A048与r46相近，A041写明条件组涂抹更严重；不能写成其他case都没有回退。A022的r46人工2分保留作回退检查。

真实参考检查沿用实际选中的输入，未重新选图：A034/A061保护角色有效token161/94，但多个slot属于不同B或背景，不等于同一车身面积；slot0/1有主要保护车部分视角。A048保护token40、slot3为0，A041仅12且车身局部。用户A041“RGB足够”的判断原样保存，与当前进入模块的局部参考区分，不擅自改为RGB不足。参考充足度、VAE压缩损失与融合绑定分别待GPU验证。

## 已修正的工程混杂

旧r47在RGB和几何同时移除时，也清空四维任务参数；其他条件组却保留指令。旧“完整比全未知洞MAE低13.3%”包含这项变化，不能作为纯先验增量证据。历史输入、权重、数值和视频均保留。

r48训练dropout和所有C消融始终保留相同参数化指令；UC显式清空新RGB/几何/指令，保留原官方CFG接口与查询mask/相对时间合同。完全关闭分支的r46是部署基线，不冒充同一学习分支的纯先验消融。

沿用389856参数的r47分支与state_dict，没有改网络、loss或参考选择。仅增加采样第1/25次C/UC的四处RGB head、几何head及门控Δ/激活RMS；非有限即停。记录输出敏感性不能代替语义或时序收益。

## CPU输入与接口验证

11例110帧通过：4train与7QUERY场景分离，真实DEV和合成DEV不入训练；查询H/alpha、参考实际PNG、数组和source-mask检查来自不可变r47/旧r21。修改隐藏X不改变条件，Y只作监督/度量；O/N/U划分、维度、有限性、控制H的浮点面积缩放一致；C指令四种组合相同、UC参数显式零且不原地改变源条件。r47 checkpoint严格加载、参数有限；添加记录前后与原r47在受控激活上逐元素相同。10份源码语法通过；GPU入口在加载官方模型前要求可用GPU。

五真实case的保存PNG、实际alpha在f05精确重现写回。A034/A061洞内alpha=1区域分别约77.8%/89.0%，该区域原生和写回完全相同；这是融合定位证据，不单独认定残影的模块根因。只看f05也不能推断整段时序。

CPU准备约62秒（0.5核/1线程）。数据盘600GB约剩92GB，不需要清理/扩盘。CPU审核页复用42条既有视频并明确标旧r46/r47，244个媒体/证据链接存在；未跑的r48留空。媒体是DELETE，不是factual原位重建。

## GPU固定短循环（尚未执行）

1. 用r47 step320检查A034/A061的冻结VAE输入→重构，正确RGB、跨case错配RGB、无RGB、全未知C四组；错配只换RGB latent，query几何、pose、valid、时间、指令固定，仅是模块探针。
2. 从同一r47 checkpoint继续，仅M010/scene-0240、M013/scene-0228、M018/scene-0290、M042/scene-0295；AdamW1e-4、seed6201、320×576、原diffusion loss，RGB/几何各25%独立dropout，主干/3D/VAE冻结。
3. 64步立即验证A034、A061_w08、A022及M003、M006；无工程/数值错误才继续到128步，在上述五例加A048、A041_w10。真实无隐藏GT，M003/M006的Y仅算恢复误差。
4. 所有推理保留旧query H/alpha、10帧、576×1024、seed42、25采样步；保存原生与写回PNG。16步断点保留branch、AdamW、顺序和随机状态，64/128固定权重不按真实结果择优。64验证必须先完成；子进程串行释放显存。

GPU启用后手动入口：

```bash
cd /root/autodl-tmp/motion_proj_v77
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/autodl-tmp/envs/driveeditor/bin/python scripts/worldsim_v77/target_protected/iteration15/run_cycle.py
```

此CPU阶段没有启动控制器、定时任务、GPU模型或等待开卡的后台进程。预计3090完整短循环约45–75分钟，依据r47耗时，仍需实际计时；工程/数值错误即停。128步后收口，不自动增步数/扩数据/扫参数。

HTML将并列r46、r47、本轮64/128，以及原生/写回、参考VAE重构和模块响应。重点是后车结构、残影、幻觉及A022回退；人工0/1/2留给用户。先确定编码信息、融合利用或监督迁移的具体证据，再提出一项模块修改，不能以本轮小数据/预算否定全部架构，也不继续无边界增加分布微调。

## 证据

本地 `outputs/v77-priors-r48/index.html`，远端run的 `review/index.html`。轻量证据：[登记](../autoresearch/worldsim_v77/target_protected_20260929/r48/manifest.json)、[CPU检查](../autoresearch/worldsim_v77/target_protected_20260929/r48/preflight.json)、[实际输入](../autoresearch/worldsim_v77/target_protected_20260929/r48/input_checks.json)、[新版人工记录](../autoresearch/worldsim_v77/target_protected_20260929/r48/human_review.json)。原件、数组、旧输出和r47权重在数据盘保留。

failure_ledger_delta: updated V77-F02（新版人工结果、消融混杂、CPU工程修正）；无新增模型科学结论或新failure ID。
'''
    (REPO/'docs/v77/TARGET_PROTECTED_SHORT_CYCLE_R48.md').write_text(report,encoding='utf-8')
    status='''# 当前研究状态

更新：2026-10-07。分支 `research/worldsim-v7.7-target-protected-editing`，主机 wm-3090-1001。

唯一run `WS-V77-TARGET-PROTECTED-20260929/r48` 已完成CPU准备，停在等待用户开GPU。训练0步、新采样0窗、无后台GPU控制器或定时任务。当前CPU实例在线、无GPU，数据盘600GB约余92GB，无需清理。

新版人工工作簿5sheet已原样归档；r47五例已评均1，A034/A061局部改善仍有薄膜、A041明确局部回退，三例未评保留空白。默认继续r46官方原权重+r21完整SAM，A022人工2正控制保留。

修正旧r47全未知C同时清空任务指令的消融混杂；r48各C组保持同一指令，UC另行清空。旧13.3%差值不再当纯先验增量。11例110帧实际合同、r47权重严格兼容、受控响应记录输出等价通过；CPU审核页已生成，r48效果栏明确留空。这些不是实际官方GPU模型验收。

下一步用户启GPU后，从r47 step320先查可靠参考VAE/正确错配无RGB响应，仅4既有train续64步并立刻验证；工程/数值正常才到最多128步，5真实DEV+2已知GT对照。保持主干冻结、原mask/geometry/seed/采样步，不扩数据/扫参数，128后停止。人工verdict只由用户填写，不自动推广或追加训练。

组件图、固定范围、启动入口及证据见 [r48报告](v77/TARGET_PROTECTED_SHORT_CYCLE_R48.md)。同一失败卡 [V77-F02](research_failures/entries/V77-F02.md)，历史r47结果保留于[原报告](v77/TARGET_PROTECTED_MULTI_PRIOR_R47.md)。
'''
    (REPO/'docs/RESEARCH_STATUS.md').write_text(status,encoding='utf-8')
    index=REPO/'docs/EXPERIMENTS.md';text=index.read_text(encoding='utf-8')
    row='| WS-V77-TARGET-PROTECTED-20260929 / r48 | 同指令消融修正；11例110帧CPU检查；r47续64/128步固定短循环待GPU | [报告](v77/TARGET_PROTECTED_SHORT_CYCLE_R48.md) |\n'
    if '/ r48 |' not in text:text=text.replace('|---|---|---|\n','|---|---|---|\n'+row,1)
    index.write_text(text,encoding='utf-8')
    failure=REPO/'docs/research_failures/entries/V77-F02.md';text=failure.read_text(encoding='utf-8')
    marker='### r48：人工局部收益与消融指令混杂'
    if marker not in text:
        text+='''\n\n### r48：人工局部收益与消融指令混杂

2026-10-07新版人工记录：r47已评5真实例均1分，A034/A061保护结构有局部改善但半透明薄膜仍在；A048与r46相近，A041写明条件组涂抹更严重，不能概括为全无回退。A013/A007/A022未填不判失败；r46 A022=2保留。

旧r47 null_priors在同时去RGB/geometry时还清空任务参数，其他C却保留。因此前述完整比全未知13.3%的MAE差值不能作为纯先验增量，历史数值保留，不否定已有合成恢复改善或局部人工收益。r48已把C指令保持与UC清空分开，训练dropout同样修正。

保存PNG/实际alpha精确检查显示A034/A061 f05 H内alpha=1区域77.8%/89.0%，原生与写回一致；不能把薄膜一概归为alpha融合。r47结构未改，11例110帧CPU合同和受控输出等价通过；实际VAE信息、错配/无RGB响应及64/128步真实迁移尚待GPU，不作架构/数据根因定论，不追加分布/步数搜索。[本轮证据与组件图](../../v77/TARGET_PROTECTED_SHORT_CYCLE_R48.md)。failure_ledger_delta: updated V77-F02。
'''
        failure.write_text(text,encoding='utf-8')
    previous=REPO/'docs/v77/TARGET_PROTECTED_MULTI_PRIOR_R47.md';text=previous.read_text(encoding='utf-8')
    correction='### r48 消融口径更正'
    if correction not in text:
        text+='\n\n### r48 消融口径更正\n\n2026-10-07 CPU核对发现旧全未知C同时清空任务参数，完整条件C保留。因此上述13.3%差值包含任务指令变化，不能认定为纯先验增量。历史结果不覆盖；r48统一C指令、UC单独清空，并归档最新人工评分，详见[r48报告](TARGET_PROTECTED_SHORT_CYCLE_R48.md)。\n'
        previous.write_text(text,encoding='utf-8')
    evidence=REPO/'docs/autoresearch/worldsim_v77/target_protected_20260929/r48';evidence.mkdir(parents=True,exist_ok=True)
    # Git只留轻量配置；完整逐帧tracks/来源仍在数据盘原manifest中，不复制进仓库。
    published={k:v for k,v in plan.items() if k!='cases'}
    fields=('case_id','scene','split','kind','folder','frame_indices','target_token')
    published.update(cases=[{k:c.get(k) for k in fields} for c in plan['cases']],
        full_runtime_manifest=str(O/'manifest.json'),input_index_is_not_full_backup=True)
    dump(evidence/'manifest.json',published)
    for source,name in [(O/'preflight.json','preflight.json'),
        (O/'input_checks.json','input_checks.json'),(O/'user_review/human_review.json','human_review.json'),
        (O/'report_state.json','report_state.json'),(O/'smoke_checks.json','smoke_checks.json')]:
        if source.exists():
            import shutil
            shutil.copy2(source,evidence/name)
    dump(evidence/'progress.json',{'task_id':plan['task_id'],'run_id':'r48','parent_run':'r47',
        'stage':'CPU_ready_waiting_user_GPU','GPU_jobs':0,'training_steps':0,
        'failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'updated V77-F02',
        'human_verdict':None,'inputs_preserved':True})
    subprocess.run([sys.executable,str(REPO/'scripts/build_research_failure_index.py')],cwd=REPO,check=True)
    print('CPU_RECORD_READY',flush=True)


if __name__=='__main__':main()

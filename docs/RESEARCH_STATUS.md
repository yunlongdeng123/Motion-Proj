# 当前研究状态

更新：2026-09-20。分支：`research/worldsim-v7.4-h2-generative-surface`。本文件是唯一当前状态；过往阶段见 [状态历史](archive/2026-09/v74-0920/STATUS_HISTORY.md)。

## 执行范围

本次 V7.4 文档与维护工具整理已完成：恢复旧失败检索，承接用户归档，修复跨版本链接与重复状态写入。没有启动模型、训练或新实验。远端在本次整理时可访问；此前关机记录是历史事件，不代表当前电源状态，也不作为本轮关机指令。

V7.4-H1 与原 H2 ordered 实现已结束，未形成验证通过的主方法。9月15日的官方模型与下游仿真探索是后续取证，不改变旧实现的关闭结论。9月20日固定 SparseDrive 补证队列已终止，没有待自动恢复的研究队列。

## 已完成与证据边界

| 研究线 | 已知结论 | 证据 |
|---|---|---|
| H1 / H2 旧方法 | H1 NO_SURVIVOR；H2 当前实现关闭；不据此证伪所有 witness-native 假说 | [H1 卡](research_failures/ids/V74.md)、[H2 F08–F11](research_failures/ENTRIES.md) |
| 官方模型首回波 | 官方 VGGT、Ω512、DVGT-1、Pi3X 及第二批参照已跑；旧白车 0.323m 射线被四主模型修复，共同低位早交不是可靠车身证据 | [F13](research_failures/entries/V74-H2-F13.md) |
| 六日志感知 | 四模型都有框精度退化，也有改善；48个预选前车条件全部保留中心匹配，47个保留 IoU 匹配 | [F21](research_failures/entries/V74-H2-F21.md) |
| 局部几何因果 | Ω 车辆局部网格校正后定位恢复，普通删除也恢复；Pi3X 修复不稳定恢复检测 | [F20](research_failures/entries/V74-H2-F20.md) |
| 仿真与规划 | 尚无“普遍 phantom → 严重驾驶危害 → 几何修复恢复”的完整证据；新来源真实基线未过准入 | [F14–F22](research_failures/ENTRIES.md) |
| SparseDrive 固定补证 | 导入阶段缺少 `terminaltables`，实际新增前向 **0/40**；依赖任务跳过，属于执行失败 | [终态](autoresearch/worldsim_simimpact/closeout_20260920/terminal_state.json)、[日志](autoresearch/worldsim_simimpact/closeout_20260920/real_policy_tail.txt) |

仿真探索累计：72次几何前向、222次 CenterPoint、818组策略/PDM 配对、18次 HUGSIM 与4次 SplatAD 反馈闭环。这些是不同口径，不能相加为独立样本；更早首回波实验单列于实验索引。人工 verdict 保持原记录，不由代理补填。

## 后续条件

本次整理结束后不自动扩展实验。如用户恢复研究，应先判断问题的论文价值及下游真实基线是否可用，再制定有限实验；不继续扩大已曝光样本来强迫旧 claim 成立。SparseDrive 如需续跑，先修复环境并使用新的 run 路径，保留本轮失败证据；本文件不预授权该续跑。

结果详情只在[实验索引](EXPERIMENTS.md)、[失败入口](RESEARCH_FAILURES.md)和[V7.4 报告目录](archive/2026-09/v74-0920/README.md)维护；不再追加历史状态到本页。

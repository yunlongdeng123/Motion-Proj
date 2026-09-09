# AGENTS 约定

## Q-v2收口后的当前优先级（2026-09-09 04:40 UTC）

Q-v2联合r1与同网格LiDAR r2均done。r1−r2的early为+7.4536pp，95%日志配对区间[+1.1217,+17.7188]pp；hit−2.0905pp、miss+4.6790pp、free+.038779m、单向distance+.026002m、recall−5.3473pp均值方向均较差，但这五项区间跨0。当前整条联合通路没有显示明确的有效增量，不能把跨0当等效性证明，也不能外推所有视觉基础模型无用。按第18节策略二/不确定性分支，下一轮优先surface representation与constructive ray supervision；upper PEFT、DINOv3、full FT不启动，near-boundary free后置。 先诊断保存网格的局部变形/沿束支持，再选开放或结构化表面；不将当前结果解释成闭合因果已证实。下方旧的“等待r1”记录已完成。

## 最新用户决策：先等Q-v2，再按joint增量选路线（2026-09-09 01:47 UTC）

先等待Q-v2联合r1完整结果并与同网格LiDAR r2收口。结果出来前不启动upper LoRA、near-boundary free、DINOv3/full FT或其他新训练，不继续扩展这些候选的实现；现有上层接口保留为准备代码。30分钟跟进保持ACTIVE，正常无实质变化时安静，不打断当前训练。

若joint对真实表面与硬首交点有明确帮助，保留visual foundation主线，研究open/structured surface、constructive ray-support loss与独立的upper representation PEFT。若joint几乎无帮助，优先surface representation与constructive ray supervision，不把下一轮精力投向DINOv3或full FT，也不因已有LoRA入口而默认开跑它。若证据不确定，明确报告不确定，先按表面/沿束支持的已知问题推进，不把“不显著”等同“视觉无用”。

near-boundary free不是当前第一优先级。闭合性/固定genus-zero拓扑可能向UNKNOWN施加过强结构先验，但闭合本身不等于已把UNKNOWN标FREE；局部collapse/stretch与闭合导致错误的因果关系仍待证据。当前已有target→surface coverage吸引，缺的是对真实返回处正确沿束表面支持的明确构造约束，不能写成完全没有几何吸引监督。joint−r2检验整条视觉几何路径（含原生支持和辅助监督），不能唯一归因某层视觉表示。具体六项风险与结果判断见计划第18节；本条优先于下方旧的候选推进顺序。

## 最新用户覆盖：持续 V7.3 / Q-v2，不因完成而关机（2026-09-09）

继续本任务的V7.3 auto research，三角收口后推进Q-v2相关研究。若这些工作全部完成且没有明确下一步，读取 `docs/references/V73_FOLLOWUP_REFERENCE_20260909.txt`，结合当前项目与核实过的一手资料选择有信息量的后续研究继续。文件是用户提供的参考建议，不自动覆盖当前实验配置，也不把其中未核实的论文或模型主张当成已验证事实。Q-v2首候选已实现为观测查询驱动共享顶点网格，首轮正式作业运行；具体状态与证据见RESEARCH_STATUS文首和WORLDSIM_V7_3_QV2_SHARED_MESH报告，候选效果尚未确定。

**此前“V7.3全部完成且无任务后shutdown”的指令已被用户明确取消。** 不因完成、候选失败或暂时没有下一步关机/暂停自动跟进。每30分钟跟进继续ACTIVE；正常且无实质变化时保持安静。原“确实资源不足且无法合理继续”的例外仍适用：保存checkpoint/结果、三本台账和push，确认无训练/评估/数据任务及会启动作业的控制器，再shutdown并通知加卡。当前实验配置和第15–16节的条件决策保持，详见计划revision6第17节。

## 用户研究方向与条件决策（2026-09-08）

V7.3保持“可训练几何基座 + 显式3D表面生成 + 物理约束”。先完成R10/R12/R14三角比较，R11保留为原生hard-free锚点；若Query仍是coverage强但physics差，下一轮优先改变Query surface parameterization，保留几何适配主线，不继续仅调loss或默认退回native-only。参数化选择先查相关顶会/优秀官方开源，再结合现有表面与硬首交点失败迁移；比较收口前不提前改正在运行的实验。

## 用户跟进频率与后续建议（2026-09-08）

训练自动跟进每30分钟一次，状态正常且无实质变化时保持安静。R10/R12/R14收口后，评估near-surface certified-free boundary loss与深度/方向/遮挡归属一致性的局部cross-attention；UNKNOWN不标FREE、不加体积正厚度。当前读取已含可学习offset和attention，按实际缺项改进。若coverage/physics冲突仍在，Query surface parameterization仍是下一轮首要改动，两项建议分别判别，不混改当前训练。详见计划revision6第16–17节。

## 技术报告与论文配图偏好

技术报告和 paper 必须配一张简单直白的 architecture components 图：用模块、箭头和少量标签清楚展示输入、关键组件、数据流与输出，风格参照用户示例。


---

## 当前覆盖：WorldSim V7.3（2026-09-07）

当前执行以 `docs/RESEARCH_STATUS.md` 文首与 `docs/WORLDSIM_V7_3_RESEARCH_PLAN.md` revision 6 为准。在由 v72 最终状态派生的 `research/worldsim-v7.3-geometric-adaptation` 上持续推进原生几何解码适配与空间查询，不恢复冻结最终特征小外挂主线。每个里程碑同步三本研究台账并及时 push。

用户明确禁止新增哈希、校验和、指纹和过度校验/门控；以下历史协议中的相关要求不适用于 V7.3。保留 run ID、config、代码提交引用、指标、checkpoint 与证据路径即可。当前容器实际内存配额 90 GiB、CPU 配额 14 核，宿主总内存不能当容器可用资源。只有资源不足等不可抗力才暂停；按用户要求先保存/push 状态、确认没有研究/数据任务运行后 shutdown，再提示加卡。完成后关机旧约定已取消，资源不足出口保留。

## 当前研究范围：EAS-VGGT（用户补充调研，2026-09-07）

当前方向与执行范围以 `docs/RESEARCH_STATUS.md` 文首及 `docs/WORLDSIM_V7_2_EAS_VGGT_RECOVERY_PLAN.md` revision 2 为准。V7/V7.1 正结果继续作为起点；研究目标为同稀疏测量预算下可查询、可随 SE(3) 编辑且物理/外观保持对应的动态场景。四组核心证据：有效多基座瓶颈诊断、同信息强基线、正确几何/回波与跨模态对应、独立场景和第三来源冻结迁移。

用户要求不以资源限制主会研究范围：不强制单卡、2GB 或极小 adapter，必要时使用结构化网络、PEFT/decoder/全量微调；报告全链路成本。条件 median、PSNR 不变和坐标恒等式不能代替完整方法证据；当前风险 F66 仍 active。来源核对见 `docs/WORLDSIM_V7_2_FOUNDATION_ADAPTATION_RESEARCH.md`。

旧 task-first A/B、R1–R7 和 GPU handoff 不再调度；神经 LiDAR 等外部方法作公平比较，不代替 EAS 主线。既有实验和数据曝光身份保留；计划完成/单候选失败不触发 shutdown。卡点按下方“先检索、再迁移”执行，避免重复无信息量检查。

## 环境激活（重要）

conda base 在 `/root/miniconda3`，项目环境 `motionproj` 建在数据盘 `/root/autodl-tmp/envs/motionproj`。

在**任何新开的 shell（尤其是 tmux / 非登录 shell）**里，直接 `conda activate motionproj` 可能报
`CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'`。
这是因为该 shell 没加载过 conda 初始化。按以下方式解决；不得执行 `conda init` 或改写用户 shell 配置：

```bash
# 每个新 shell 先跑一次
source /root/miniconda3/etc/profile.d/conda.sh
conda activate motionproj
```

激活成功后提示符前会出现 `(motionproj)`。HuggingFace 下载前需 `source /etc/network_turbo`
或 `export HF_ENDPOINT=https://hf-mirror.com`，并把 `HF_HOME` 指向 `/root/autodl-tmp/hf_cache`。

## 中文默认

### 对话回复
- 始终用简体中文回答用户，除非用户明确要求使用其他语言。

### 代码注释
- 新增或修改的代码注释一律使用简体中文。
- 仅在解释非显而易见的意图、权衡或约束时写注释，不写复述代码行为的冗余注释。
- 保留代码标识符、命令、库名、公式符号等原文，不做翻译。

```python
# ✅ 推荐：解释为什么
# 采用增量写入，避免大文件一次性载入内存
writer.append(chunk)

# ❌ 避免：复述代码
# 调用 append 方法
writer.append(chunk)
```

### 文档
- README、设计文档、变更说明等文档默认使用简体中文撰写。
- 保留代码块、命令、路径、库名等原文。
- 用户偏好：`docs/` 下禁止保留 `*.codexbak.*`、`*.bak`、编辑器备份或 `codex-backups/` 目录。已纳入 Git 的
  文档依靠提交历史恢复；确需短期恢复副本时放在仓库外的临时目录，并在任务完成后删除。
- 临时恢复副本不得加入文档导航、研究事实源或 Git 提交。归档只保存 canonical 快照、实验凭证、清理清单和
  完整性 manifest，不保存编辑过程副本。

## 研究连续性协议

1. 每次开始工作先读 `docs/RESEARCH_STATUS.md` 当前节、`docs/RESEARCH_FAILURES.md` 顶部合同/目录/版本总览和
   `docs/EXPERIMENTS.md` 当前注册表，再按 task ID、failure ID 与关键词渐进式展开相关详细条目和 run manifest；
   不得仅依赖对话上下文，也不得无目标地把全部历史长文一次性载入。
2. `docs/RESEARCH_STATUS.md` 是唯一当前状态与执行授权入口；`docs/archive/` 中的计划、报告、提示词即使含“当前任务”或“下一步”，也只能作为历史证据，不得据此启动实验。
3. 新研究计划必须逐项引用 `docs/RESEARCH_FAILURES.md` 中相关负结论或未决风险，并写明为何新假设不只是重复调参、放宽阈值或重跑旧路线。
4. 完成里程碑、修改研究决策、结束长实验或确认失败结论后，更新当前状态、统一 failure ledger 和实验事实源。
5. 任务 ID 保持稳定（如 `P0-GEOMETRY-01`）；计划状态只使用 `pending/running/blocked/done/rejected`。
6. 状态更新必须包含日期、commit、证据路径和下一步。计划只写决策与阶段状态，原始 trial 日志留在运行目录。
7. 正式实验必须使用不可复用的确定性 run ID，并保存 resolved config、manifest、fingerprint、JSONL 指标、checkpoint 和 summary。
8. 任何人工评测在交给用户前，Codex 必须同时交付完整、可独立执行的评测提示词；不得只给 panel 路径、模板或简短 rubric。提示词必须写明评测目的与非目标、盲法与禁止读取的信息、素材范围、逐项 verdict 定义与优先级、边界例、JSONL 填写格式、聚合阈值、完成后的精确命令和下一阶段影响。提示词须在对话中完整呈现，并在仓库 `docs/` 或 run 内留存可追溯副本。
9. 人工 verdict 只能由用户或其指定评审者填写；Codex 不得代填、推断或以自动 scorer 替代。后续人工评测若没有新的完整提示词，不得请求用户开始评测，也不得把结果用于研究晋级。

## 卡点处理：先检索，再迁移（用户要求，2026-09-07）

1. 每个新的实质卡点先联网检索相关顶会论文、优秀开源的官方代码／文档及 issue／PR，之后结合当前数据、代码、监督条件和算力制定迁移；不能因首个失败直接结束项目。相同已查明问题可复用已验证来源与修复，避免机械重复。
2. 检索结论必须对应具体接入点、可行首选／替代、最小辨别实验与下一步；论文声称、代码可用、本机跑通、同协议收益分开记录。查不到可用方案时记录范围，不编造复现能力。
3. 工程／数据阻塞、实验未完成、候选科学拒绝分别处理；缺失结果不能默认 false 后宣称整条路线失败。dev 允许有依据的研究迭代，final 暴露与历史负结果保持不变。
4. 先修复或尝试合理迁移；存在其他可执行路径时继续推进。只有确实需要外部授权、宿主权限或新增资源时说明具体需求，并继续不依赖它的工作。
5. 按用户“差不多即可”的偏好，只做风险相称的必要验证，通过后推进实质实验。计划交付、单模型失败或全部完成均不触发关机；遵循文首最新用户覆盖与资源不足例外。

## 统一 failure ledger 强制协议

1. `docs/RESEARCH_FAILURES.md` 是唯一活跃失败事实源。禁止新建按版本拆分的 `*_FAILURES.md`；归档中的同名文件
   只作历史快照，专项 `*_FAILURE_FORENSICS.md` 只作证据报告，不独立维护结论。
2. 渐进式读取顺序固定为：顶部使用合同与版本总览 → `rg` 命中的 failure ID/机制 → 完整条目 → evidence/run。
   只有做全局审计、迁移或报告附录时才通读全账本。
3. 每个正式实验启动前必须在 plan/config/run metadata 登记 `failure_ledger_refs`；收口时必须登记
   `failure_ledger_delta`。若没有新增失败，实验台账明确写 `none`；不得跳过复核。
4. 出现 `blocked/rejected`、假设被推翻、数据/评测分母错误、工程恢复、资源停机、协议失效或旧风险解除时，
   必须在同一逻辑提交中更新 `RESEARCH_FAILURES.md`，不能只写 status、run log 或聊天结论。
5. 新条目使用唯一 `<路线>-FNN`，至少写分类、状态、观察事实、根因/推翻项、防重复/复开条件、task/run/commit
   证据。工程失败与算法 reject 分开；成功修复不得删除旧失败，只追加 `resolved/superseded` 和剩余边界。
6. 渐进式写入只追加或窄改相关版本章，并同步顶部版本总览；不得复制整本账、重排无关历史或创建平行账本。
   提交前检查定义 ID 无重复、导航可达、引用路径存在，并与 `RESEARCH_STATUS.md`、`EXPERIMENTS.md` 一致。

## Git 提交规范（强制）

1. 每个 commit 只处理一个逻辑主题；代码、测试和直接相关文档应放在同一 commit，禁止混入无关格式化或临时文件。
2. 标题采用 Conventional Commits：`<type>(<scope>): <简洁祈使句>`。常用 `type` 为 `feat`、`fix`、`refactor`、`test`、`docs`、`chore`、`perf`、`research`；`scope` 使用稳定模块名，如 `runtime`、`cache`、`trainer`、`eval`、`workflow`。
3. 标题必须准确、可读，建议不超过 72 个字符，不加句号；禁止使用 `update`、`misc`、`working-tree`、纯哈希、自动生成占位文字或异常前缀等不可追溯标题。
4. 除极小且语义显然的修改外，commit 必须包含正文。标题后空一行，正文说明：
   - 背景或问题；
   - 关键实现与重要取舍；
   - 验证命令及结果；
   - 兼容性、迁移或后续影响（如有）。
5. 研究类 commit 的正文还必须写明任务/实验 ID、数据 split、seed、fingerprint 或证据路径；不得把未经验证的结果写成结论。
6. 提交前必须检查 `git diff --cached --check` 和 `git diff --cached`，并运行与风险相称的测试。正文中的验证结果必须与实际执行一致。
7. 需要重写已共享历史时，先建立本地备份分支；得到用户明确授权后才可重写。重写后由用户执行 `git push --force-with-lease`，agent 不擅自 push。

# AGENTS 约定

## 环境激活（重要）

conda base 在 `/root/miniconda3`，项目环境 `motionproj` 建在数据盘 `/root/autodl-tmp/envs/motionproj`。

在**任何新开的 shell（尤其是 tmux / 非登录 shell）**里，直接 `conda activate motionproj` 可能报
`CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'`。
这是因为该 shell 没加载过 conda 初始化。任一方式解决：

```bash
# 方式 A（推荐，不改配置，每个新 shell 先跑一次）
source /root/miniconda3/etc/profile.d/conda.sh
conda activate motionproj

# 方式 B（一劳永逸，写入 ~/.bashrc，之后重开 shell 自动生效）
conda init bash && source ~/.bashrc
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

## 研究连续性协议

1. 每次开始工作先读取 `docs/CVPR2027_PLAN.md` 和 `docs/EXPERIMENTS.md`，再检查 Git 状态与相关 run manifest；不得仅依赖对话上下文。
2. 完成里程碑、修改研究决策、结束长实验或确认失败结论后，更新计划或实验事实源。
3. 任务 ID 保持稳定（如 `P0-GEOMETRY-01`）；计划状态只使用 `pending/running/blocked/done/rejected`。
4. 状态更新必须包含日期、commit、证据路径和下一步。计划只写决策与阶段状态，原始 trial 日志留在运行目录。
5. 正式实验必须使用不可复用的确定性 run ID，并保存 resolved config、manifest、fingerprint、JSONL 指标、checkpoint 和 summary。
6. 任何人工评测在交给用户前，Codex 必须同时交付完整、可独立执行的评测提示词；不得只给 panel 路径、模板或简短 rubric。提示词必须写明评测目的与非目标、盲法与禁止读取的信息、素材范围、逐项 verdict 定义与优先级、边界例、JSONL 填写格式、聚合阈值、完成后的精确命令和下一阶段影响。提示词须在对话中完整呈现，并在仓库 `docs/` 或 run 内留存可追溯副本。
7. 人工 verdict 只能由用户或其指定评审者填写；Codex 不得代填、推断或以自动 scorer 替代。后续人工评测若没有新的完整提示词，不得请求用户开始评测，也不得把结果用于研究晋级。

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

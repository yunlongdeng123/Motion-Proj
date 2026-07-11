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

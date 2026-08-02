# DR-M5 DGGT inference-only 最终报告

- 裁决日期：2026-07-29
- 研究状态：`blocked`
- 正式恢复实例：
  `/root/autodl-tmp/runs/dynamic_recon/DR-M5-DGGT-NUSC-01/20260729T133346__native-nusc-s0-wm3090/`
- runner SHA-256：`3be81eef40d2062b9a8000ed086a5d9fbbb99e81e7aa25d3345dc90b4c07f445`
- DGGT commit：`a3276d2bbe4cbb03bcc117830b1836110a27adeb`
- 模型 revision：`735ac9a6486057b1eb886c33a8c6dc79e0b43214`

## 终态

M5 没有产出 DGGT 质量、速度或 1-view/3-view 数字。恢复实例在安装 pointops2 时到达明确 upstream packaging
blocked 终态：

```text
ModuleNotFoundError: No module named 'torch'
ERROR: Failed to build .../third_party/pointops2 when getting requirements to build wheel
```

普通 `pip install .` 启动 PEP 517 隔离 build env；pointops2 的 setup requirements 阶段直接 import `torch`，
但隔离环境没有继承主 DGGT env 已安装的 torch 2.4.1。正式 stage 为 `rc=1`，不是 OOM、GPU OOM、网络下载失败
或 DGGT 推理质量失败。

## 阶段证据

| 阶段 | 状态 | 关键事实 |
|---|---|---|
| `env_create` | done | Python 3.10 隔离环境 |
| `env_torch` | done | torch 2.4.1 / torchvision 0.19.1 / torchaudio 2.4.1，cu118 |
| `env_requirements` | done | `rerun-sdk 0.23.1 / opencv-python 4.11.0.86 / numpy 1.26.4` |
| `env_pointops2` | blocked | PEP 517 build isolation 缺少 torch，`rc=1` |
| fixed-window input staging | not run | blocked 发生在 staging 前 |
| 5.41 GB checkpoint | not run | 未下载，不存在本地 checkpoint SHA |
| untouched `difix` smoke | not run | 没有伪造预期 failure |
| patched 1-view / 3-view | not run | coverage 不存在，不写成 0/18 模型表现 |
| common-observation metrics | not run | 216-target 映射仅完成只读预审 |

pointops2 阶段资源：峰值 cgroup memory `16,839,843,840` bytes，GPU 0 MiB，最小可用磁盘
`70,653,599,744` bytes，`oom=0 / oom_kill=0`。

## common-observation 边界

在正式 run 外，M5 finalizer 启动前做过只读映射预审：六场景 × 三窗口 × 四帧 × 三相机共 `216/216`
target 均能映射到 M4 冻结 AD-GS render/GT，解码 RGB 像素逐一相同；其中 train split 180、held-out test split
36。因为 M5 native 已在 input staging 前 blocked，正式 common-observation finalizer 没有启动，也没有计算
Alex-LPIPS 或 side-by-side 数字。该预审不能冒充正式 M5 结果。

## 门禁解释

权威计划第 15.1 节允许“DGGT smoke 可运行或有明确 upstream blocked 证据”。本 run 保留完整 manifest、source
snapshot、stage stdout/stderr、资源采样、summary 和 terminal，因此满足继续 M6 的替代前置证据。没有在同一
instance 事后加入 `--no-build-isolation` 覆盖失败；若未来仍要修 DGGT packaging，必须使用新的任务/run 和独立
兼容性证据，不能改写本轮 blocked 终态。

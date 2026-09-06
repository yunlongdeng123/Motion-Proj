# 动态驾驶场景可编辑重建与失败诊断计划 V2

- **版本**：V2
- **日期**：2026-08-02
- **目标执行者**：Codex Agent
- **执行环境**：远端 AutoDL，单卡 NVIDIA GPU，项目根目录默认 `/root/autodl-tmp/motion_proj`
- **权威前序计划**：`docs/archive/2026-07/dynamic-reconstruction-v1/DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md`
- **V1 终态**：AD-GS 六场景精确复现成功；DGGT 因 pointops2 构建方式阻塞；M6 只完成身份资产审计，未真正执行编辑压力测试；候选 A“补实例身份并做轨迹编辑”因创新性重合被拒绝
- **V2 当前状态**：`M0–M4 done / M5 pending and authorized`
- **V2 核心任务**：补齐前馈对照和可编辑基线，真正产生编辑结果与失败证据，再决定是否存在新的研究贡献

### 预执行现场校准（2026-08-02）

本节记录 V2 执行前已经核实的现场事实；它只减少重复下载和错误假设，不表示 M0/M1 已执行：

- 机器为 RTX 3090 24,576 MiB，driver `580.105.08`，cgroup memory 90 GiB；
- V2 授权前仓库基线为 `main@1e83ad5b`，当时只有本计划未跟踪；
- AD-GS 六个正式 `model_60000` checkpoint、official render、metrics 与 39G processed 输入受保护；
  100/1,000-step profiling 载荷已按清理账本移除；
- DGGT 完整预下载候选已经驻留于
  `/root/autodl-tmp/checkpoints/dggt_preload/model_latest_nuscenes.pt`，大小 `5,411,266,466`
  bytes，SHA-256 `fd15644b3a878849470cbf5f0f9eae39167cfec1b853092898ae754c4f3acde9`；
  M1 必须验证固定 revision/license/provenance 后复用，不再重复下载；
- V1 的 DGGT 失败环境与不完整 `.partial` 下载已经清理；M1 仍须新建 `dggt-v2`，不能把清理理解为
  V1 失败证据消失；
- DriveStudio source/env 存在，repo commit 为
  `e59bda4fa681f829dbb1d65f0de582b0f633c450`，但未发现 V2
  `scene-0230/0242/0255` processed data 或 actor-aware checkpoint；M3 初始路径应视为“源码/环境可用、
  pilot 资产缺失”；
- 尚未创建 `/root/autodl-tmp/runs/dynamic_editing_v2`，本次文档/存储预整理不计作 M0。

---

## 0. Codex 执行总指令

本计划是一个**新的研究任务**，不得修改、覆盖或“恢复” V1 的 `rejected` 终态。

执行顺序固定为：

```text
M0 事实源与环境镜像基线
→ M1 修复 DGGT 推理对照
→ M2 建立 nuScenes 真值 actor 评测适配器
→ M3 建立可运行的对象级编辑基线
→ M4 单场景真实编辑闭环
→ M5 三场景编辑与去遮挡压力测试
→ M6 基于真实失败重新做创新性门禁
→ M7 条件式方法实现与消融
→ M8 人工盲审与最终裁决
```

### 0.1 小步快跑规则

1. 每次开始工作，依次读取：
   - `docs/RESEARCH_STATUS.md`
   - `docs/RESEARCH_FAILURES.md`
   - `docs/EXPERIMENTS.md`
   - 本计划
   - `AGENTS.md`
2. 检查 Git 状态、当前 commit、现有 run 和资源状态。
3. 一次只推进一个里程碑。
4. 每完成、阻塞或拒绝一个里程碑，必须先更新：
   - 本计划的里程碑表和更新日志；
   - `docs/RESEARCH_STATUS.md`；
   - `docs/EXPERIMENTS.md`；
   - 如出现新的可复用失败，再更新 `docs/RESEARCH_FAILURES.md`。
5. 更新完事实源并提交独立 commit 后，才允许进入下一个里程碑。
6. 不得在对话中声称“正在后台继续”。所有长任务必须进入可查看的 `tmux` 会话，并提供 run 路径、日志路径和状态命令。
7. 遇到内存、显存、磁盘或外部实例中断时，保存现场后停止，等待用户调整资源；不得死磕、杀用户服务或静默缩协议。

### 0.2 本计划禁止事项

- 不恢复 cut-in 挖掘、阈值调节或事件池建设。
- 不改写 AD-GS 六场景已冻结的 M4 数值。
- 不把 DGGT 安装修复、数据适配或 actor registry 工程写成论文创新。
- 不再以“AD-GS 没有持久实例标识”为理由跳过编辑实验。
- 不把 nuScenes 真值 `instance_token` 作为 AD-GS 自监督训练输入；它只允许用于评测对象选择、投影、真值轨迹和 oracle 诊断。
- 不把插值框、伪掩码或生成区域写成真实观测。
- 不因为某一编辑结果不好看就事后换 actor、换 scene 或改编辑幅度。
- 不在 M6 通过前写新方法代码。
- 不在没有真实方法结果时生成虚假的人工审核包。

---

## 1. V1 结果的准确继承

### 1.1 已经成功并冻结的部分

AD-GS 官方六场景复现已完成，冻结结果：

```text
scene-0230, scene-0242, scene-0255,
scene-0295, scene-0518, scene-0749

官方 test mean：
PSNR  = 31.174515
SSIM  = 0.927661
LPIPS(VGG) = 0.163489
coverage = 6/6
```

V2 只读取这些 checkpoint、渲染和指标，不重复训练，除非文件完整性审计明确失败。

### 1.2 V1 未完成的部分

V1 的以下内容不能被表述为已经完成：

1. **DGGT 实际推理未运行**：pointops2 在 PEP 517 隔离构建中找不到已安装的 PyTorch，checkpoint 和 inference 均未启动。
2. **编辑压力测试未运行**：V1 M6 只检查了 camera-local SAM ID 和 AD-GS checkpoint 的二值 `obj` 字段，然后将所有编辑行写为 `ABSTAIN`。
3. **去遮挡与噪声实验未运行**：没有真实编辑视频、对象残影指标、遮挡重算结果或 pseudo-hole 结果。
4. **创新性门禁只否定候选 A**：持久实例身份、actor-centric Gaussian 和基础轨迹编辑与现有工作重合；不能扩张为“整个反事实编辑方向被否定”。

### 1.3 V2 的研究边界

V2 不预设最终贡献。它首先回答：

> 在同一批 nuScenes 驾驶场景中，现有强重建基线和对象级编辑基线在真实轨迹修改、遮挡重算、去遮挡和非目标区域保持方面，究竟会稳定失败在哪里？

只有失败在至少 3 个 scene 重复，且能建立真实或分层真值，才允许注册方法假设。

---

## 2. AutoDL 环境与国内镜像规范

本节是强制执行合同。Codex 必须先生成并运行 `scripts/bootstrap_autodl_v2.sh`，不得在每个里程碑中临时拼装不同的源配置。

### 2.1 目录和缓存

所有大文件、环境和缓存放在数据盘：

```bash
export PROJECT_ROOT=/root/autodl-tmp/motion_proj
export ENV_ROOT=/root/autodl-tmp/envs
export CACHE_ROOT=/root/autodl-tmp/cache
export CONDA_PKGS_DIRS=/root/autodl-tmp/cache/conda-pkgs
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export HF_ENDPOINT=https://hf-mirror.com
export TORCH_HOME=/root/autodl-tmp/cache/torch
export XDG_CACHE_HOME=/root/autodl-tmp/cache/xdg
export PIP_CACHE_DIR=/root/autodl-tmp/cache/pip
export TMPDIR=/root/autodl-tmp/tmp
mkdir -p "$ENV_ROOT" "$CACHE_ROOT" "$CONDA_PKGS_DIRS" "$HF_HOME" "$HF_HUB_CACHE" \
  "$TORCH_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR"
```

禁止将 5 GB 级 checkpoint、Conda 环境或编译缓存写入系统盘 `/root`。

### 2.2 Conda 激活

每个新 shell、tmux pane 和非登录脚本开头必须执行：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
```

使用前缀环境，不依赖环境名解析：

```bash
conda activate /root/autodl-tmp/envs/<env-name>
```

禁止为了方便直接执行 `conda init` 并无审计地改写用户 shell 配置。

### 2.3 网络加速优先级

优先级固定为：

1. Conda/PyPI 通用依赖默认使用项目级清华 TUNA 镜像；
2. Hugging Face 默认使用 `https://hf-mirror.com`，但仍固定 revision、license、字节数和 SHA-256；
3. GitHub 先启用 AutoDL `/etc/network_turbo` 访问官方仓库；传输不稳定时，允许用户已授权的学术加速
   fallback，但必须在 clone 后核对 official remote、固定 commit、submodule 和 license；
4. PyTorch/CUDA 扩展遵循官方兼容版本和官方 wheel variant，镜像只加速传输，不改变构建类型；
5. 禁止来源不明的 GitHub 代理、浮动分支或随机网盘权重。

每个需要联网的 stage 先尝试：

```bash
if [ -f /etc/network_turbo ]; then
  source /etc/network_turbo
fi
```

网络代理不保证稳定。下载脚本必须支持断点续传、重试上限、最终 SHA-256 校验，不能无限重试。

### 2.4 项目级 Conda 镜像，不污染全局配置

创建：

```text
configs/env/autodl_condarc_v2.yaml
```

建议内容：

```yaml
channels:
  - conda-forge
  - defaults
show_channel_urls: true
channel_priority: flexible
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

运行前：

```bash
export CONDARC="$PROJECT_ROOT/configs/env/autodl_condarc_v2.yaml"
conda config --show-sources
```

要求：

- 不覆盖 `~/.condarc`；
- 如果 TUNA 当前不可达，只在该 stage 临时切回官方 channels；
- source 变更必须写入 run 的 `environment/source_resolution.json`；
- 不通过 Conda 安装 PyTorch，避免拿到 CPU 构建或不同 CUDA 变体。

### 2.5 pip 镜像

通用 Python 包使用 TUNA：

```bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_DEFAULT_TIMEOUT=120
export PIP_RETRIES=3
```

不要全局写入 `~/.config/pip/pip.conf`。每个 run 记录：

```bash
python -m pip config debug
python -m pip --version
```

对于 PyTorch、CUDA 扩展和项目官方明确指定的私有 index，必须优先遵循官方版本和官方 wheel index，不能为了下载快而改变构建类型。

### 2.6 PyTorch 安装策略

DGGT 冻结版本：

```text
Python 3.10
PyTorch 2.4.1
TorchVision 0.19.1
TorchAudio 2.4.1
```

首选：在 `network_turbo` 已启用时使用官方 CUDA wheel index：

```bash
python -m pip install \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

安装后必须通过：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device="cuda")
print(float((x @ x).mean()))
PY
```

失败处理：

- 保存完整安装日志；
- 最多重试一次官方 index；
- 再尝试 AutoDL 官方文档建议的国内 pip 源路径；
- 无论使用哪条路径，都必须验证 `torch.cuda.is_available()`、CUDA 版本、GPU forward/backward，并保存 wheel/包版本和 SHA-256；
- 不允许安装成功但 CUDA 不可用时继续编译 pointops2。

### 2.7 Hugging Face 与大权重下载

默认使用镜像：

```bash
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export HF_ENDPOINT=https://hf-mirror.com
```

镜像是传输端点，不是事实源。必须使用官方 repo ID、固定 revision 和官方 license；若镜像缺文件或元数据
不一致，在保留失败日志后临时切换官方 endpoint，不得自动追随新 revision。

必须记录：

- repo ID；
- revision；
- endpoint；
- 文件字节数；
- SHA-256；
- 下载命令和返回码；
- 模型许可证。

禁止用“最新”或未固定 revision 的权重。

### 2.8 Git 和源码

```bash
source /etc/network_turbo 2>/dev/null || true
git clone <official-url>
git checkout <frozen-commit>
git submodule update --init --recursive
```

要求：

- 首选官方仓库；使用用户授权的学术加速时，只改变传输 URL，并记录加速 endpoint 与官方 URL 的映射；
- 固定 commit；
- clone 后把 `origin` 校准或另记为官方 URL，并验证目标 commit 可由官方仓库解释；
- 保存 `git status --short`、`git diff` 和 submodule commit；
- compatibility patch 单独存放在 `compatibility/`，不得直接手改后不留 patch；
- 上游失败和 patch 后结果必须使用不同 stage 或不同 run。

---

## 3. 资源与运行合同

### 3.1 启动前审计

每个 GPU run 记录：

```bash
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used \
  --format=csv,noheader
cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
df -h /root/autodl-tmp
```

最低条件：

- GPU 显存不低于 24 GiB，或对应 stage 有明确较低要求；
- stage 启动前 GPU 已占用不超过 2 GiB；
- cgroup 内存不低于 32 GiB；
- 数据盘可用空间不低于 60 GiB；
- 始终保留 20 GiB 安全余量。

### 3.2 停机条件

任一条件触发即停止当前 stage：

- `memory.current / memory.max >= 0.90` 连续两个采样；
- `memory.events` 中 `oom` 或 `oom_kill` 增加；
- RC 137、SIGKILL 或 CUDA OOM；
- 磁盘可用空间低于 20 GiB；
- 必须降分辨率、删相机、缩窗口或换数据才能继续；
- 外部实例重建、系统重启或文件系统异常。

停止后写 `terminal.json = blocked`，保存日志和资源曲线，等待用户授权。禁止杀 Cursor、Jupyter、TensorBoard 或其他用户进程。

### 3.3 Run 目录

所有 V2 run 使用：

```text
/root/autodl-tmp/runs/dynamic_editing_v2/<TASK_ID>/<instance_id>/
├── manifest.json
├── resolved.yaml
├── source_snapshot/
├── environment/
├── stages/
├── logs/
├── resource.jsonl
├── metrics.jsonl
├── artifacts.json
├── summary.md
└── terminal.json
```

`terminal.json.status` 只允许：

```text
pending | running | blocked | done | rejected
```

失败实例禁止覆盖；重跑必须使用新 `instance_id`。

---

## 4. 里程碑注册表

| 里程碑 | Task ID | 当前状态 | 核心产物 | 解锁条件 |
|---|---|---:|---|---|
| M0 V2 事实源与镜像基线 | `DR-V2-M0-BOOTSTRAP-01` | done | V2 状态、环境 bootstrap、镜像 smoke | 所有源可审计，旧结果不被覆盖 |
| M1 DGGT 修复与正式推理 | `DR-V2-M1-DGGT-REPAIR-01` | done | 18 窗口 1-view/3-view、common 与区域诊断 | 1-view/3-view 18/18，216/216 common target |
| M2 nuScenes actor 评测适配器 | `DR-V2-M2-ACTOR-EVAL-01` | done | raw 2 Hz 轨迹、精确投影、冻结 3×2 actor cohort | 三场景 eligible=16/20/6，各选 2 |
| M3 对象级可编辑基线 | `DR-V2-M3-EDIT-BASELINE-01` | done | DriveStudio/StreetGS actor registry、原始渲染、编辑 API smoke | scene-0230 一对象可移除、平移并三相机渲染 |
| M4 单场景真实编辑闭环 | `DR-V2-M4-EDIT-PILOT-01` | done | 0230 固定 actor 的原始/横移/删除视频与指标 | 编辑真实执行、无空结果、证据可审计 |
| M5 三场景压力测试 | `DR-V2-M5-STRESS-3SCENE-01` | pending | 3 scene × 2 actor × 4 edit 的 failure matrix | 至少 3 scene 完整 coverage，不能全 ABSTAIN |
| M6 创新假设门禁 | `DR-V2-M6-HYPOTHESIS-01` | pending | 唯一 hypothesis、novelty delta、primary endpoint | 跨 3 scene 稳定失败且 novelty 不重合 |
| M7 方法与消融 | `DR-V2-M7-METHOD-01` | pending | matched ablation、3 seeds、统计结果 | M6 通过且 effect size 预注册 |
| M8 人工盲审与终局 | `DR-V2-M8-HUMAN-01` | pending | 完整盲审包、提示词、validator、终局报告 | M7 产生可审方法结果 |

---

# 5. M0：事实源、代码边界与 AutoDL bootstrap

## 5.1 目标

把 V1 的“路线终止”保留为历史结论，同时明确 V2 是用户新授权的独立诊断路线。

## 5.2 必做任务

1. 新建 Git 分支或独立 worktree：

```text
research/dynamic-editing-v2
```

2. 不修改 V1 run 和文档中的历史数值。
3. 更新根目录 `README.md`，修正仍指向 OccGS V7 的过时描述。
   同时修正 `AGENTS.md` 中“可执行全局 `conda init`”与 V2 禁止全局配置的冲突；V2 只允许每个 shell
   显式 `source conda.sh`。
4. 更新 `docs/RESEARCH_STATUS.md`：
   - 当前计划切换到 V2；
   - V1 保留为历史终态；
   - M0 为当前唯一授权里程碑。
5. 将本计划放入：

```text
docs/DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md
```

6. 创建：

```text
scripts/bootstrap_autodl_v2.sh
configs/env/autodl_condarc_v2.yaml
```

7. bootstrap 脚本只做：
   - 缓存目录；
   - 镜像环境变量；
   - 网络连通 smoke；
   - Python/Conda/pip/Git/HF 基础信息记录；
   - 不安装大型依赖。
8. 对现有 V8 代码做 source audit，至少记录：
   - `scripts/run_dr_m5_dggt.py` 中 pointops2 当前安装命令；
   - `scripts/run_dr_m6_stress.py` 是否真正执行编辑；
   - `motion_proj/dynamic_recon/pseudo_tracks.py` 的固定零覆盖逻辑；
   - V7 `actor_registry`、`trajectory_editor`、`counterfactual_render` 与 DriveStudio adapter 的可复用边界。
9. 复核本计划“预执行现场校准”和
   `docs/archive/2026-08/v2-preflight/CLEANUP_MANIFEST.md`，确认被清理的历史环境/中间 checkpoint
   不被误判为 V2 资产缺失。

## 5.3 测试

```bash
bash -n scripts/bootstrap_autodl_v2.sh
shellcheck scripts/bootstrap_autodl_v2.sh  # 若 shellcheck 已安装；不得仅为此污染环境
python -m pytest -q tests/test_dr_pseudo_tracks.py tests/test_v71_actor_registry.py
```

## 5.4 通过条件

- V1 历史状态没有被重写；
- V2 是新的 task namespace；
- bootstrap 在空 shell 和 tmux shell 均可运行；
- 镜像配置不改全局 `~/.condarc` 或 pip 配置；
- `README/STATUS/PLAN` 三者状态一致。

## 5.5 M0 commit

```text
research(workflow): 启动动态编辑 V2 诊断路线
```

---

# 6. M1：修复 DGGT，并完成真正的推理对照

## 6.1 目标

修复 V1 的工程阻塞，不改变 DGGT 模型。完成官方 nuScenes checkpoint 的 inference-only 对照。

## 6.2 冻结资产

优先复用 V1 已锁定值；执行前与官方仓库核对：

```text
DGGT repo commit:
a3276d2bbe4cbb03bcc117830b1836110a27adeb

model revision:
735ac9a6486057b1eb886c33a8c6dc79e0b43214

checkpoint:
model_latest_nuscenes.pt
expected bytes:
5,411,266,466
local preload:
/root/autodl-tmp/checkpoints/dggt_preload/model_latest_nuscenes.pt
expected SHA-256:
fd15644b3a878849470cbf5f0f9eae39167cfec1b853092898ae754c4f3acde9
```

如果官方仓库或权重已经变化：

- 不自动追随新 main；
- 优先使用 V1 冻结 commit/revision；
- 旧 revision 不可用时，记录 blocked 并请求用户决定是否升级协议。

## 6.3 新环境

不得在 V1 失败环境内“就地补一下”。新建：

```text
/root/autodl-tmp/envs/dggt-v2
```

```bash
source /root/miniconda3/etc/profile.d/conda.sh
export CONDARC=/root/autodl-tmp/motion_proj/configs/env/autodl_condarc_v2.yaml
conda create -y -p /root/autodl-tmp/envs/dggt-v2 python=3.10 pip
conda activate /root/autodl-tmp/envs/dggt-v2
source /root/autodl-tmp/motion_proj/scripts/bootstrap_autodl_v2.sh
```

安装顺序：

1. 精确 PyTorch 2.4.1 CUDA 构建；
2. `requirements.txt`；
3. pointops2；
4. import/forward smoke；
5. 核验并复用本地 full preload checkpoint；
6. inference。

checkpoint 复用规则：

- 先核对固定 model revision、model card/license、字节数与上述 SHA-256；
- 通过后在 `/root/autodl-tmp/checkpoints/dggt-v2/` 建立同文件系统 hardlink，或直接以只读 preload
  路径运行；
- 不复制第二份 5.41 GB 文件，不重新下载已验证的相同内容；
- 若官方 revision 元数据无法证明该本地文件来源，M1=`blocked`，不得仅凭文件名启动 inference。

## 6.4 pointops2 修复

V1 的失败属于 PEP 517 build isolation 没有继承 PyTorch。M1 必须先保存旧失败，不覆盖；新 run 按 DGGT upstream 方式执行：

```bash
cd /root/autodl-tmp/third_party/dggt/third_party/pointops2
export MAX_JOBS=8
python setup.py install
```

要求：

- 第一优先使用 upstream `python setup.py install`；
- 不先尝试原来的 `pip install .`；
- 编译日志完整保存；
- 记录 GPU compute capability、GCC、NVCC、PyTorch CUDA ABI；
- 安装后运行 pointops2 最小 CUDA forward/backward；
- 若失败，最多允许一个有明确根因的 compatibility patch；
- patch 只解决安装/ABI，不改算子语义。

## 6.5 输入协议

保留 V1 固定窗口：

```text
每个 scene：
[10,11,12,13]
[34,35,36,37]
[66,67,68,69]

6 scenes × 3 windows = 18 windows
```

必须保存 raw frame、processed frame、staging frame 的映射。

DGGT mode 2 若仍忽略 CLI `start_idx`，继续使用每窗口独立 staging，从 `0` 开始，禁止声称 CLI 生效。

## 6.6 上游错误和兼容补丁

执行顺序：

1. 未修改 upstream 的 `--help`；
2. 未修改 upstream 的最小 inference，保存 `difix/diffusion` 原始失败；
3. 应用 V1 的最小单行 patch；
4. patch 后重新运行；
5. 原始失败和修复结果分开归档。

## 6.7 正式运行

### 1-view primary

- 18/18 固定窗口；
- nuScenes checkpoint；
- diffusion 关闭；
- 输入 resize 和实际 shape 完整记录；
- 输出 RGB、depth、camera pose、Gaussian/中间结构（若 upstream 提供）；
- 记录 wall time、GPU time、峰值 VRAM。

### 3-view diagnostic

- 先运行 1 个窗口 smoke；
- 24 GiB 能运行时扩至 18 窗口；
- 若 OOM，按资源合同停止，不缩 sequence length、分辨率或模型替代正式 3-view。

### common-observation diagnostic

在相同 target frame 上读取已冻结 AD-GS 渲染，报告：

- 两者各自输入帧数；
- 是否使用真值 pose；
- 是否逐场景优化；
- PSNR、SSIM；
- LPIPS 必须分开命名 `LPIPS(Alex)` 与 `LPIPS(VGG)`；
- 动态区域和边界区域诊断；
- 速度和显存。

不得写成同预算排行榜。

## 6.8 M1 通过条件

满足其一：

### `done`

- 1-view coverage = 18/18；
- checkpoint 与输入 hash 完整；
- 所有输出非空；
- 指标和资源记录完整；
- 3-view 为 done 或有独立资源 blocked 证据。

### `blocked`

- 按 upstream 安装方式仍然存在不可修复的源码/权重问题；
- 有完整原始失败、最小补丁尝试和回滚证据；
- 不以环境失败推断 DGGT 方法质量。

## 6.9 M1 commit

```text
fix(dggt): 按上游构建 pointops2 并完成推理协议
```

## 6.10 执行结果（2026-08-02）

M1=`done`，不改变 DGGT 模型语义：

- 官方 repo commit `a3276d2bbe4cbb03bcc117830b1836110a27adeb`、model revision
  `735ac9a6486057b1eb886c33a8c6dc79e0b43214`、`CC-BY-NC-4.0` 和 5.41 GB checkpoint
  SHA-256 均通过；checkpoint 以同文件系统 hardlink 复用；
- 新环境 `/root/autodl-tmp/envs/dggt-v2` 固定 Python 3.10、PyTorch 2.4.1+cu121；因宿主
  toolkit 只有 CUDA 11.8，从官方 NVIDIA Conda 包固定 12.1 compiler/runtime/headers；
- pointops2 严格按 upstream `python setup.py install` 构建，CUDA forward/backward=`PASS`；
- untouched `--help` 通过；untouched inference 精确复现 `args.difix` 错误；实例内最小
  patch 仅修改 `args.difix -> args.diffusion`；
- 原生 r6 完成 18/18 1-view 和 18/18 3-view；1-view mean 为
  `PSNR 20.707359 / SSIM 0.856031 / LPIPS(Alex) 0.135780 / 1.785527 s`，3-view mean 为
  `PSNR 21.165262 / SSIM 0.771051 / LPIPS(Alex) 0.165553 / 4.517659 s`；
- common r8 完成 216/216 target 和 216/216 GT 像素身份校验；冻结 AD-GS 同 target
  1-view mean 为 `34.581860 / 0.951918 / 0.062490`，3-view mean 为
  `34.894344 / 0.951711 / 0.061447`（PSNR/SSIM/LPIPS(Alex)）；
- regional r9 完成 AD-GS 216、DGGT 1-view 72、DGGT 3-view 216 行。在各自冻结分辨率下，
  动态区 PSNR 为 `29.640118 / 22.999911 / 22.902139`，7x7 形态学边界带 PSNR 为
  `29.480968 / 22.017347 / 21.810579`；
- DGGT 与 AD-GS 的输入观测数、pose、逐场景优化和分辨率不匹配，所有对照只是
  failure characterization，不是 matched leaderboard。

正式证据链：

```text
native:
/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M1-DGGT-REPAIR-01/20260802T125138Z__native-nusc-s0-r6
common:
/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M1-DGGT-REPAIR-01/20260802T133151Z__common-retry-s0-r8
dynamic/boundary:
/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M1-DGGT-REPAIR-01/20260802T133912Z__regional-s0-r9
```

---

# 7. M2：nuScenes 真值 actor 评测适配器

## 7.1 目标

解决 V1 M6 的“没有持久 pseudo ID”问题，但不修改 AD-GS 训练协议。

nuScenes 的 `sample_annotation.instance_token` 用于连接同一 scene 中同一个对象的多帧标注。V2 将其作为**评测真值和对象选择依据**。

## 7.2 允许与禁止

### 允许

- 选择评测车辆；
- 获取 2 Hz 原始三维框、类别、可见性和真值轨迹；
- 投影到 AD-GS 三相机；
- 构造 target mask、原 footprint、新 footprint；
- 评估轨迹跟随、遮挡和对象区域指标；
- 构造 oracle diagnostic。

### 禁止

- 输入 AD-GS 官方训练；
- 将插值框写成 raw truth；
- 将真值 actor adapter 伪装成自监督方法贡献；
- 用评测真值修补 baseline 渲染后仍称为原始 baseline。

## 7.3 新模块

建议新增：

```text
motion_proj/dynamic_editing_v2/
├── __init__.py
├── nuscenes_actor_eval.py
├── frame_mapping.py
├── actor_projection.py
├── actor_selection.py
└── schema.py

scripts/build_dr_v2_actor_eval.py
configs/dynamic_editing_v2/actor_selection_v1.yaml
```

## 7.4 数据模型

每个 actor 至少输出：

```json
{
  "scene_id": "scene-0230",
  "instance_token": "...",
  "category_name": "vehicle.car",
  "raw_annotations": [
    {
      "sample_token": "...",
      "timestamp_us": 0,
      "translation_global": [0, 0, 0],
      "size_wlh": [0, 0, 0],
      "rotation_quaternion": [1, 0, 0, 0],
      "visibility_token": "4",
      "num_lidar_pts": 0,
      "num_radar_pts": 0,
      "provenance": "nuscenes_raw_2hz"
    }
  ],
  "camera_observations": [],
  "interpolated_visualization": [],
  "support_summary": {}
}
```

硬要求：

- raw 和 interpolated 字段物理隔离；
- 所有坐标变换写明 `T_dst_src`；
- box size 保持 nuScenes 的 `wlh`，内部使用其他顺序时显式转换；
- camera projection 记录裁剪前后 polygon、可见面积和图像边界；
- 任何无效投影写明原因，不静默丢失。

## 7.5 frame mapping

建立：

```text
nuScenes raw sample/sample_data
↔ AD-GS scene frame index 10..69
↔ DriveStudio/StreetGS processed frame
↔ render output frame
```

映射必须由 timestamp 和 token 双重校验，禁止只按文件名猜测。

## 7.6 actor 选择

M2 只能根据输入支持度选择 actor，不能看编辑输出。

预注册最低门槛：

- 类别前缀为 `vehicle.`；
- clip 内至少 4 个 raw 2 Hz annotation；
- 至少 2 个 raw annotation 的 `num_lidar_pts + num_radar_pts > 0`；
- 在三路前向相机中，至少 4 个 camera timestamp 的投影中心位于图像内；
- 有效投影的中位可见框面积不低于 500 px；
- 无 box 尺寸异常、NaN、token 断裂或 scene 越界。

每 scene 固定：

- `high-support`：支持分数最高；
- `boundary-support`：满足门槛但支持分数最低；
- 分数并列按 `instance_token` 字典序；
- 若只有一个合格 actor，保留 1/2 coverage；
- 不得用“看起来更适合编辑”的下一辆车替换。

支持分数写入 config 并冻结，例如：

```text
raw_annotation_count
+ valid_camera_observation_count
+ log(1 + lidar_radar_point_sum)
+ median_projected_area 的单调项
```

具体权重必须在运行前写入配置，不看编辑结果调节。

## 7.7 M2 QA

每个 scene 生成：

- actor cohort 表；
- raw 2 Hz 三维轨迹俯视图；
- 三相机同步投影 panel；
- raw/interpolated 图例；
- 代表性 box 与 instance token；
- 失败 actor 原因统计。

人工只做身份和投影 QA，不评价方法结果。

## 7.8 单元测试

至少覆盖：

- `instance_token` 链遍历；
- timestamp 映射；
- `wlh ↔ lwh` 转换；
- global→ego→camera→pixel round trip；
- 图像边界裁剪；
- raw/interpolated provenance；
- actor selection 确定性；
- 缺 annotation、错 scene、NaN、无可见相机时 fail closed。

## 7.9 M2 通过条件

- `0230/0242/0255` 至少各有 1 个合格车辆；
- actor cohort 在看编辑输出前冻结；
- 三相机投影人工 QA 无系统错位；
- raw 2 Hz 和插值证据没有混写；
- 输出 hash 和输入 metadata hash 完整。

如果三场景中有两个以上没有合格 actor，M2=`blocked`，先检查 frame mapping/投影，不得降低门槛直到“有样本”。

## 7.10 M2 commit

```text
feat(eval): 建立 nuScenes 持久 actor 评测适配器
```

## 7.11 执行结果（2026-08-02）

M2=`done`：

- 新增 `motion_proj.dynamic_editing_v2` 的 schema、frame mapping、box projection、actor selection
  和 nuScenes adapter；
- 三场景原始 actor 数为 `58/53/56`，按预注册门槛合格数为 `16/20/6`；
- 在任何 M3/M4 编辑输出存在前，每 scene 确定性冻结 high-support 与
  boundary-support 各 1 个，slot coverage=`6/6`；
- 全部 4,356 个 actor-camera observation 均以 timestamp + exact `sample_token` 双重映射；
  无效投影保留原因、零面积和裁剪前后 polygon，不静默丢失；
- raw 2 Hz 注释与 visualization interpolation 物理分离；本运行未生成任何插值 truth；
- 11 个 metadata/frame-table 输入完整哈希，167 行 actor support metrics，每场景 cohort CSV、
  6 组三相机 panel 和 6 张 raw BEV 轨迹图均已封存；
- 自动代表 panel exact-token QA=`18/18`；Codex 逐张视觉 QA 未见身份错配或系统性偏移；
- DriveStudio processed frame 因 M3 资产仍缺失而显式记为
  `assets_missing_until_m3`，未用文件名猜测伪造映射。

正式 run：

```text
/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M2-ACTOR-EVAL-01/20260802T140312Z__actor-eval-s0-r5
```

---

# 8. M3：建立对象级可编辑基线

## 8.1 目标

AD-GS 继续作为高质量重建基线，不强行从其二值 `obj` 中恢复实例。V2 使用已有 DriveStudio/StreetGS actor graph 作为**可编辑基线**。

优先复用 V8 已存在的：

```text
motion_proj.resim.actor_registry
motion_proj.resim.drivestudio_adapter
resim/v71_build_registry.py
resim/v71_build_world_state.py
resim/c0_counterfactual_render.py
resim/s0_trajectory_editor.py
```

但这些代码来自 V7，不能直接假设能在 AD-GS 六场景上工作，必须先做适配审计。

## 8.2 baseline 身份

V2 baseline 必须明确命名，例如：

```text
DriveStudio/StreetGS actor-aware native baseline
```

禁止称为 proposed method。

## 8.3 M3-A：资产和源码审计

检查远端 live repo，不以 ZIP 快照替代：

- DriveStudio/StreetGS 官方仓库路径、commit、license；
- 环境是否存在；
- nuScenes 三个 pilot scene 的 processed data；
- checkpoint 和 config；
- `RigidNodes` 中 actor index 与 `instance_token` 的映射；
- 原始渲染命令；
- remove/transform/render 能力；
- 三相机同步输出；
- 是否有硬编码旧 `mini` 路径或旧 scene ID。

输出 `baseline_readiness.json`，每项为 `available/missing/incompatible`。

## 8.4 M3-B：选择路径

### 路径 1：已有可用 checkpoint

如果 scene-0230 已有 actor-aware checkpoint：

1. 校验 checkpoint/config/hash；
2. 原始轨迹渲染；
3. 建立 `instance_token → dataset column → rigid model index` 一一映射；
4. 继续编辑 smoke。

### 路径 2：没有 checkpoint，但官方训练路径可用

只允许先训练 `scene-0230`：

- 使用 baseline 原生 nuScenes config；
- 不调超参；
- 不看输出挑 checkpoint；
- 先 100/1,000 step profile；
- 资源通过后运行 config 指定的正式训练；
- 记录 native 指标，但不与 AD-GS 直接称 matched。

### 路径 3：baseline 无法在冻结预算内建立

若一次 untouched-upstream 训练 smoke 加至多一次有明确根因的最小 compatibility patch 后，仍无法在
资源合同内获得 scene-0230 的对象级 checkpoint：

- M3=`blocked`；
- 交付缺口清单和最小复现日志；
- 不允许自行开发新的实例高斯方法绕过 baseline gate；
- 请求用户决定是否改用其他官方可编辑基线。

## 8.5 actor registry v2

输出映射必须以 nuScenes `instance_token` 为主键：

```text
instance_token
→ raw annotation chain
→ processed true instance ID
→ dataset instance column
→ RigidNodes model index
→ checkpoint tensor slice
```

一一映射失败时 fail closed。

V7 原先使用整数 `true_instance_id`，V2 必须保留 token 与整数 ID 的双向 provenance。

## 8.6 编辑 API smoke

scene-0230 的 `high-support` actor 必须支持：

1. 原轨迹渲染；
2. actor remove；
3. actor-local 横向平移 `+1.0 m`；
4. 三相机同步渲染；
5. 原 checkpoint 不被修改；
6. 编辑可逆，重新加载原状态得到相同 hash。

不得在 M3 就做质量结论。

## 8.7 M3 通过条件

- 原始重建能渲染；
- actor registry 一一映射；
- remove 和 lateral edit 均生成非空输出；
- 三相机帧数与时间一致；
- 非目标 actor 的参数 hash 不变；
- 原始状态可恢复；
- 具备进入 M4 的稳定命令。

## 8.8 M3 commit

```text
feat(edit): 接入对象级驾驶场景编辑基线
```

## 8.9 执行结果（2026-08-03）

M3=`done`，基线身份为 `DriveStudio/StreetGS actor-aware native baseline`：

- 官方 DriveStudio commit `e59bda4fa681f829dbb1d65f0de582b0f633c450`、MIT license 和原生
  `configs/streetgs.yaml + nuscenes/3cams` 固定；scene-0230 原生索引为 `179`；
- raw/processed 资产为 `1176 images / 196 lidar / 196 poses / 1176 extrinsics / 588 sky masks`；
  100/1,000-step profile 通过后完成原生 30,000-step 训练；
- RTX 3090 的旧 `gsplat/nvdiffrast` 二进制不含 SM 8.6，均从固定官方源码重建；没有修改算子、
  模型或训练超参；
- 正式 checkpoint 为 `386,398,646` bytes，`step=30000`，SHA-256
  `8ed405767b851aaad98b550c203778ce41625fc2fbd43a170446595a38a73f9e`；
- 上游训练后的 full render 将帧常驻内存，r8 在 `577/588` 时触发 cgroup 90% 守卫；
  `oom=0/oom_kill=0`。r8 保持 `blocked`，r12 只复核并引用已完成的 step-30000 checkpoint；
- token-first registry 映射冻结 high-support actor
  `af663976db5e412e83db033d309c5c29 → true id 13 → dataset column 13 → rigid model 5`；
  正式 slice 含 `2,683` 个高斯。24 个原生 model 中 23 个非空，训练裁剪为空的非目标 model 14
  显式标记 `unavailable_empty_checkpoint_slice`；
- 3 帧 × 3 相机 × `original/lateral +1m/remove` 共 27 张 smoke 输出全部非空；checkpoint 文件、
  非目标 actor 参数和两次 reload 哈希均精确不变；编辑差异非零；
- final readiness 为 `available=11 / missing=0 / incompatible=0`。这些结果只证明编辑 API 可运行，
  不在 M3 作质量结论。

正式完成 run：

```text
/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M3-EDIT-BASELINE-01/20260802T163930Z__formal-checkpoint-recovery-s0-r12
```

---

# 9. M4：scene-0230 单场景真实编辑闭环

## 9.1 目标

第一次真正执行对象编辑，不再用“无身份”批量填充 ABSTAIN。

## 9.2 固定样本

- scene：`scene-0230`
- actor：M2 冻结的 `high-support`
- cameras：`CAM_FRONT / CAM_FRONT_LEFT / CAM_FRONT_RIGHT`
- frame range：与 baseline 有效 clip 对齐，映射表冻结
- edit：
  - `original`
  - `lateral +1.0 m`
  - `delete`

若横向正方向与道路语义无关，明确定义为 actor-local `+y`，不称 cut-in、换道或道路合法编辑。

## 9.3 输出

每个 variant：

- 三相机同步 MP4；
- 单帧 PNG；
- target actor mask；
- source footprint；
- edited footprint；
- depth/visibility overlay；
- actor transform JSON；
- world state hash；
- 渲染资源记录。

## 9.4 最低指标

### 对象运动

- 预期轨迹与实际 actor transform 的 SE(3) 误差；
- canonical point pairwise-distance drift；
- actor 尺寸漂移；
- 多相机投影一致性。

### 非目标保持

使用 `original baseline render` 作为 paired reference：

- non-target PSNR；
- non-target LPIPS；
- static background difference；
- 其他 actor mask 内差异。

### 原位置残留

- source footprint residual energy；
- 原 actor mask 内与相邻帧背景的差异；
- 删除后残留轮廓/阴影诊断。

### 新位置遮挡

- edited actor depth 与背景 first-hit depth 顺序；
- 新 footprint 内的 alpha/深度冲突；
- 三相机 visibility 一致性。

这些指标用于诊断，不预设方法通过阈值。

## 9.5 视觉 QA

必须生成一个可审计页面：

```text
原始 / 编辑 / 差分
三相机同步
source footprint / edited footprint
actor ID / frame / raw-or-interpolated
深度与遮挡顺序
```

页面不得只放文件链接；必须支持时间轴同步或至少按相同 frame 行排列。

## 9.6 M4 通过条件

- 两个编辑都真正执行；
- 输出不是全黑、空 actor 或重复 original；
- actor transform 与预期一致；
- 指标和可视化可定位失败；
- 没有因 unsupported 区域而把整条实验写成 ABSTAIN。

若 baseline 无法执行任一基本编辑，M4=`blocked`，不能进入 M5。

### 2026-08-03 完成事实

- 正式 run：
  `/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M4-EDIT-PILOT-01/20260802T171000Z__scene0230-pilot-s0-r7`；
- `196 frames × 3 cameras × 3 variants = 1,764` 张 RGB 全部生成，并同步保存深度、
  opacity、dynamic opacity、target mask、source/edited footprint、9 个 MP4 和内嵌图片的 QA 页面；
- 两个编辑各有 `588` 行 paired metrics，总计 `1,176` 行；16/16 运行与不变量检查通过；
- actor-local `+y 1.0 m` 的最大平移误差为 `3.814697265625e-06 m`，旋转、尺寸、
  canonical pairwise distance 和跨相机 world transform mismatch 均为 `0`；
- lateral/delete 的 non-target PSNR 分别为 `93.394483 / 95.598042`，LPIPS(Alex, 256px)
  分别为 `5.260851e-09 / 3.052960e-09`；这些值只证明非目标保持，不代表视觉质量通过；
- 正式 run 用时 `685.3 s`，峰值 GPU `8,543 MiB`、峰值 cgroup `58,478,706,688 bytes`，
  `oom=0 / oom_kill=0`；人工抽检确认三组渲染非黑，source/edited footprint 有预期位移，
  差分集中于目标区域；
- effect mask 是 `original/delete` 与 `lateral/delete` 的模型内反事实差分，不是真实观测真值；
  M5 仍必须执行 Tier A/B/C pseudo-hole/观测边界评测。

## 9.7 M4 commit

```text
feat(edit): 完成单场景对象编辑闭环
```

---

# 10. M5：三场景对象编辑、去遮挡和感知压力测试

## 10.1 目标

在真实可编辑 baseline 上获得跨场景失败矩阵，为 M6 提供证据。

## 10.2 固定 cohort

场景：

```text
scene-0230
scene-0242
scene-0255
```

每场景：

- `high-support`
- `boundary-support`，如不存在则如实记录 1/2 coverage

不得根据 M4 表现替换 actor。

## 10.3 固定编辑

正式编辑集：

| 编辑 | 参数 |
|---|---:|
| lateral | actor-local `+1.0 m` |
| speed | `0.75×` |
| stop/restart | 中段停止 `1.0 s` 后平滑恢复 |
| delete | 全轨迹删除 |

其他幅度只允许作为后续 sensitivity，不进入 M5 primary failure matrix。

## 10.4 预期样本量

最大：

```text
3 scenes × 2 actors × 4 edits = 24 sequences
每条 3 cameras
```

所有失败、ABSTAIN 和缺失都进入 coverage。

## 10.5 Truth tier

### Tier A：held-out observed

同一静态表面在其他时间或相机中有真实观测，能够通过 pose/depth 对齐到编辑后去遮挡区域。

允许：

- RGB/LPIPS；
- depth；
- detector consistency；
- temporal consistency。

### Tier B：geometric support

有 LiDAR、多视角或静态高斯支持，但没有同视角真实 RGB。

允许：

- depth；
- reprojection；
- visibility；
- multi-view consistency。

### Tier C：unsupported

没有可验证观测。

只允许：

- uncertainty；
- coverage；
- ABSTAIN；
- 人工审核。

禁止在 Tier C 报“补全准确率”。

## 10.6 pseudo-hole benchmark

不直接把真实反事实编辑当作有真值。

构造流程：

1. 从真实可见静态区域选择可在其他时间/相机验证的 Tier-A 区域；
2. 用冻结 actor footprint 或合成遮挡 mask 隐藏该区域；
3. 从训练/重建输入侧移除该观测；
4. 保留 held-out 真实观测只用于评测；
5. 比较：
   - no completion；
   - baseline 原生背景；
   - 2D framewise inpaint diagnostic；
   - 后续 proposed method（仅 M7 后）。

2D inpaint 只能作为诊断下界，不能成为 3D world state。

## 10.7 感知一致性

复用 M2 已固定的 Grounding DINO/Grounded-SAM 资产，或在运行前冻结一个公开 detector。禁止看输出后换模型。

报告：

- 非目标 vehicle detection matching IoU；
- 非目标类别/置信度变化；
- false disappearance；
- target actor 的预期变化；
- 三相机检测一致性；
- tracker 可用时的 IDF1/ID switch。

感知模型只作为 task-aligned evaluator，不代表真实安全性。

## 10.8 指标矩阵

### 轨迹与刚体

- trajectory SE(3) error；
- rigidity drift；
- scale drift；
- actor boundary temporal LPIPS。

### 原位置

- source residual；
- shadow/residual edge；
- background contamination；
- Tier-A disocclusion PSNR/LPIPS/depth。

### 新位置

- depth ordering violation；
- occlusion conflict；
- alpha overlap；
- multi-camera visibility disagreement。

### 非目标区域

- non-target PSNR/SSIM/LPIPS；
- static-region preservation；
- non-target perception preservation；
- temporal warp error。

### coverage

- per scene；
- per actor stratum；
- per edit；
- per truth tier；
- ABSTAIN reason。

## 10.9 failure taxonomy

至少包含：

```text
ACTOR_IDENTITY_MISMATCH
ACTOR_GEOMETRY_DEFORMATION
TRAJECTORY_NOT_FOLLOWED
SOURCE_RESIDUAL
SHADOW_RESIDUAL
BACKGROUND_HOLE
UNSUPPORTED_DISOCCLUSION
DEPTH_ORDERING_ERROR
NEW_OCCLUSION_ERROR
MULTICAMERA_INCONSISTENCY
TEMPORAL_FLICKER
NON_TARGET_VISUAL_DRIFT
NON_TARGET_PERCEPTION_DRIFT
BASELINE_RUNTIME_FAILURE
INSUFFICIENT_EVIDENCE
```

每条 sequence 可以有多个 failure code，但需要固定 primary failure 优先级。

## 10.10 M5 通过条件

“通过”不代表方法好，而是实验有效：

- 至少 3 个 scene 有可运行编辑；
- 至少 4 个 actor slot 有结果；
- 四类编辑均有 coverage；
- 不能因一个依赖缺失将全部结果写为 ABSTAIN；
- 失败矩阵、视频和指标一致；
- 至少一种失败在 3 个 scene 重复，或诚实结论为“没有稳定失败”。

## 10.11 M5 commit

```text
research(eval): 完成三场景编辑与去遮挡压力测试
```

---

# 11. M6：基于真实失败重新做创新性门禁

## 11.1 原则

M6 不允许再次选择“补持久身份 + actor binding + 基础轨迹编辑”作为贡献，该候选已被 V1 拒绝。

M6 必须先查看 M5 的全量 failure matrix，而不是挑成功或失败视频。

## 11.2 候选方向

### B：编辑诱发的可见性重算与证据分层去遮挡

只有在以下失败跨 3 scene 重复时考虑：

- source residual；
- background hole；
- depth ordering error；
- new occlusion error；
- unsupported 区域被无条件生成。

### C：非目标感知保持

只有在 RGB 全局指标尚可但非目标 detector/tracker 明显退化时考虑。

### D：前馈初始化 + 对象约束精修

只有 DGGT 与 AD-GS 对照显示稳定的速度/几何 trade-off，且能设计 matched protocol 时考虑。

### Reject

如果没有一个失败在至少 3 scene、至少两个 actor stratum 或多个编辑中重复，不注册方法，M6=`rejected`。

## 11.3 最新工作重新核对

M6 开始时必须进行一次新的官方来源 novelty audit，至少检查：

- InstDrive；
- Director；
- OmniRe；
- HorizonForge；
- G²Editor；
- DrivingEditor；
- VAD-GS；
- GA-GS；
- DenoiseGS；
- Perception-aware 3DGS；
- 2026-07-29 之后出现的相关工作。

只使用论文、官方项目页和官方代码。输出：

```text
claim
existing work
exact overlap
remaining delta
required baseline
required metric
novelty verdict
```

不得只比较标题或摘要关键词。

## 11.4 hypothesis 文档

若通过，只允许一个主假设，写入：

```text
docs/DR_V2_M6_HYPOTHESIS.md
```

必须包含：

- 观察到的跨场景失败；
- 因果假设；
- 与现有工作的实质差异；
- 最小模块；
- primary endpoint；
- guardrail；
- effect size；
- 3 seeds；
- matched baseline；
- 明确拒绝条件。

## 11.5 effect size

根据 M5 baseline 方差，在看到 proposed 结果前冻结。不得凭空写“提升 10%”。

建议 primary endpoint 从以下选择一个，不允许全部都当 primary：

- Tier-A disocclusion LPIPS；
- non-target perception preservation；
- depth-ordering violation rate；
- temporal warp error；
- risk-coverage/AURC。

## 11.6 M6 终态

### `done`

- 有一个稳定失败；
- 有独立 novelty delta；
- primary endpoint 有真值或明确 truth tier；
- M7 被授权。

### `rejected`

- 失败不稳定；
- 与现有工作直接重合；
- 无可信 endpoint；
- 不启动 M7/M8。

## 11.7 M6 commit

```text
research(hypothesis): 注册或拒绝 V2 编辑失败假设
```

---

# 12. M7：条件式方法与 matched ablation

本节只有 M6=`done` 时启用。

## 12.1 方法范围

只实现 M6 选中的最小机制。禁止把下列所有模块一次性堆入：

- 实例身份；
- 轨迹编辑；
- occupancy；
- 扩散；
- 物理；
- 感知损失；
- 不确定性；
- 前馈初始化。

一次只回答一个因果假设。

## 12.2 默认消融模板

若选择 B：

```text
A0 editable baseline
A1 A0 + explicit visibility recomputation
A2 A1 + observed/multiview/unsupported evidence typing
A3 A2 + confidence or abstention
```

若选择 C：

```text
A0 editable baseline
A1 A0 + non-target pixel preservation
A2 A1 + frozen perception feature/output preservation
```

若选择 D：

```text
A0 DGGT native
A1 DGGT initialization + baseline-native refinement
A2 A1 + object/visibility constraint
```

## 12.3 正式实验

- 同 scene、frame、camera、actor、edit、seed、预算；
- 至少 3 seeds；
- 报 mean/std、bootstrap 95% CI；
- per-scene、per-actor、worst-case；
- coverage 和 ABSTAIN；
- runtime、VRAM、RAM；
- 不删除失败 run。

## 12.4 通过条件

- primary endpoint 达到预注册 effect size；
- 95% CI 满足预注册标准；
- guardrail 不退化；
- 至少 3 scene 一致；
- 不靠降低 coverage 改善均值；
- 人工审核前结果和方法标签冻结。

---

# 13. M8：人工盲审与最终裁决

只有 M7 产生完整 proposed 结果时启用。

## 13.1 盲审样本

固定包含：

- 3 个 scene；
- high-support 和 boundary-support；
- original、lateral、speed、stop/restart、delete；
- baseline 与 proposed 随机 A/B；
- 三相机同步视频；
- target crop、full frame、depth/ID overlay；
- 成功与失败全部保留。

## 13.2 评审维度

- actor 身份；
- 刚体和外观；
- source residual；
- shadow residual；
- 去遮挡背景；
- 新遮挡；
- 深度顺序；
- 多相机一致性；
- 时间闪烁；
- 非目标区域保持。

每项：

```text
PASS | FAIL | UNCERTAIN
```

## 13.3 Codex 责任

Codex 必须交付：

- 完整审核提示词；
- 盲法和禁止读取的信息；
- 逐项 rubric；
- JSONL 模板；
- validator；
- 聚合脚本；
- 不可变文件 hash；
- debug 页面与 blind 页面分离。

Codex 不得填写人工 verdict。

---

# 14. 测试策略

## 14.1 单元测试

每个新模块至少具备：

- happy path；
- invalid input；
- provenance；
- deterministic selection；
- coordinate round trip；
- hash tamper；
- resume/overwrite protection。

## 14.2 集成测试

固定由小到大：

```text
synthetic actor
→ scene-0230 单帧
→ scene-0230 短序列
→ scene-0230 完整编辑
→ 3 scenes
```

禁止第一次直接运行 24 条正式编辑。

## 14.3 测试命令记录

每个 commit 正文必须包含实际运行的测试及结果。提交前：

```bash
git diff --cached --check
python -m pytest -q <relevant tests>
```

不允许在正文写未实际执行的测试。

---

# 15. 文档和提交规范

## 15.1 文档事实源

- 当前授权与下一步：`docs/RESEARCH_STATUS.md`
- 全部实验事实：`docs/EXPERIMENTS.md`
- 可复用失败：`docs/RESEARCH_FAILURES.md`
- 当前计划：本文件
- 原始日志：run 目录

文档只摘要，不把数 GB 日志复制进 Git。

## 15.2 Commit 拆分

建议：

```text
research(workflow): 启动动态编辑 V2 诊断路线
fix(dggt): 按上游构建 pointops2 并完成推理协议
feat(eval): 建立 nuScenes 持久 actor 评测适配器
feat(edit): 接入对象级驾驶场景编辑基线
feat(edit): 完成单场景对象编辑闭环
research(eval): 完成三场景编辑与去遮挡压力测试
research(hypothesis): 注册或拒绝 V2 编辑失败假设
```

禁止把所有里程碑压成一个 commit。

---

# 16. 总体终止条件

出现任一情况，V2 应结束而不是继续改题：

1. DGGT 安装修复后仍因明确上游问题无法运行，且可编辑 baseline 也无法建立；
2. 三场景没有足够可评测 actor，且问题不是映射/投影 bug；
3. 对象级 baseline 在冻结预算内无法产生有效编辑；
4. M5 没有跨场景稳定失败；
5. M6 候选与已有工作直接重合；
6. primary endpoint 没有真实或分层真值；
7. M7 改善只来自降低 coverage 或挑选样本；
8. 资源不足且用户未开放新资源。

准确终局必须区分：

```text
engineering_blocked
baseline_unavailable
no_stable_failure
novelty_rejected
method_rejected
method_supported
```

不得统一写成“项目失败”。

---

# 17. 预期最终交付

无论终局如何，必须交付：

1. V2 计划与完整更新日志；
2. AutoDL 环境 bootstrap 和镜像配置；
3. DGGT 原始失败、修复和推理报告；
4. nuScenes actor cohort 与投影 QA；
5. 可编辑 baseline 的来源、commit、环境和 checkpoint；
6. 所有编辑视频、指标和 failure matrix；
7. truth tier 和 coverage；
8. M6 novelty matrix；
9. 若有方法：消融、统计和人工盲审包；
10. 若无方法：准确的负结果和下一步边界。

---

# 18. Codex 首轮执行提示词

将以下内容作为执行入口：

```text
执行 docs/DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md。

严格遵守 AGENTS.md 和计划中的小步快跑、事实源、Git、资源与人工评审规则。
先只执行 M0，不得提前安装 DGGT 或启动 GPU 任务。

M0 开始前：
1. 读取 RESEARCH_STATUS.md、RESEARCH_FAILURES.md、EXPERIMENTS.md、归档 V1 计划、V2 计划和 AGENTS.md；
2. 检查 live repo Git 状态、当前 commit、远端 run 目录、GPU/cgroup/磁盘；
3. 创建独立 V2 分支或 worktree，保留用户已有 dirty/staged 内容；
4. 完成 V2 事实源切换、README 修正、AutoDL bootstrap 和项目级镜像配置；
5. 运行 M0 测试；
6. 更新 V2 计划、RESEARCH_STATUS.md、EXPERIMENTS.md；
7. 只提交 M0 相关文件；
8. 向用户报告 M0 证据、commit、run 路径和 M1 是否解锁。

不要一次越过 M0 和 M1。不要恢复 cut-in。不要修改 V1 已冻结结果。
```

---

# 19. 外部一手参考

执行时应以官方页面的当前内容为准，并将实际访问日期、commit/revision 与许可证写入 run：

- AutoDL 网络加速与软件源帮助；
- 清华 TUNA Anaconda/PyPI 镜像帮助；
- PyTorch 官方历史版本安装页；
- Hugging Face Hub 环境变量与缓存文档；
- DGGT 官方仓库与 nuScenes 数据说明；
- nuScenes 官方 schema/tutorial，尤其是 `sample_annotation` 与 `instance_token`。

本计划中的镜像用于加速，不改变固定模型、代码、数据或指标协议。镜像内容与官方来源冲突时，以固定官方版本和 SHA-256 为准。

---

# 20. 初始决策摘要

```text
保留：AD-GS 六场景 exact reproduction
修复：DGGT pointops2 构建和正式 inference
新增：nuScenes GT actor evaluation adapter
基线：DriveStudio/StreetGS actor-aware editor
先做：真实对象编辑与失败诊断
不做：cut-in 挖掘、AD-GS 伪实例恢复创新
门禁：跨 3 scene 稳定失败 + 独立 novelty + 可验证 endpoint
```

**V2 的成功标准不是“必须造出新方法”，而是用可运行、可审计的编辑实验把研究问题收敛到一个真实且尚未被解决的缺口。**

---

# 21. 里程碑更新日志

### 2026-08-02 — V2 计划创建

- 基于 V1 实际终态和 Motion-Proj V8 源码审计建立独立 V2 路线；
- 保留 AD-GS 六场景复现结果，不恢复 cut-in；
- 将 V1 M6 重新定性为“持久身份资产审计”，不再视为完整编辑压力测试；
- M1 首要工程任务固定为按 DGGT upstream `python setup.py install` 修复 pointops2；
- M2 引入 nuScenes `instance_token`，仅作为评测对象与真值轨迹；
- M3–M5 要求真实生成对象编辑视频和跨场景 failure matrix；
- M6 只允许根据真实失败选择新假设；候选 A 不得复活；
- 当前唯一授权里程碑：`DR-V2-M0-BOOTSTRAP-01`。

### 2026-08-02 — M0 `done`

- 新建分支 `research/dynamic-editing-v2`，V1 历史终态与冻结数值未改写；
- 根 `README.md` 已切换到 V2，`AGENTS.md` 已删除全局 `conda init` 路径；
- 新增 `configs/env/autodl_condarc_v2.yaml`、`scripts/bootstrap_autodl_v2.sh` 和
  `docs/DR_V2_M0_SOURCE_AUDIT.md`；
- 首个 smoke 实例
  `/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M0-BOOTSTRAP-01/20260802T114342Z__bootstrap-s0`
  因空 shell 中裸 `python` 不在 PATH 以 `blocked` 保留；修复只显式选择 Miniconda Python；
- source-snapshot 完整的正式实例
  `/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M0-BOOTSTRAP-01/20260802T115419Z__bootstrap-s0-r3`
  以 `done` 结束：空 shell/tmux shell 均通过，TUNA Conda/PyPI、HF mirror、GitHub 4/4 返回 200；
- AD-GS `model_60000`、official render 与 processed 输入 6/6 完整；DGGT preload 字节数与 SHA-256 通过；
- `python -m pytest -q tests/test_dr_pseudo_tracks.py tests/test_v71_actor_registry.py`：`7 passed`；
- DriveStudio source/env 可用但三个 pilot scene 资产缺失的现场事实保持不变；
- 下一里程碑：`DR-V2-M1-DGGT-REPAIR-01`。

### 2026-08-02 — M1 `done`

- 以固定 CUDA 12.1 Conda toolchain 和严格依赖约束新建 `dggt-v2`，pointops2 upstream
  CUDA forward/backward 通过；
- 未修改 upstream 推理的 `args.difix` 原始失败与单行 compatibility patch 分开封存；
- r6 完成 18/18 1-view 与 18/18 3-view，r8 完成 216/216 common targets，r9 完成
  504 行动态区/边界诊断；
- 包解析、CUDA toolkit、Torch API 与 common evaluator 依赖的失败实例均保留为独立
  `blocked` run，没有覆盖重跑；
- M1 门禁通过，下一唯一授权里程碑为 `DR-V2-M2-ACTOR-EVAL-01`。

### 2026-08-02 — M2 `done`

- 建立 nuScenes raw 2 Hz `instance_token` actor chain、明示 `T_camera_global`、近平面/图像边界
  polygon 裁剪与 AD-GS frame/render 映射；
- 预注册门槛未调节，`scene-0230/0242/0255` 合格 actor 为 `16/20/6`，每场景冻结
  high/boundary 各 1 个；
- 4,356/4,356 camera observations 以 timestamp+exact sample token 映射，11 个输入 metadata
  文件全部哈希；
- 6 组投影 panel 与 6 张 raw 轨迹图通过逐张身份/投影 QA，raw/interpolated 没有混写；
- r1–r4 的 devkit 反向索引、Decimal、invalid projection schema 与 token 精确映射失败
  均作为独立 `blocked` 证据保留；
- 正式 r5 为 `done`，下一唯一授权里程碑为 `DR-V2-M3-EDIT-BASELINE-01`。

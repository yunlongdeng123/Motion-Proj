# Motion-Proj V2 运行环境

- 更新时间：2026-08-02
- 研究授权：只看 [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)
- 当前计划：[`DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md`](DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md)

## 1. 当前机器

```text
GPU             NVIDIA GeForce RTX 3090, 24,576 MiB
driver          580.105.08
cgroup memory   96,636,764,160 B (90 GiB)
memory.events   oom=0 / oom_kill=0
data filesystem /dev/nvme0n1, 250G
```

清理前为 `185G used / 66G available`。清理后的实际值将在
[`archive/2026-08/v2-preflight/CLEANUP_MANIFEST.md`](archive/2026-08/v2-preflight/CLEANUP_MANIFEST.md)
完成后回填。

## 2. V2 环境驻留状态

| 环境 | 路径 | 用途 | 状态 |
|---|---|---|---|
| motionproj | `/root/autodl-tmp/envs/motionproj` | 主仓库、M0/M2 与轻量测试 | 保留 |
| drivestudio | `/root/autodl-tmp/envs/drivestudio` | M3 对象级 baseline | 保留；Python 3.9 / torch 2.1.2+cu118 |
| adgs | `/root/autodl-tmp/envs/adgs` | 读取/必要时渲染冻结 AD-GS | 保留 |
| adgs-sam | `/root/autodl-tmp/envs/adgs-sam` | M5 冻结感知 evaluator | 保留 |
| dggt-v2 | `/root/autodl-tmp/envs/dggt-v2` | M1 新隔离环境 | 尚未创建；M0 前禁止创建 |

V1 的 `/root/autodl-tmp/envs/dggt`、中断副本、ReSim V6 环境和 DPT 预处理环境属于可重建历史环境，
按 V2 启动前清理清单移除。环境被移除不改变历史 run 结论。

## 3. 代码与版本

| 项目 | 路径 | commit / 状态 |
|---|---|---|
| Motion-Proj | `/root/autodl-tmp/motion_proj` | V2 授权前基线 `1e83ad5b` |
| AD-GS | `/root/autodl-tmp/third_party/AD-GS` | `9a208512e49c8ddbaa20387921d9648adcd21cb4`；有已登记 compatibility 修改和 build 产物 |
| DGGT | `/root/autodl-tmp/third_party/dggt` | `a3276d2bbe4cbb03bcc117830b1836110a27adeb`；worktree clean |
| DriveStudio | `/root/autodl-tmp/third_party/drivestudio` | `e59bda4fa681f829dbb1d65f0de582b0f633c450`；worktree clean |
| Grounded-SAM-2 | `/root/autodl-tmp/third_party/Grounded-SAM-2` | `b7a9c29f196edff0eb54dbe14588d7ae5e3dde28`；worktree clean |
| Depth Anything V2 | `/root/autodl-tmp/third_party/Depth-Anything-V2` | source 保留；env/weight non-resident |

## 4. 数据与最终产物

必须保留：

```text
/root/autodl-tmp/data/dynamic_recon/raw_subset/adgs_nuscenes_v1
/root/autodl-tmp/data/dynamic_recon/manifests
/root/autodl-tmp/data/dynamic_recon/processed/adgs_nuscenes_v1
/root/autodl-tmp/runs/dynamic_recon/*/正式轻量证据
/root/autodl-tmp/runs/dynamic_recon/DR-M3-ADGS-0230-01/.../model_60000
/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/.../model_60000
```

39G AD-GS processed 数据暂时保留，因为现有
`scripts/prepare_dr_m5_dggt_inputs.py` 直接读取其 `image/sky/semantic`，M1 改造前不能删除。

DriveStudio 历史数据只有旧 mini/OccGS 资产，不能替代 V2 的 `scene-0230/0242/0255` processed data。

## 5. 权重

```text
DGGT full preload
path   /root/autodl-tmp/checkpoints/dggt_preload/model_latest_nuscenes.pt
bytes  5,411,266,466
sha256 fd15644b3a878849470cbf5f0f9eae39167cfec1b853092898ae754c4f3acde9
```

M1 必须先核对固定 revision、license 与 hash，再以同文件系统 hardlink 或只读路径复用；不得再次下载一份
5.41 GB 副本。旧 `/root/autodl-tmp/checkpoints/dggt/*.partial` 已列入清理。

Grounding DINO/Hugging Face cache 与 CoTracker3 权重保留供 M5 使用。DPT 权重属于已完成 AD-GS 预处理的
可重下载输入，V2 不再训练 AD-GS，因此改为 non-resident。

## 6. 缓存和镜像

V2 统一使用：

```bash
export PROJECT_ROOT=/root/autodl-tmp/motion_proj
export ENV_ROOT=/root/autodl-tmp/envs
export CACHE_ROOT=/root/autodl-tmp/cache
export CONDA_PKGS_DIRS=/root/autodl-tmp/cache/conda-pkgs
export PIP_CACHE_DIR=/root/autodl-tmp/cache/pip
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export HF_ENDPOINT=https://hf-mirror.com
export TORCH_HOME=/root/autodl-tmp/cache/torch
export XDG_CACHE_HOME=/root/autodl-tmp/cache/xdg
export TMPDIR=/root/autodl-tmp/tmp
```

- Conda/PyPI 默认使用项目级 TUNA 配置，不写全局配置；
- Hugging Face 默认使用镜像，同时固定 repo revision、记录 license、字节数和 SHA-256；
- GitHub 先启用 `/etc/network_turbo`；需要时允许用户授权的学术加速传输，但必须核对官方 remote、固定
  commit、submodule 和 license；
- PyTorch/CUDA 扩展使用官方兼容 wheel/index，镜像不能改变构建变体。

## 7. 激活与停止合同

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/<env-name>
```

V2 禁止执行 `conda init` 或写全局 pip/Conda 配置。GPU run 前按计划审计 GPU、cgroup、磁盘和进程；
低于 20 GiB、OOM/RC137、cgroup 90% 或需缩协议时立即停止并保留现场。

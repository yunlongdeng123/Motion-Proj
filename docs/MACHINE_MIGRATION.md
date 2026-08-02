# 换机 / 清空 context 后的 V2 接续指南

## 1. 唯一恢复顺序

```text
AGENTS.md
→ docs/RESEARCH_STATUS.md
→ docs/RESEARCH_FAILURES.md
→ docs/EXPERIMENTS.md
→ docs/DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md
→ 相关 run manifest/terminal
```

归档计划、旧 summary 和聊天记录都不授予下一步。当前唯一动作以 `RESEARCH_STATUS.md` 为准。

## 2. 代码

```bash
cd /root/autodl-tmp/motion_proj
git status --short --branch
git log -3 --oneline --decorate
```

若需要重新 clone，优先使用 `/etc/network_turbo`。用户允许学术加速传输 GitHub，但 clone 后必须把官方
repository URL、固定 commit、submodule 和 license 写入 run；不能把镜像分支名当 provenance。

## 3. 环境

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/motionproj
```

V2 禁止 `conda init` 和全局镜像配置。M0 将创建项目级 condarc 与 bootstrap；M0 完成前不得自行拼装
`dggt-v2`。

当前应驻留环境：`motionproj`、`drivestudio`、`adgs`、`adgs-sam`。历史 `dggt/resim/adgs-dpt`
已按清理账本移除，不能据此推断数据或历史实验丢失。

## 4. 关键非 Git 资产

```text
/root/autodl-tmp/data/dynamic_recon/raw_subset/adgs_nuscenes_v1
/root/autodl-tmp/data/dynamic_recon/manifests
/root/autodl-tmp/data/dynamic_recon/processed/adgs_nuscenes_v1
/root/autodl-tmp/runs/dynamic_recon
/root/autodl-tmp/checkpoints/dggt_preload/model_latest_nuscenes.pt
/root/autodl-tmp/hf_cache
/root/autodl-tmp/weights/cotracker3
```

DGGT preload 必须为 `5,411,266,466` bytes，SHA-256
`fd15644b3a878849470cbf5f0f9eae39167cfec1b853092898ae754c4f3acde9`。

六场景 AD-GS 只保留最终 `model_60000` 大载荷；100/1,000-step 中间 checkpoint non-resident 是预期状态。

## 5. 网络

```bash
source /etc/network_turbo 2>/dev/null || true
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

正式下载仍须固定 revision、记录 endpoint/命令/返回码/字节数/license/SHA-256。

## 6. 快速自检

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/motionproj
cd /root/autodl-tmp/motion_proj
pytest -q tests/test_dr_pseudo_tracks.py tests/test_v71_actor_registry.py
```

该命令只是现有代码 smoke，不代表 V2 M0 完成。M0 还必须建立分支、修正文档入口、生成 bootstrap、验证镜像
并更新三份事实源。

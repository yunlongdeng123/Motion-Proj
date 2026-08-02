# DR-V2 M0 source audit

- Task ID：`DR-V2-M0-BOOTSTRAP-01`
- 审计日期：2026-08-02
- 审计基线：`09fbb55`（`main`，V2 preflight 后）
- 目的：确认 V1/V7 代码的真实行为与 V2 可复用边界，不把历史占位或 ABSTAIN 路径误写成可编辑实现。

## DGGT pointops2

历史 runner `scripts/run_dr_m5_dggt.py` 的 `ensure_environment()` 使用：

```text
<dggt-env>/bin/pip install .
```

这会触发 `PIVOT-F14B` 已记录的 PEP 517 build isolation，临时构建环境无法导入已经安装的 PyTorch。V2 M1
必须新建 `/root/autodl-tmp/envs/dggt-v2`，优先按 upstream 在
`third_party/dggt/third_party/pointops2` 内执行 `python setup.py install`；历史失败 run 不覆盖。

## V1 M6 stress runner

`scripts/run_dr_m6_stress.py` 没有调用 renderer 或 trajectory editor。它先读取冻结 SAM mask 和 checkpoint
identity 审计，然后对所有对象编辑、pseudo-hole 与噪声行直接写 `ABSTAIN`。因此 V1 M6 是身份资产审计，
不是对象级编辑压力测试。

`motion_proj/dynamic_recon/pseudo_tracks.py` 的 `audit_mask_id_continuity()` 固定输出
`vehicle_eligible_count=0`：冻结 mask 没有类别字段，函数也明确禁止事后重关联。该逻辑应保留为 V1 负证据，
不能扩展为 V2 actor 真值适配器。

## V7 可复用边界

| 组件 | 可复用内容 | V2 必须补齐 | 禁止推断 |
|---|---|---|---|
| `motion_proj.resim.actor_registry` | 一一映射、连续 model index、canonical hash、tamper 检查 | 以 `instance_token` 为主键并保留 token↔整数 ID provenance | 现有 `true_instance_id` 自动对应 nuScenes token |
| `motion_proj.resim.drivestudio_adapter` | gsplat first-hit depth、global actor Gaussian mask | DriveStudio live checkpoint、dataset column、RigidNodes index 的实证映射 | adapter 存在即代表 pilot scene 可渲染或可编辑 |
| `resim/v71_build_world_state.py` | typed WorldState 与坐标合同 | V2 scene/frame/token mapping | V7 mini 资产自动适配 0230/0242/0255 |
| `resim/s0_trajectory_editor.py` | 轨迹编辑接口与状态变换模式 | actor-local `+y`、speed、stop/restart 的 V2 API 与可逆性验证 | 编辑命令已在 V2 baseline 上运行 |
| `resim/c0_counterfactual_render.py` | RigidNodes pose 注入和 typed render 思路 | 三相机同步输出、原 checkpoint hash、非目标 actor hash | V7 结果可作为 V2 真实编辑结果 |

## 资产与清理边界

`docs/archive/2026-08/v2-preflight/CLEANUP_MANIFEST.md` 明确删除的是 V1 失败环境、缓存、profiling checkpoint
和成功 run 的临时 work；六个 AD-GS `model_60000`、official render/metrics、39G processed 输入、raw subset、
DGGT full preload、DriveStudio source/env 均保留。M1/M3 不得把已登记的可再生清理项误判为 V2 输入损坏。

## M0 裁决

- V1 历史数值与 `rejected` 终态保持不变；
- V2 使用独立 task namespace 和 run 根目录；
- DGGT 修复属于 packaging/ABI 工程，不是方法贡献；
- V2 必须用 nuScenes `instance_token` 建立评测 cohort，并在 DriveStudio actor-aware baseline 上真正执行编辑；
- M6 通过前不得实现新方法。

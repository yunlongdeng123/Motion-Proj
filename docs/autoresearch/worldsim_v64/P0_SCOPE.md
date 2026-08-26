# WorldSim V6.4 P0 Scope

- Task: `WS-V64-P0-SCOPE-GIT-01`
- Status: `done`
- Branch: `research/worldsim-v6.4-native-uq`
- Upstream: `research/worldsim-v6.3-surface-tail@c192955`
- Plan commit: `ca930a0`
- Main integration: `origin/main@c192955`
- Integration branch: `origin/integration/worldsim-v6.3-to-main@c192955`
- V6.3 terminal: `v63_surface_architecture_family_closed_negative_p7_locked`
- Failure refs: `V62-F01,V62-F05,V62-F06,V62-F07,V63-F02,V63-F19,V63-F24`
- Failure delta: `V64-F01 resolved_pre_quality_read`
- V6.4 quality read: `none`
- GPU use: `none`

## Frozen inheritance

V6.4 保留 Native B2 为强基线，不重开 Surface-Mean/Max/CVaR，不读取 V6.3 legacy、calibration、confirmation 或 test。
新路线只允许使用原生 IR-WM logits/BEV feature，研究 aleatoric/epistemic uncertainty 与条件 authority；不使用 scene ID
记忆，不增加哈希、校验和或指纹产物。

## Validation

首次控制台 `pytest`入口在 collection 前失败；改用项目根目录下的
`python -m pytest -q tests/worldsim_v62/test_projection.py` 后得到 `1 passed in 1.59s`。本阶段没有增加其他测试。

## Resource observation

当前为单卡 RTX 3090 24GB，观测时 GPU 空闲；`/root/autodl-tmp`约剩余 60 GiB。该余量足够进入 native/UQ 最小开发，
但大规模 sidecar 前必须估算体积。

## Next

直接冻结最小 UQ 对比并盘点可复用 native sidecar；不先铺设冗长的协议或测试矩阵。

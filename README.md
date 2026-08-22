<div align="center">

# Motion-Proj

### Verifiable World Compiler for Editable Driving World Simulation

[![Status](https://img.shields.io/badge/WorldSim_V6-selector_family_frozen-2ea44f?style=flat-square)](docs/RESEARCH_STATUS.md)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![GPU](https://img.shields.io/badge/validated-1%C3%97RTX_3090-76B900?style=flat-square&logo=nvidia&logoColor=white)

</div>

Motion-Proj 是面向可编辑驾驶世界模拟的研究代码库。当前主线是 WorldSim V6：把场景、actor、轨迹、传感器与验证证据组织成可追溯、可复算、失败可保留的 world-compiler 链路。正式实验以干净且已推送的源提交、内容寻址产物、冻结分区、资源审计和 terminal manifest 为准。

## 当前结论

WorldSim V6 的 selector 研究族已在 R140 后冻结并收口。R141 未执行；后续不再继续 threshold=13/45、新 actor、新方向或其他 selector 变体研究。现有证据同时保留成功、拒绝和基础设施失败，不通过重跑或事后调参改写结论。

| 证据 | 结果 | 关键结论 |
| --- | --- | --- |
| R134 | rejected | StreetGS 的 threshold=13 不能直接迁移为跨 frontend 不变量；AD-GS frame 13 出现 1 个 FN。 |
| R136 | rejected, exact-once | AD-GS threshold=1 在唯一 heldout confirmation 中出现 1 个 FP；候选已消耗。 |
| R137 | accepted, development | AD-GS 157 帧 exact-input identity guard：0 false reuse，perception 调用减少 16.56%，628 个输出 hash 全部精确重建。 |
| R138 | failed, consumed | 负向量 CLI 绑定错误在 frame 0 前失败；没有方法结论，记录为 V6-F96。 |
| R139 | accepted, exact-once | 正交 +z 0.5 m 的 39 帧确认：0 false reuse，调用减少 17.95%，156 个输出 hash 全部精确重建。 |
| R140 H001/H002 | failed | Python 中的 JSON 风格 `false` 导致 closeout 失败，分别记录 V6-F97/V6-F98。 |
| R140 H003 | accepted | 仅完成 `false → False` recovery，得到完整 end-to-end wall-time accounting。 |

## R140 端到端计时

R140 把共享的 sensor rendering 时间同时计入 full 与 selective 路径，避免把 perception component savings 误报为整条 pipeline savings。

| 条件 | 端到端 wall-time reduction |
| --- | ---: |
| StreetGS selective execution | 13.5337% |
| AD-GS development identity guard | 11.1434% |
| AD-GS exact-once confirmation | 1.66365% |
| Macro | 8.78024% |
| Worst condition | 1.66365% |

三种条件均为零 reconstruction error。该结果说明 identity-only reuse 有正的端到端收益，同时也显示 AD-GS confirmation 中 sensor rendering 已成为主要瓶颈；本轮收口不继续寻找新的 selector 机制。

Canonical run：

```text
run://worldsim_v6/WS-V6-R140-CROSS-FRONTEND-END-TO-END-UTILITY-01/20260822T063937Z__end-to-end-utility-s20260821-r1
```

关键 SHA256：

- certificate: `913833af47e4171e27707f71418b6625ed358b538d1c8a5a18bca5ac7f585363`
- gate: `ac3c79c0e93f2932a076da8323b89a210ff2cbaac27ffa13079ce89ae9d07b51`
- summary: `50900ff99736055a10c32f4362176b7fc87862ae84667591077d6c17024e635b`
- manifest: `1cc753b3c0a9489ced2a58b23035466ee26cba963d2abc56c84ebd4d057e5a62`
- resource audit: `06c110236591529d5fef5f4178bfed696b6c0ad0cfbce94c497896dc92230265`
- terminal: `be263ba010cdb936fbd01dbfa0fe294b8022101aae348c95030d2a42d45fdb77`

## 证据入口

- [研究状态](docs/RESEARCH_STATUS.md)
- [V6 可验证 World Compiler 计划](docs/WORLDSIM_V6_VERIFIABLE_WORLD_COMPILER_AUTORESEARCH_PLAN.md)
- [Selector 研究族收口](docs/autoresearch/worldsim_v6/SELECTOR_RESEARCH_FAMILY_CLOSEOUT.md)
- [AutoResearch 状态](docs/autoresearch/worldsim_v6/AUTORESEARCH_STATE.json)
- [假设与结果流水](docs/autoresearch/worldsim_v6/HYPOTHESES.jsonl)
- [反思流水](docs/autoresearch/worldsim_v6/REFLECTIONS.jsonl)
- [失败记录](docs/RESEARCH_FAILURES.md)
- [R140 H003 accepted artifact](docs/autoresearch/worldsim_v6/R140_CROSS_FRONTEND_END_TO_END_UTILITY_ACCEPTED.json)

## 研究边界

当前证据支持的是：冻结输入与实现下的 selector 行为、exact-input 重用的逐输出重建等价性，以及包含共享 sensor 时间的单样本端到端成本核算。

当前证据不支持：

- 语义正确性、物理有效性、规划质量或安全性；
- 对所有 frontend、场景、actor、方向或编辑幅度的普遍化；
- replicated benchmark、统计显著性或生产吞吐保证；
- 将 rejected/failed 运行追认为 accepted。

## 仓库结构

```text
motion_proj/                  核心 Python 包与 WorldSim 实现
scripts/worldsim_v6/          V6 正式 runner 与隔离 worker
configs/worldsim_v6/          冻结实验配置
docs/autoresearch/worldsim_v6 内容寻址结果、状态、假设与反思
docs/RESEARCH_FAILURES.md     失败与拒绝的不可变记录
tests/                        回归与合约测试
```

## 基本环境与检查

项目依赖由 `pyproject.toml` 和 `requirements.lock.txt` 描述。正式 V6 证据是在单张 RTX 3090 24GB 资源合约下产生；不同 frontend 可能仍依赖各自冻结的上游环境和数据资产。

```bash
python -m pytest -q
```

不要仅凭测试通过宣称复现实验结论。正式复现还必须满足对应 config 中的 source hash、数据分区、资源、attempt-count、manifest 和 terminal 合约。

## 历史路线

V3.x、V4 与 V5 的实现和研究文档仍保留在 `docs/`、`configs/` 与 Git 历史中，但不再代表当前活动研究线。仓库最终以唯一远端 `main` 承载 WorldSim V6 收口状态。

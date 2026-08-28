# V6.6 Actor 资产审计与迁移边界

日期：2026-08-28

Task：`WS-V66-P0-HARP-SCOPE-01`

基线：`main@288fa9f`
结论：现有资产足以直接做 Actor-grounded development atlas 与确定性证书；无需更换 reconstruction backbone。

## 可直接复用的接口

| 资产 | 已有能力 | V6.6 用法 | 边界 |
|---|---|---|---|
| `motion_proj/worldsim_v62/evidence.py` | Actor hit/current/swept envelope indices 与 IDs；static/Actor evidence 分离 | Actor existence、sensor support、owned envelope factor | envelope 不是自然 artifact 真值 |
| `motion_proj/worldsim_v62/query_dataset.py` | query 到 Actor ID/current/swept support 的 grounding | 复用稀疏索引映射语义 | 不把 query label 当 Actor identity 真值扩张 |
| `motion_proj/worldsim_v6/sceneir.py` 与 adapters | Actor ID、轨迹、visibility、owned chunks、来源 | 后续 P4/P6 Actor-preserving bake | 现有 development package 不等于 fresh benchmark |
| `motion_proj/worldsim_v6/r13_dynamic_edits.py` | remove/translation/clone 与 collision dependency recomputation | artifact/hazard edit capability的实现参考 | 旧实现含历史完整性字段；V6.6 不新增同类字段 |
| V6.4 frozen q0 | hidden-FREE point risk | P1 的 actor-blind baseline diagnostic | 不作为 Actor legitimacy authority |
| V6.5 P10V evidence/native | 6 scenes、72 units、816 actor-unit envelope候选 | P1-D 开发 atlas；GPU q0 forward 与 CPU grounding重叠 | 已消费，只能 Tier-L mechanism |
| `motion_proj/resim/cutin_receiver.py` / `event_kinematics.py` | receiver、gap、closing speed、TTC、kinematic审计 | 后续真实 hazard edit/adjudication | `V1-F06`：稀疏 cut-in 不作主入口 |

## 第一轮迁移

P1-D 以 actor-unit 为 base case。每个 base 先从 METHOD evidence 统计 hit/current/swept support，并把 native
boundary q0/entropy/margin聚合到 Actor envelope；随后构造 paired validity 与 task-hazard属性。artifact family只通过
可观测 factor（support缺失、duplicate overlap、lifecycle gap、kinematic jump、shape jump）进入推理，family name与
artifact label只用于评测，不进入 certificate/model feature。

首轮 hazard intervention 只测试 artifact score 对任务危险属性的结构不变性，不宣称生成了物理合法 cut-in。真实轨迹危险
编辑必须后续接入 cut-in/kinematic证书并单独报告。

## 来源与许可边界

- 仓库未发现根级 `LICENSE` 声明；内部模块只在本项目研究分支复用，不对外重新授权。
- nuScenes 数据保持原地读取，不复制到 Git，不在 run 中发布原始传感器内容。
- V6.4/V6.5 model/run只以本机路径引用；V6.6 不复制第三方权重，不新增外部代码。
- Instant NuRec、UniSim、ReactSim 等只作为研究设计参考；P1-D 不搬运其代码或权重。

## 资源判断

P1-D 是既有72-unit sidecar的流式聚合，q0显存预计远低于1 GiB，native sidecar本身单worker历史峰值约4.13 GiB。
RTX 3090足够，不需要多卡。I/O使用双线程预取下一unit，GPU对当前unit做q0 forward，避免全量I/O barrier。

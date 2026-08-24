# WorldSim V6.2 P3 可行性投影收口

- Task：`WS-V62-P3-FEASIBILITY-PROJECTION-01`
- Hypothesis：`WS-V62-H-P3-001`
- 状态：`done_projection_contract_passed`
- Canonical：`run://worldsim_v62/WS-V62-P3-FEASIBILITY-PROJECTION-01/20260824T080731Z__projection-s0-r1`

## 实现

`motion_proj/worldsim_v62/projection.py` 接收最后一维为 `FREE/OCCUPIED/UNKNOWN` 的 logits。未约束 query 保留
softmax；约束 query 按以下优先级投影：

```text
contradiction or simultaneous FREE+OCC → UNKNOWN
observed FREE                         → FREE
observed OCC                          → OCCUPIED
outside actor lifecycle/envelope     → UNKNOWN
otherwise                            → soft prior
```

该约束逐 query 可分，因此第一版无需通用凸优化器或迭代 proximal solver。约束行对 logits 的梯度为零；未约束行保留
softmax 梯度。后续只有在训练实验显示局部时序/表面能量是明确瓶颈时，才能把迭代 residual 作为单因素 recovery。

## 精简验证

### Synthetic contract

```text
pytest -q tests/worldsim_v62/test_projection.py
1 passed in 1.84s
```

一个测试覆盖 hard FREE、hard OCC、FREE+OCC contradiction、显式 contradiction、lifecycle、simplex、finite gradient、
约束行零梯度与未约束行非零梯度。

### 真实 evidence fixture

输入为 V6.1 ME-0 scene-0048/f052 `O_method`：`3,600,000` static voxels、`28,248` actor voxels。从
FREE/OCC/UNKNOWN 各取 16 个 query，共 48 个；UNKNOWN 子集另外注入 simultaneous evidence、显式 contradiction 与
outside-lifecycle 三种控制。

结果：

```text
hard FREE max error                         0.0
hard OCCUPIED max error                     0.0
contradiction/lifecycle UNKNOWN max error   0.0
simplex max error                           0.0
gradient finite                             true
unconstrained gradient nonzero              true
fresh-process repeat match                  true
```

CPU only；没有读取 O_eval、confirmation/test，没有启动模型或训练，也没有新增哈希、校验和或指纹。

## 裁决

`projection_gate=PASS`。P2 围绕此 operator contract 构建 evidence-query dataset；不再为 P3 增加 smoke 或大回归。

# WorldSim V3.3 S5：语义门控渲染与删除回灌防护

- Task：`WS-V33-S5-SEMANTIC-RENDER-01`
- 终态：`done`
- canonical：`20260810T220500Z__s5-semantic-gate-canonical-s0-r4`
- production arm：`G0_raw_3d`
- enhancement 结论：`rejected_on_heldout_confirmation`
- R3D2：`blocked_pretrained_model_unavailable`

## 1. 目标与不可变边界

S5 不把通用 2D 网络放进删除主链。插入路径只允许：

```text
output = input + gate × clamp(Harmonizer(input) - input, ±12/255)
```

gate 由 S4 actor footprint 确定性生成，只覆盖 actor boundary、ground contact、shadow/seam support；
far non-target 权重为 0。删除 production 始终复制 `erase_background` 的 raw 3D render。D2 checkpoint、
actor registry、S4 package 与五视图 render 均为只读输入，不创建 checkpoint，也不执行 optimizer。

选择协议固定为：

1. edit target f091/c1 与 development f005/c0、f065/c1 选择 G0/G1；
2. 只有 development 先选中 G1，才读取 heldout f020/c0、f060/c1 作一次确认；
3. heldout 不调 gate、阈值、residual cap 或分支；任一确认门失败即生产回退 G0；
4. 冻结五视图不相邻，因此 temporal consistency 为 `not_evaluated`，不得外推视频时序质量。

## 2. 来源与条件式 SOTA

| 组件 | 固定来源 | 处置 |
|---|---|---|
| NVIDIA Harmonizer | commit `dd5799e50855c5bcb1f6ef52a77b5b644b4798c0`；Apache license SHA `58d1e17f...d8bd` | exported non-temporal JIT 只生成诊断候选；插入 residual 经 gate/cap 后才可写回 |
| Harmonizer model | `1,448,843,112 bytes`；SHA `ece8e2d...06e90` | before/after exact；无训练 |
| R3D2 | commit `3fc6e317d9fea9d800d3f8706554ad6ac794d980`；tree `759dd48...79c7`；Apache license SHA `43070e2d...79c1` | 官方仓库没有作者 exported pretrained pipeline；不加载 SD base 冒充 R3D2，不从零训练 |
| SAM2.1 large | commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`；checkpoint SHA `2647878d...d318` | 冻结 box prompt；检测 deleted actor semantic mass；checkpoint before/after exact |

R3D2 的阻塞只表示当前 pretrained inference 前置条件不存在，不是算法质量负结论。

## 3. 生产安全机制

每个 view 同时保存 `full_raw/full_unconstrained/full_gated` 与
`delete_raw/delete_unconstrained/delete_production`。生产删除图与 raw 文件 SHA 相同；SAM2 对相同内容复用
同一 logits 文件，使像素 fallback 与语义安全证明处于同一内容寻址合同。

五视图结果：

- `changed_far_non_target_pixels=0`；
- maximum applied residual=`12/255`；
- actor interior L1 delta=`0`；
- delete raw/production pixel SHA exact=`5/5`；
- delete production semantic mass/fraction delta=`0/0`，safe=`5/5`；
- unconstrained candidate 在 f091/c1 的 semantic mass/fraction 增加
  `+0.126399/+0.133885`，被 detector 标记；其余四视图未标记。

这证明 detector 确实能拦到一种实际回灌，而 production safety 不依赖“候选恰好没出错”。

## 4. 开发选择与留出确认

development 三视图的平均 L1 delta：

| region | G1 - G0 L1 | 裁决 |
|---|---:|---|
| boundary | `-1.837229` | 改善 |
| ground contact | `-2.771866` | 改善 |
| actor interior | `0.000000` | exact preserve |

因此 development 按预注册规则选择 `G1_semantic_gate`。随后只读 heldout：

| view | boundary delta | contact delta | shadow delta | actor interior |
|---|---:|---:|---:|---:|
| f020/c0 | `-0.090179` | `-2.424960` | `-2.517318` | `0` |
| f060/c1 | `-1.648594` | `+0.422686` | `+0.638112` | `0` |

f060/c1 的 contact 退化超过冻结上限 `+0.25`。没有放宽上限，也没有按五视图平均值掩盖单视图失败；
G1 被拒绝，五视图 production insertion 全部回退 `G0_raw_3d`。S5 工程与安全合同完成，但不声明
semantic-gated Harmonizer 已获得泛化增益。

## 5. canonical 证据

```text
/root/autodl-tmp/runs/worldsim_v33/WS-V33-S5-SEMANTIC-RENDER-01/
20260810T220500Z__s5-semantic-gate-canonical-s0-r4
```

| 产物 | SHA-256 |
|---|---|
| config | `b3848289add5e0f401d7386abf3e72caed80d3fa126b63a34694787463b18c89` |
| input manifest | `939a829eac74014ff913eb8d02058ef83166a576c8b93e89d5b7689bd58a635c` |
| Harmonizer manifest | `1da253d85e98babc1a8b33187f48cfe4b1a7a6c712cacc5cc25886e836913863` |
| SAM2 detector manifest | `c03fe7c9c4c25d56fc256d9c3328ecc70453b2daef05f4a61f2ed76da3c58b19` |
| decision | `988b6647a0d2a17a58d82b53b0c54c5e9854ba37a9ec8c4511f4d2b2cde6159d` |
| summary | `1e0bfb59602a012c799c94d2c18e9e0a35bfa09ecc3c05adbce2e22c37160761` |
| status | `969bb00995b592889803b9b8a147096ddde61037c250e4608d609d05cbe6fb97` |

资源：

- Harmonizer wall=`30.180697 s`，peak NVIDIA sampled/torch reserved=`3,553/3,940 MiB`；
- SAM2 wall=`5.701133 s`，peak NVIDIA sampled/torch reserved=`2,399/2,070 MiB`；
- run bytes=`34,548,858`；OOM/oom_kill delta=`0/0`；
- 单阶段 wall `<300 s`、peak NVIDIA `<12,000 MiB`、run `<256 MiB`。

r2、r3 与 r4 独立执行；30 个 RGB 产物在三次 run 中 SHA 全相同，decision SHA 相同。r3 仅修复
SAM2 输入 NumPy 只读 view 警告；r4 仅清理 input-prep EOF 空白并确保 source snapshot 与提交态 exact，
均没有改变协议、阈值、图像或裁决。

## 6. 失败 run 与测试

- r1：准备与 Harmonizer 成功；SAM2 冻结环境缺 SciPy，而共享模块在顶层导入形态学依赖，terminal=`failed`。
  修复为只在 gate 构建函数中延迟导入 SciPy；没有安装或污染 SAM2 环境。
- r2：完整 `completed/accepted`，首次得到 `rejected_on_heldout_confirmation`；SAM2 报只读 NumPy warning。
- r3：消除 warning 后重跑，图像与 decision exact；input-prep source snapshot 比提交候选多 EOF 空白，降为 noncanonical。
- r4：清理空白后的 canonical 重跑；8 个 source snapshots 与提交候选 byte-exact。
- S5 专项：`8 passed`；V3.3/V3.2 定向回归：`80 passed`；全部 Python `py_compile` 与 launcher
  `bash -n` 通过。

## 7. 结论与 R0 输入

S5 的突破是把“视觉增强”改造成可证伪、可回退的 optional delta，而不是强行让通用生成网络进入
WorldSim production：development 收益不能越过留出退化，删除语义回灌不能越过 raw 3D fallback。R0 应登记：

- production renderer=`G0_raw_3d`；
- semantic-gated G1=`rejected_on_heldout_confirmation`；
- delete semantic reintroduction prevention=`supported`；
- R3D2=`blocked_pretrained_model_unavailable`；
- temporal claim=`not_evaluated`。

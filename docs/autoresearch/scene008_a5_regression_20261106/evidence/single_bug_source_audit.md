单车源码调查尚未确立唯一的 a5 缺损根因。已找到一项可独立复现的屏幕半径裁剪实现缺陷；父代理 CPU probe 另已复现 static 分支吞掉 `time_shift`，现列为候选接线缺陷，原始设计意图尚未确定。坐标、点时间、直接 NPS 覆盖及已保存 a5 谱系的 SGCM 误裁假说未通过证据检验。调查仅做 CPU/只读操作，新增训练更新为 0；运行中的双车组未变更。

[架构图：输入、组件与数据流](../architecture.svg)

远端 engine：`/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018/training_source/scene0008_089_137`。下文 `E/` 均指此绝对路径；逐条源码 SHA、精确行号和完整 CPU 失败例保存在同目录 `single_bug_source_audit.json`。

| 判定 | 证据与边界 |
|---|---|
| 确定实现缺陷：screen-radius 统计失效 | `E/scene/gaussian_model.py:574` 在 postfix 清零全部 `max_radii2D`；`:698–721` 先 clone/split 再检查半径。原方法 CPU 适配中，半径 `[100,1]`、阈值 20、无新增点，正常路径删 0，`prune_only=True` 对照删 1。它保留过大 splat，尚不是 a5 缺损的因果证明。 |
| 已复现分支行为 / 候选接线缺陷：static time_shift | `E/gaussian_renderer/__init__.py:265,341` 无条件覆写输入；dynamic `:189–197` 使用输入。父代理真实环境 CPU probe 记录原10K的5,119次非零shift，static输出变化为0，dynamic变化非0；候选修复不改变None-shift输出。见 `static_time_shift_cpu_probe.json`。代码可追溯至上游 `c49e90c`（2024-12-29）；尚无设计文档证明 static 必须与 dynamic 使用相同自监督；旧实现可能基于静态分支假设，原意未定。隔离训练验证仍另行计账，a5 因果未知。 |
| 准备合同缺口，未证程序错误 | 原 `prepare_single_input.py:24–28` 仅复用 600K scene PLY、复制 a37 完整 PLY。a5 完整 21,358 源点只剩 1,442 个同时间匹配源点、1,496 个 sampled 行。`:65` 明文承诺 original B temporal static，因此不能把“不补齐完整 a5 PLY”冒称违反既存配方。原验收 `audit_static_seeds.py:25` 只要求剩余大于 0，无法证明支撑保持。 |
| 设计假设冲突 | `E/utils/hybrid30k_losses.py:61–75,140–162` 的 static 时长项偏好长寿命、速度项偏好低速度。a5 的真实 world 中位速度为 7.819 m/s，但本组把它放入 static；RGB 分项由 global1+dynamic5 变为 global1+static1。实现符合当前公式，不能称为新的代码 bug。 |

同一固定 121,128 像素帧域上，10K 与 11,977 的 KEEP MAE 从 **0.062096** 降到 **0.043573**，相对下降 **29.8%**；两组均完成 98 张 native KEEP/DELETE 精确重放，最大 uint8 差为 0。此时 a5 平均总 alpha 从 0.97436 升到 0.98686；高总 alpha 也可能由车后的地面等内容填充，不能证明车体几何完整。

| 实际 checkpoint 的 a5 源谱系 | 10K | 11,977 |
|---|---:|---:|
| 幸存初始祖先 / 初始 1,496 | 1,094 | 1,093 |
| 后代高斯行数 | 1,927 | 3,488 |
| peak opacity 中位数 | 0.02984 | 0.05953 |
| 时间 sigma 中位数 | 0.15361 | 0.05143 |
| SH degree | 3 | 3 |
| a5 谱系落入 SGCM 区域 | 0 | 0 |

这段延伸多了 1,977 步，包含 19 次普通增密节点（10,100 至 11,900）。最近一次 static-only opacity reset 在 9K，下次在 12K；保存终点 11,977 **尚未触发 12K reset**。`E/train.py:781–783` 只重置 static，`E/scene/gaussian_model.py:414–417` 将峰值 alpha 压到不超过 0.01 并清除 opacity Adam 矩。端点的后代增加、opacity 恢复和时间局部化与改善方向一致，但不能分离普通 SGD、增密、prune 和 reset 恢复各自的贡献。早期另存 single1K 就已有 roof 未重建，早于首次 3K reset；所以不能把 9K reset 说成缺陷最初形成的原因，也没有证据证明屋顶曾完整形成后再消失。

数据和归属排查有三个可重复的阴性结果：

- 21,358 个 a5 原始 PLY 点逐点等于 source LiDAR indices 经本帧 pose 变换后的 world float32 坐标；`nx` 逐点等于该帧 model_time。1,496 个 sampled a5 static 行的时间错配为 0。model time / physical second 约为 0.2，坐标与时间归一化一致。
- single 的 0.1m 动态排除没有额外删除任何已匹配 a5 static 行。其 source 稀疏发生在原 600K scene 采样与动态分支退出，而非这次 a37 最近邻排除误删。
- a37 NPS mask 加四邻域与 a5 的 121,128 固定像素帧重叠为 0；a5 精确源谱系在初始、10K、11,977 的 SGCM inside/prune/clone 均为 0。这不排除间接参数耦合，也不证明已消失谱系在所有未保存时刻都未进入 SGCM 区域。

另一个已验证行为是：即使 `time_split=False` 且 `no_time_split=True`，空间 split 仍随机改变子点时间中心；CPU 例中父时间 0.123 变成约 -0.33383、0.23751，而 sigma 保持 0.3。代码在 `E/scene/gaussian_model.py:628–649`。缺少禁止中心 jitter 的明确方法合同，故目前只记录行为，不列为已确立 bug。

最小后续验证应围绕现有可证失败例进行。static time-shift 使用父代理已准备的同源 10K 完整状态和既存未修 1,977 步分支作隔离对照；需匹配相机/shift 日程并分别核对增密引起的随机流分叉。screen-radius 修复应先保持旧源码冻结并单独验证拓扑影响，因为 11,977 实际有 1,038,376 行累计半径超过 20px，直接启用原阈值会大幅改变场景。完整非选中 a5 源点补齐应先明确数据支撑合同，再作单独对照，不能与 renderer 修复混在一起。本审计未启动这些 GPU 验证。

可交付 CPU 复现源位于 `../source/probe_screen_prune_cpu.py`、`../source/probe_a5_lineage_cpu.py`、`../source/probe_a5_source_time_cpu.py`。它们仅执行 CPU/只读计算，使用既存 drivingeditor 环境和原始远端路径；没有提供一揽子未验证修复。父代理已在真实 drivingeditor Python 中独立执行该交付版屏幕半径脚本（`CUDA_VISIBLE_DEVICES` 为空），结果为 PASS、CUDA 未初始化、无优化更新；阈值反例仍为两点中普通路径删 0、`prune_only=True` 删 1。交付源 SHA `4e0b9a26dc15aa56f66f03b3c626698002c113374c1aee72d88ad19b4e13eef8` 未变；执行凭据见 [screen_prune_cpu_delivery_probe.json](screen_prune_cpu_delivery_probe.json)。此独立执行仍不证明 a5 因果。

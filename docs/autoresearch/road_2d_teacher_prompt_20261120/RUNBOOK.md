# 二维教师默认配置与下一轮推理

用户于 2026-11-20 审阅后采纳整帧 1152×768＋注意力 LoRA T0，认为当前教师整体质量合格。此决定记录在 [teacher_defaults.json](teacher_defaults.json)。上一轮原始实验合同、指标与报告保留原样。

![组件与数据流](architecture.svg)

## 固定方法

SD1.5 inpainting；T0 step 600；注意力 LoRA rank/alpha 4/4；DDIM 20 步，CFG 3，eta 0，strength 1；BF16 运行，fp16 权重来源，LoRA FP32。整帧输入为 1152×768，合法输入构造和 W/K 范围不变。保留教师生成原图，其他监督视图从源图直接采样并使用已有完整支持规则。

## 已执行合同

[next_round_config.json](next_round_config.json) 固定六个任务、五个 seed（101、202、303、404、505）、两种提示词。30 个任务/seed 配对，总计 60 次推理。

- 旧提示词保留为基线。旧提示词 × 两个已有 seed 的 12 次结果精确重放，计入这 60 次，不把它们算成新 seed 证据。
- 新提示词去掉固定灰色要求，强调周围材质的颜色、亮度、细密随机纹理，并要求标线延续有可见依据。该措辞仍是软条件，不保证模型遵守。
- 两种提示词使用同一数值 seed、画幅、输入、掩码和采样参数；除文本外保持相同。扩展 seed 检验随机噪声稳健性，不增加独立场景数。
- 先完成一轮。第二轮仅在存在明确遗留问题时采用一个固定的有针对性改词；最多两轮，不逐图换词，本轮不依据结果继续追加 seed。第二轮不自动启动。
- 本轮没有训练、LoRA更新、精度对照、新 VAE 控制或三维渲染修改。

## 评价与后续选优

生成阶段不得读取合成 GT。所有预测完成、原图和索引冻结且范围检查通过后，才进入独立评价。配对报告颜色、亮度、随机纹理、错误标线、周期条纹和 seed 间质量分布，不能只展示最好的一张。

若五个 seed 的质量仍明显不稳定，再设计候选评分模块。先排除结构错误和非法改写，再评价材质匹配；不能用高频能量最大代替质量最高。逐图、逐 seed 的候选选优不使用隐藏 GT。合成 GT 可用于预测冻结后的整组提示词诊断比较，以及后续独立场景上的评分校准；当前两张底图不能用来调权后再自证泛化。

真实遮挡区域没有隐藏 GT，相邻可见道路仅是材质参考。教师配置被采纳不等于三维渲染问题解决。

## 真实 GPU 执行

用户随后明确远端已开机、保持开机，并要求跳过独立 CPU 预检阶段，直接真实 GPU smoke，通过后并行执行。已在 RTX 4080 SUPER（32760 MiB）上完成：

- smoke：两个真实任务 × 两种提示词 × seed 101，共 4 次，49.059 秒；其中两张旧提示词三类 PNG 均与历史逐位一致，实际输出保存及回读通过。
- 正式：4 个进程各 15 次，共 60 次，墙钟 292.856 秒；12 个历史重放、全量逆变换及 W/K 像素范围门通过。
- 总计 64 次扩散，训练、额外 VAE 控制和三维调用均为 0。采样到的 GPU 总显存峰值为 26458 MiB，四进程运行期间利用率约 99–100%。
- 实际 CLIP 编码包含 BOS/EOS：旧提示词 23 token，新提示词 40 token，均未截断。加载过程内部执行必要输入约束检查，没有单独停留在 CPU 预检阶段。
- smoke 的四个候选与正式结果三类 PNG 再次逐像素一致；四个 worker 与 smoke 使用相同执行源码。凭据见 [execution_audit.json](evidence/execution_audit.json)。

远端目录：`/root/autodl-tmp/drivingeditor_repro/campaign/road_2d_teacher_prompt_20261120_r1`。用户本次明确要求保持开机，本轮不执行关机。

此前本地计划使用 Windows 文件 pin，仅供当时的合同验收；本次已在远端重新 `prepare`，正式使用 `prepared_r1/plan.json` 的 Linux 文件绑定。本地旧计划不可直接复制过去运行。

原始远端依据：

```text
/root/autodl-tmp/drivingeditor_repro/campaign/road_2d_teacher_hf_cpu_20261119_r1/prepared_r2/pilot_manifest.json
/root/autodl-tmp/drivingeditor_repro/campaign/road_2d_teacher_hf_cpu_20261119_r1/GPU_pilot_r1/predictions.json
```

GPU smoke 与正式分片复用 [teacher_engine.py](code/teacher_engine.py) 的生成路径，入口 [smoke_and_workers.py](code/smoke_and_workers.py)；[launch_parallel.py](code/launch_parallel.py) 在真实 smoke 通过后启动四进程，收齐后合并并执行全 60 像素检查。推理入口内部核对来源、checkpoint 和真实 CLIP token；超长文本直接停止，禁止静默截断或自动改词。显式 `--execute-gpu` 才允许模型推理。

原 `score_pilot.py` 固定 48 组，历史源码保留原样。已新增 [score_next_round.py](scoring/score_next_round.py)，对完整 60 组先执行实际像素门，再读取独立 GT；指标实现沿用历史代码。已完成全部正式评分与五 seed 汇总，见 [scores.json](evidence/scores.json) 和 [comparison.json](evidence/comparison.json)。汇总曾因单进程/四分片回执结构不同而报错，已修复回执读取；原始预测、评分、GPU 结果均未改动或重跑。

## 本轮结论

合成道路诊断（两张底图等权，越低越好）：旧提示词到新提示词的 ΔE76 为 2.239597→2.242254，共同三频带偏差为 1.201894→1.141373。20 个配对中，颜色改善 7 个，频谱改善 8 个，两者同时改善 3 个；频谱均值收益集中在一个任务，不能视为普遍改善。平均任务内 seed 标准差也未降低。真实任务只评价已知可见域，不作隐藏道路准确性结论。

本轮不足以支持新提示词整体稳定胜出，默认方法仍为用户采纳的 1152＋注意力 LoRA，活动提示词暂保留历史版本。没有逐 seed 选优，也不追加提示词变体。改词同时涉及颜色、材质和标线措辞，不能将效果因果归因于其中某一个词段；假标线问题仍须结合原图审阅。

保留全部结果和工程失败现场，完成独立评价、原图审阅、报告、提交推送及取回核验。电源策略遵循用户本次明确指令：保持开机。小时级后续实验仍另遵守约十分钟真实 smoke 与保存/评价/完整恢复的要求。

三臂解释准备已完成，V 的终点结果尚未生成。材料准备时已经知道短300步、B1/B2终点和V首步核验结果，因此只声明保留既有锁定协议，不声称这是完全盲法预注册。原 λ=.05、11项主指标、6域上下文、49帧、support阈值和推进规则均不更改。

```text
同一30k完整状态 + 原始输入
            ↓
     B1原版 / B2原版重复 / V单项候选
            ↓
   真实2000步 + 49帧固定域 + 四件终态
            ↓
  component唯一canonical三臂指标与重复差异
            ↓
 独立reader核对 → 每帧代价与固定真实图 → 有限范围解释
```

`verify_actual_three_arm_metrics_cpu.py` 的默认模式仅核对冻结工具/协议并等待；没有V端点输出或占位结果。真实V完成并导出后，`--completed` 对实际2000日志、恢复、首步、输入、GT/域、四件终态证据，以及 canonical `three_arm_locked_metrics.json` 的11主＋126上下文／V-B1／V-B2／所有49帧逐项核对。均值／分位数的独立CPU复算只容许1e-12舍入差；原首十标量、源码、逐帧原值、域count依然精确相等。这个舍入检查不是质量阈值。

它保存全部49帧全图composite、actor composite与observed static MSE差，并单独指出最大不利变化帧。额外例图明确标为端点结果选择的不利样本，同时保留预定raw0/10/17/20/24/30/40/48与完整49统计。图像和效果解释由实际查看完成，不由reader自动宣布正常或通过。

`measure_actual_removal_locality_cpu.py` 只在remote的真实V完成之后读取既有NPZ，补充此前已写入独立解释协议的 composite↔static差值。它沿用原actor的PIL nearest和sky的OpenCV linear解码，核对全部实际六域counts，以原mask文件SHA绑定输入。指标为 `mean(abs(clip(composite)-clip(static)))`，支撑为两图分别量化到8位后差值任一通道>2。全49等权和像素加权分别报告；不改canonical主指标。

这份局部性测量相对于原composite而非dense edited GT。body-mask外可含原车影子，移除影子也可能是合理编辑；遮挡区没有dense GT。重建mask解码和域counts不等于独立保存的runtime mask像素或actualScene pose快照。已有NPZ/checkpoint大哈希由真实CPU导出receipt绑定，本补充不重复全部大文件哈希。

最后解释须同时考虑V对B1与B2的全49逐帧方向和幅度、一次原版重复分布、composite与目标车辆代价、动态背景支撑、静态foreground/environment与真实移除局部性。均值正负抵消和挑选好图不能替代这组证据；一次重复不是显著性检验或通用噪声界。后续训练仍需root另行批准。

**连续深度与前景 RGBA 固定实验审查**

已完成 C2 深度后验、固定 matte RGBA 组合、相同 40BUILD 像素域的 32 步 smoke 与 1000 步颜色拟合，以及 49 帧原生渲染评价。此处的 PASS 表示保存、恢复、导出、重载和评价工程通过；不等于自动采用，也不表示达到了参考图质量。原 PVD 权重未训练，新的 chart 几何确实改变；颜色拟合阶段才冻结该几何。

![architecture](architecture.png)

新 canonical 资产为 `observed_fit/formal/complete_a6.npz`，45,314 行。生成 chart opacity 为 `.99 * sampled_alpha`，真 106 行保留 `.99`；alpha 零的非真实前景点不出生。初始颜色来自 prototype 与固定 matte 采样，没有继承旧粗糙深度的正式拟合色。394,206 个 BUILD RGB 目标与原 HR 拟合逐字一致；新资产所有这些像素 alpha > .9804，没有按新覆盖率缩减目标。25,095 个零贡献颜色列保留初值。

|固定指标|旧 projective|新 C2 + matte|
|---|---:|---:|
|HR all49 straight RGB MAE|0.042121|0.041305|
|HR HELD9 straight RGB MAE|0.040365|0.039359|
|native480 all49 black RGB MAE|0.065308|0.072399|
|native480 HELD9 black RGB MAE|0.065848|0.072624|
|native480 all49 LiDAR z MAE (m)|0.317099|0.201782|
|native480 HELD9 LiDAR z MAE (m)|0.302434|0.189463|
|area480 all49 black RGB MAE, 双方均补偿|0.048629|0.047182|
|area480 all49 LiDAR z MAE (m), 双方均补偿|0.270509|0.168410|

固定面积补偿是同一 canonical 资产的 viewport 对照，没有第二次颜色拟合。它在 HR 逐字恒等，在 480 降低 chart opacity；新资产 visible joint alpha 均值由 native 的 .998828 降至 .976360。原生 negative FoV、投影、轨迹和编译 rasterizer 保持原样。HR 和 area 指标有所改善，native480 RGB 则退化，必须分开报告。

深度连续性在相同 erode1 支持内的相邻边 p99 从 .237219m 降至 .027087m，>10cm 的边从 5684 降至零。但 2210/42536 (5.196%) chart 点移动超过 .6m，最大前移 2.70124m，锚点最大残差仍为 2.62045m。大移动点只有 270 个 prior alpha≤.9，中位 alpha=.99746、中位距 matte 边界40.38HRpx，不能解释成只修低 alpha 边缘。相同点集的 out-of-box 总数1231→1221，chart1075→1065；最大 chart 轴向越界 .15376→.15809m。没有为改善这些数字加硬边界或重调 sigma。

![depth](depth_posterior.png)

背景、a37、道路源参数冻结不等于最终成像不变。native 新旧对照中，a37 keep 的 RGB 变化最大 .015596，536128 固定像素中零个>.02；道路 keep 有6/1075281像素>.02；outside keep 有6043/6312448像素>.02，最大变化 .723011；outside delete 有9004像素>.02，最大 .738736。道路 reobserved delete 的 RGB 误差 .020423→.020120，但相对最早 thinM2 的 .018075 仍较差。所有具体 mean/max/阈值计数、固定分母和深度变化保留于 `review.json` / `observed_fit/evaluation/summary.json`；删除状态 a37 像素没有当成隐藏道路真值。

106/113/130 和105的实际黑底、keep/delete街景、同相机crop与RGB/alpha/depth replay保存在 `observed_fit/evaluation/raw*/`。HELD9 不进入本次颜色拟合目标，但 raw113 等此前已经用于薄片、球核、表面表示缺陷诊断与内部方法开发；这些数值属于内部比较，不能称独立未触碰的泛化测试。原评价 receipt 的历史 final-only 标记与全部原始指标保留，由 `evaluation_roles.json` 明确修正解释。`HR_summary` 中 initial 只覆盖四个预览视角，不能当成 all49 对照。

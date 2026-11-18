# TRELLIS 高频初始化：CPU 准备

![架构组件](architecture.svg)

本支线只复用上轮一次 TRELLIS.2 推理的固定 mesh、逐面 UV 和无损 baseColor。新推理、模型微调、原生渲染、场景训练及电源操作均为零。最终输入是受限的 H 参数初值；A₀底色、低频场、共同 A200 状态和原观测台账不由本支线改写。

原参考只来自raw86/110，13,109/16,384节点有效（真实A为1,726、C/S伪教师11,383）；3,275个未知节点使用显式灰占位。raw134/179/198未作输入教师。继承199帧BUILD先验，本轮不声明盲测。

旧棕色块未发现新的可验证 UV、V方向、通道、尺度或 alpha 软件错误。原始 PNG 均值 RGB 已为(0.5066,0.3926,0.3075)，暖色先于世界映射出现；实际世界几何身份误差5.27×10⁻⁸m。旧 GH 硬边界积分问题是真实工程错误，已在上轮 midpoint 对照中修复。直接把 PBR baseColor 接到 native sRGB、未知灰参考的条件影响、未分离光照仍是限制，不证明模型无材质适配能力。

固定名义2.56m路块按真实面 UV 采样到2.5mm平均切平面网格，再用σ=2cm Gaussian抗混叠并落到实际2cm H节点，随后提取σ=6cm高通。它是软频率滤波，非理想截断；平均切平面尺度不冒充曲面测地距离。生成 alpha 从不进入 opacity。源 HF 未资格化 RMS为0.005886，不能把后续编码损失误称为原纹理没有 HF。

许可来自 `permission/final_r1`，不按原 source/known/valid 一刀切。当前路块有7,609个合法 teacher-only 节点，排除938个当前干净 A 节点；其中3,855个合法节点仍有旧source1标签，2,520个仅RGB差异诊断冲突未被当作HF冲突拒绝。原标签只保留为台账，不把生成先验升格为真实观测。

所有方法 H bound共同为0.34。目标 HF 幅度候选上限±0.02；最终 seed 的 r=tanh(H)整体限幅≤0.8。编码求解实际算子 `0.5*b*mH*(r-G6cm(r))`，不是把高通目标直接喂入再次高通。r可在共同latent及合法halo内用于抵消边界，但纯外域/纯A的实际输出目标为0；不扩张write、mH、教师或参数许可。

节点目标、来源、mask和均值保持冻结。offgrid目标采用实际2cm场重建：`bilinear(common_mH)*bilinear(target_node_unmodulated_H)`。纯外域为连续先验窗为0，纯 A按A gate=1，混合边界单独统计；不再把“任一corner不合法”额外当作外域侵蚀，也不把独立窗与源纹理相乘产生新高阶目标。

最终矩阵无显式场景损失优化；LSMR只是确定性初值编码。采用精确列范数右预条件、显式ridge增广行和实际replicate padding的转置，随机内积相对误差≤7.2×10⁻¹⁶，列范数校验≤2.2×10⁻¹⁶。原失败/截断/不一致target现场保留。

最终 `seed_r5_nodal_b034` 三通道均stop2收敛，8,858/3,834/3,757次迭代，77.78s CPU。结果分为两种不同含义：

| 项目 | 结果 |
|---|---:|
| 原faithful-transfer门 | **FAIL**：能量58.05%，cos=0.8280<0.9 |
| 部分投影HF初值 | **有效探索输入**：finite、非零、输出保护通过 |
| 实际HF RMS / max |0.003348 / 0.01999998|
| 实际DC RGB |约(-2.62,-3.01,-3.11)×10⁻⁵|
| 实际低频泄漏RMS |0.001115|
| 纯外域节点 max / RMS |1.11×10⁻⁵ / 2.43×10⁻⁷|
| 纯外域quarter max / RMS |2.17×10⁻⁵ / 3.23×10⁻⁷|
| 混合边界queries / max |11,380 / 0.009682|

`role=bounded_projected_HF_initialization_only`。它不是完整T2纹理或容量测试；原faithful门未被放宽。主四组2×2对照完全不依赖本支线；相同reader下只在 B入口替换 H初值，后续损失、正则、Adam-H零初态、B300/C300采样及预算一致，不加prior loss，不替换C/S教师。

CPU验证包括7项单测、实际共享runtime H算子、连续双线性读法、保存重载，以及既存固定541行的GH15/41和固定16行密midpoint双积分。全局世界chart和common valid均用于归一化，未把crop外当无材质。点读与Gaussian平均预期不同；不把两者差异伪称错误。runtime连续场初值幅度的逐cell保守上界为0.019999985<0.02。

主要入口与证据：

- `code/prepare_source.py`、`receipts/bug_review.json`：冻结原源与软件问题复核。
- `extraction_r1/nodes_2cm.npz`：物理抗混叠后的共享H节点来源。
- `code/project_h_seed.py`：确定性、输出受约束的初值编码器。
- `seed_r5_nodal_b034/{H_seed.npz,receipt.json,source.py,seed_algebra.py}`：最终seed及其代码快照。
- `reader_audit_r5/receipt.json`、`runtime_audit_r5/receipt.json`：双积分、真实CPU算子与重载验证。
- `seed_entry.json`：选定输入与独立的faithful/探索状态。

`GPU_entry_permitted=false`。真实A200恢复/Adam-H零状态、原生K保护、像素basis和真实相机smoke仍须在用户另行授权GPU后验证；本次CPU结果不替代这些检查。

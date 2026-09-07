# EAS-VGGT 贡献—证据映射

| 贡献 | 证据 | 当前判定 |
|---|---|---|
| 连续证据与视觉 late fusion | dev Brier `.19186→.18755`；route 可靠性折扣 `.23753→.23434`，NLL `.95972→.94380` | route-select 支持；没有 source 泛化证据 |
| blocking/detection 分类回波测度 | controlled 16,384 rays：NLL `1.05022→.86215`，Brier `.11969→.000005`；质量分裂误差 0 | 机制支持；真实全场景回波待做 |
| 物理/外观解耦且有父表面对应 | 309 物理 parents、3,090 visual primitives、6/6 held-out views 提升，PSNR `+0.545 dB`，物理/轨迹无 RGB 更新 | 单序列桥接支持；距 StreetGS 仍 `7.62 dB` |
| SE(3) 刚体轨迹等变组合 | 11 Actors / 40 poses，最大 commutation `6.82e-13 m`，最大位移跨度 `5.24 m` | 表示层支持；不含 pose/trajectory 预测 |
| 多基座叙事 | VGGT 正、Pi3X 负、MapAnything 原生公制提示已运行 | 只支持 VGGT 主实现，不支持通用 EAS 插件 |
| 独立 source 确认 | 12 frozen windows，8 pose matches，0 observed candidates | 不可计算；不得声称独立确认 |

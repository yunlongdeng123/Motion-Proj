# WorldSim V7.1 M9：GT 约束的 Gaussian 碰撞支持

## 缺口

M7/M8 的center、set coverage、first-return与scale都有训练信号，但现有literal evaluator仍把每个primitive当作无体积点；
learned scale不参与free-space/first-return。中心正确不等于Gaussian协方差体没有伸入观测自由空间，因此当前证据仍不能支持
“Gaussian primitive本身物理自洽”的强表述。

## 一手资料与最小迁移

- 2DGS（SIGGRAPH 2024）把3D volume压成oriented planar disks，并用perspective-correct ray--splat intersection；
- SuGaR（CVPR 2024）明确用regularization使Gaussians贴合场景表面；
- DN-Splatter（WACV 2025）将depth和normal cue直接加入Gaussian优化；
- PGSR（TVCG 2024）指出仅靠image reconstruction不能保证几何与多视一致性，并显式render Gaussian plane depth。

M9先冻结一个保守、可解释的中间表示：每个isotropic Gaussian的`1σ`协方差等值面作为collision support sphere。
该边界不是置信filter，也不决定primitive保留；所有children仍输出。它只把M7/M8已经预测但未参与物理评价的scale纳入GT射线
训练和exact support audit。若该表示通过，再升级oriented planar Gaussian；不同时引入normal head以避免归因混淆。

## 训练目标

从canonical M8初始化center与scale。对observed anchors使用固定`0.02m` support，children使用预测scale
`s_j∈[0.02,0.40]m`。scale-aware differentiable renderer在每个ray depth sample计算

\[
\rho(x)=\sum_j\exp\left(-\frac{\|x-c_j\|_2^2}{2s_j^2}\right),
\]

再以alpha compositing产生首回波深度。`L_support-first`和`L_support-free`相对冻结M8 support输出按Actor归一化；原M8
union/frame geometry、local plane/scale、point first/free-space均保留。support physics与point physics等权平均后，继续与
geometry使用同一PCGrad。单seed、6轮、无loss/scale/cutoff sweep。

## 精确评价

评价不用采样renderer：ray与每个`1σ` sphere求最小正交点，所有anchors与children共同竞争first return。以相同`0.20m`
depth tolerance统计early/hit。Primary仍保留原point-based五项合同；新增唯一mechanism gate为M9相对冻结M8的hazardous
sphere-support early rate至少下降5%。support hit与all/clear完整报告但不新增门。

## 边界

- `1σ`是明确报告的collision iso-surface，不等价于无限Gaussian tail或闭环安全概率；
- 不按support audit删除、缩放或mask primitive；scale只能通过训练目标改变；
- trajectory、hazard、time、image/SH/opacity不进入geometry head；
- development有历史预训练暴露；M9若通过才冻结同一fresh AV2 evaluator；
- 若support gate失败，登记下一failure并转向oriented planar Gaussian，不调`σ`倍数恢复。


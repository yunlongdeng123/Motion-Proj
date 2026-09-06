# WorldSim V7.1 M11 — Exact Support Supervision Plan

状态：`frozen / development-only`  
日期：2026-09-04  
分支：`research/worldsim-v7.1-learned-evidential-surface`

## 1. 纠偏与假设

M10把isotropic sphere拆为oriented oblate Gaussian后，exact hazardous early相对M8改善`18.451%`，但hit下降
`2.393pp`。训练日志同时显示sampled-density free-space目标持续改善，因此失败不是“模型没学会”，而是训练算子和部署
算子不同：训练读alpha-composited sampled density，验收读hard analytic `1σ` ellipsoid entrance。

M11假设：固定M8已经通过的center/tangent geometry，仅以GT local surface和与部署相同的解析ray--ellipsoid算子学习
normal/thickness，可以保留point/temporal几何，同时改善Gaussian support的FREE一致与target hit。

## 2. 相关工作迁移边界

- RayGauss（WACV 2025）：https://openaccess.thecvf.com/content/WACV2025/papers/Blanc_RayGauss_Volumetric_Gaussian-Based_Ray_Casting_for_Photorealistic_Novel_View_Synthesis_WACV_2025_paper.pdf
- EVER（ICCV 2025）：https://openaccess.thecvf.com/content/ICCV2025/papers/Mai_EVER_Exact_Volumetric_Ellipsoid_Rendering_for_Real-time_View_Synthesis_ICCV_2025_paper.pdf
- RayGaussX（ICCV 2025）：https://www.openaccess.thecvf.com/content/ICCV2025/papers/Blanc_RayGaussX_Accelerating_Gaussian-Based_Ray_Marching_for_Real-Time_and_High-Quality_Novel_ICCV_2025_paper.pdf

只迁移“Gaussian/ellipsoid应直接进入ray-casting forward model”和“false-positive intersection需要由primitive scale
约束”的原则；不迁移RGB appearance、BVH、densification或事后density cutoff。

## 3. 表示与任务分解

每个child部署为`(c, n, s, h)`：

- `c,s`逐值取canonical M8输出并冻结；
- `n`由build PCA parent normal加learned bounded residual后单位化；
- `h`由build-only head预测，初始化`0.02m`；
- immutable anchors继续作为isotropic `0.02m` ellipsoid；
- trajectory、hazard、time、image均不是shape/support输入。

这一步显式把已经成立的point geometry和待验证的collision support分开，不再让support loss移动M8中心。

## 4. GT与监督

### 4.1 Surface geometry

- sign-invariant GT 8NN local-plane normal loss；
- fixed `0.02m` normal surface-band prior；
- 对每个GT endpoint计算所有predicted child的normalized ellipsoid quadratic value，最小值应接近`1`；该boundary
  residual既监督真实surface support，也为当前无相交的ray提供梯度。

### 4.2 Exact ray physics

对ray `o+t d`与oblate ellipsoid直接求二次方程两个根，取所有primitive的最小正入口。训练和部署使用同一实现：

- first loss：解析首交深度对GT return depth的Smooth-L1；
- FREE loss：首交早于`target_depth-0.20m`时处罚；
- 无正根时以`target_depth+0.50m`作为有限fallback，surface boundary loss负责恢复相交。

GT return之后不标FREE或OCCUPIED，不引入伪behind-surface标签。

## 5. 优化与单次合同

- 初始化：canonical M8；seed=`71113`；6 epochs；batch=4；
- normal/boundary geometry与exact first/free physics使用symmetric PCGrad；
- point center/tangent不从M11 head读取，因此训练中不可变化；
- 只跑一个开发seed，不扫描sigma、temperature、support level、thickness、loss、seed或epoch。

## 6. 评价与门槛

66个既有development holdout，pretrained exposure=true：

1. 原point surface指标逐值保持canonical M8；
2. hazardous exact support early相对M8 oriented initializer下降`>=5%`；
3. all exact support hit delta `>=-1pp`；
4. Actor/hazard retention=`100%`；
5. 所有anchors/children无条件保留。

任一失败登记`V71-F15`并关闭该exact support head，不以权重、厚度、阈值或第二seed恢复。通过才允许冻结fresh AV2
external evaluator；开发结果不进入已有M5/M7/M8外域选择。

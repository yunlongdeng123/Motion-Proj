# WorldSim V7.1 M35 — Analytic Gaussian Transmittance Authority

日期：2026-09-05  
状态：frozen

## 纠偏依据

M34取得anchor occupied correlation 0.462却使early return恶化，说明producer evidence可辨识，但空间Gaussian
energy沿射线softmax没有首碰的有序生存语义。Volumetrically Consistent 3D Gaussian Rasterization（CVPR
2025）直接积分3D Gaussian计算transmittance；GaussRender（ICCV 2025）用$T_i\alpha_i$施加深度和占用监督。

## 表示

对ray segment $[d_k,d_{k+1}]$精确积分每个冻结isotropic Gaussian的归一化一维截面，得到

$$\tau_k=\sum_j m^O_j\exp(-r_{\perp j}^2/2s_j^2)
[\Phi((d_{k+1}-\mu_j)/s_j)-\Phi((d_k-\mu_j)/s_j)].$$

令$T_k=\exp(-\sum_{\ell<k}\tau_\ell)$，首碰权重$p_k=T_k(1-\exp(-\tau_k))$。当前冻结数据只包含
Actor-box内return，故对$\{p_k\}$条件归一化；同时报告未条件化no-return mass，不扩张为no-return claim。

## 训练与边界

- M8 centers/scales/trajectory冻结；M33 producer input；anchor输出连续F/O/U；children unit；
- held-out soft F/O/U与有序first-return NLL共同训练；32 train segments / 64 eval segments；
- 不学习光学scale，不调Gaussian scale，不阈值、不删除、不做surface filter；单seed 71135、6 epochs；
- 同时报unit-transmittance，区分composition自身与learned authority的作用；
- 相对原unit-energy baseline沿用M34四项判定；不读selection/final/external/M21 partial。

若失败，不调segment/scale/seed；下一failure ID=`V71-F37`。

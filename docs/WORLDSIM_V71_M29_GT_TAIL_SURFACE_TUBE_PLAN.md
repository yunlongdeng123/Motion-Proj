# WorldSim V7.1 M29：GT Tail Surface Tube

## 研究问题

M8的mean Chamfer、mean point-to-plane和mean first-return能改善平均几何，却不直接限制最危险的
尾部表面偏差。hard Hausdorff又会被单个标注或LiDAR outlier主导。M29因此只修改训练期GT
geometry objective：对双向nearest-surface distance的最差10% 取均值，形成可微、稳定的tail tube。

这不是post-hoc filter、部署阈值或selector；所有predicted children仍原样进入physical surface。

## 一手机制参考

- Learning Local Displacements for Point Cloud Completion（CVPR 2022）：直接回归local displacement并在完整表面上训练：
  <https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Learning_Local_Displacements_for_Point_Cloud_Completion_CVPR_2022_paper.html>
- P2C（ICCV 2023）：region-aware Chamfer与normal consistency说明全局mean CD需要局部几何补充：
  <https://openaccess.thecvf.com/content/ICCV2023/html/Cui_P2C_Self-Supervised_Point_Cloud_Completion_from_Single_Partial_Clouds_ICCV_2023_paper.html>
- HyperCD（ICCV 2023）明确指出标准Chamfer的配对权重与outlier问题：
  <https://openaccess.thecvf.com/content/ICCV2023/html/Lin_Hyperbolic_Chamfer_Distance_for_Point_Cloud_Completion_ICCV_2023_paper.html>

M29不复现上述网络；只迁移“最终表面必须被GT local/tail geometry直接监督”。

## 冻结目标

对frame-balanced Actor-canonical GT target `T`和完整物理表面`S`，定义

\[
d_{T\rightarrow S}(x)=\min_{s\in S}\|x-s\|_2,
\qquad
d_{S\rightarrow T}(s)=\min_{x\in T}\|s-x\|_2.
\]

`CVaR_0.9`在实现中为最大10% sample distances的算术平均。以冻结M8对同一Actor的初始值
`r_TS, r_ST`归一化，

\[
\mathcal L_{\rm tail}=\tfrac12\left[
\frac{\operatorname{CVaR}_{.9}(d_{T\rightarrow S})}{r_{TS}}+
\frac{\operatorname{CVaR}_{.9}(d_{S\rightarrow T})}{r_{ST}}
\right],
\]

\[
\mathcal L_G^{M29}=\mathcal L_G^{M8}+0.25\mathcal L_{\rm tail}.
\]

M8 literal first-return/free-before-hit目标保持不变，仍用已有PCGrad解决几何与射线冲突。

## 数据、训练与边界

- initialization=冻结M8 r2 checkpoint；
- input=仅build evidence；无image/semantic/motion/hazard/target leakage；
- train=`593` source Actors；holdout=`66` pretrained-exposed development Actors；
- tail samples=each target frame deterministic maximum 64 rays；不按Actor长度加权；
- optimization=seed71126，4 epochs，lr `5e-5`，one run，no sweep；
- deployment=immutable anchors + all generated children；no UNKNOWN hard mask/deletion；
- external=M21仍独立等待20/20；M29不读partial quality。

66个holdout已在M8及后续研究暴露，因此M29最多支持mechanism、不支持source generalization。

## 唯一退出判定

与冻结M8逐Actor对比，一次运行同时要求：

1. aggregate target-to-surface tail与surface-to-target tail均严格下降；
2. hazard literal early rate不高于M8；
3. mean Chamfer最多恶化0.5 mm；hit recall最多恶化1 pp。

若任一不满足，则登记负结果并停止tail-loss调参，不扫tail fraction/weight/seed。

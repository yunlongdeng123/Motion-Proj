# WorldSim V7.1 M27：Hierarchical Visual Residual

状态：`frozen / development-exposed diagnostic`  
日期：2026-09-05  
分支：`research/worldsim-v7.1-learned-evidential-surface`

## 1. 纠偏与假设

M26的surface-derived 2781 surfels把pooled held-out PSNR提高到`17.5725dB`，但相对M25只有最大footprint视图为正，
另外5/6视图退化。这说明单一fine surface scale偏向某类投影，不能用aggregate掩盖。

M27假设：将M25的309个isotropic coarse visual parents与M26的2781个oriented fine surfels显式分层，允许RGB loss
仅学习两层attributes/opacity，可能在不改变3D support的前提下恢复per-view中位数。它只诊断visual hierarchy，
不改变physical geometry、ray physics或trajectory。

## 2. 文献迁移边界

- Octree-GS（TPAMI 2025）：https://github.com/city-super/Octree-GS
  用LOD-structured Gaussians表达不同尺度共同贡献；
- LOD-GS（CVPR 2025）：https://openaccess.thecvf.com/content/CVPR2025/html/Shen_LOD-GS_Achieving_Levels_of_Detail_using_Scalable_Gaussian_Soup_CVPR_2025_paper.html
  说明Gaussian表示可显式组织为多层尺度，而非无结构增密。

M27只迁移coarse/fine共同渲染。两层几何都冻结，不迁移distance-based selector、pruning/growing、triangle geometry、
image-driven covariance或新的MLP。

## 3. 冻结表示与训练

- coarse：M23的309个M8 centers/isotropic scales/identity rotations，加M25已训练SH/opacity；
- fine：M26的2781个固定PCA tangent surfels及其已训练SH/opacity；
- fusion：两分支opacity各取原optical depth的一半，`alpha'=1-sqrt(1-alpha)`，避免直接相加两套完整密度；
- trainable：目标Actor两层SH DC/rest与opacity；所有center/scale/rotation、trajectory、Background、其他Actor冻结；
- physical：仍只有309个M8 carriers进入energy/collision；coarse/fine visual层都不进入physical query；
- views/loss/lr/320 steps复用M25，seed71125；不做任何sweep。

## 4. 暴露状态与判定

六个development held-out views在M26后已经暴露，因此M27只作机制诊断，不产生新的held-out或泛化claim。仍逐值复用它们，
以冻结M25 per-view final为reference：

- 最小实现判定沿用nonzero footprint、final>initial、geometry excluded；
- 唯一hierarchy判定：六个`M27 final - M25 final`的中位数严格大于0，即至少4/6方向为正且不被最大footprint主导；
- 同时报告pooled/min/max/positive count；不事后换view或加权；
- 若失败，关闭coarse+fine opacity fusion，不调branch权重、不延长steps，visual分支停止在“capacity probe”边界。

不写full checkpoint，不读M21 partial/external，不加hash/checksum/fingerprint或额外回归矩阵。r1在数据加载前因
`streetgs_config`漏写既有`work_dirs/worldsim_v4_streetgs`路径层而终止，0 quality exposure，登记`V71-F29`；r2仅
修正路径，协议不变。下一failure ID=`V71-F30`。

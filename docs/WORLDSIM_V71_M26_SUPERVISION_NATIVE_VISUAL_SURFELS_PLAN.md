# WorldSim V7.1 M26：Supervision-Native Visual Surfels

状态：`frozen / single-run development capacity probe`  
日期：2026-09-05  
分支：`research/worldsim-v7.1-learned-evidential-surface`

## 1. 纠偏与单一问题

M25已经把问题拆开：在309个冻结M8 physical carriers上只训练SH/opacity，6/6 held-out views均改善，pooled
PSNR提高`1.0837dB`；但相对原32522-Gaussian Actor仍差`8.4093dB`。因此下一卡点不是优化或图像监督缺失，而是
one-carrier同时承担physical support与visual sampling的容量混淆。

M26只回答：在physical query逐值保持309个M8 carriers时，由其训练内GT-supervised surface确定性生成的独立visual
surfels，是否比M25提高held-out外观容量。它不重新解决物理首返回、动态轨迹、跨域或语义。

## 2. 顶会迁移边界

- Scaffold-GS（CVPR 2024 Highlight）：https://openaccess.thecvf.com/content/CVPR2024/html/Lu_Scaffold-GS_Structured_3D_Gaussians_for_View-Adaptive_Rendering_CVPR_2024_paper.html
  说明一个稀疏anchor可组织多个local Gaussians，而无需把全部visual primitives当独立world authority。
- 2D Gaussian Splatting（SIGGRAPH 2024）：https://www.cvlibs.net/publications/Huang2024SIGGRAPH.pdf
  说明surface-aligned oriented disks比无方向3D blob更接近几何表面表达。

M26只迁移“anchor→local visual children”和“surface-aligned covariance”。不迁移image-driven position/scale/rotation、
densification、pruning、view-adaptive geometry或2DGS rasterizer；因此RGB梯度不能改变任何3D support。

## 3. 表示与监督来源

对目标Actor的每个冻结M8 carrier `(c_i,s_i)`：

1. 在309个canonical centers内取固定8NN，对局部协方差做PCA；最小特征向量为normal，最大特征向量与叉积为两条
   tangent。输入只有M8输出，没有image、target frame、trajectory或hazard；
2. 在两条tangent上放置固定`[-0.5,0,0.5]^2`九点格，得到`309×9=2781`个visual-only centers；
3. tangent scale固定为`s_i/3`，normal thickness固定为V7.1既有surface band `0.02m`；orientation由PCA frame给出；
4. M23同parent SH复制到九个children；opacity按`alpha_child=1-(1-alpha_parent)^(1/9)`初始化，保持完全重叠时
   的总transmittance，而不是事后调opacity；
5. visual center/scale/rotation全程冻结。只有目标Actor的SH DC/rest与opacity接收GT RGB loss。

physical query、M21 evaluator和碰撞表征仍只读原309个M8 carriers；2781个surfels绝不进入physical energy。该隔离不是
声称visual geometry本身有新GT精度，而是保证它完全由已有训练内3D supervision派生，图像不能反向污染physics。

## 4. 冻结实验

- scene/Actor/trajectory/checkpoint与M24/M25相同；
- train8 / held-out6 view pairs逐值复用M25，且早于M25/M26 quality冻结；
- seed=`71124`，320 steps；SH/opacity lr、spill与opacity-anchor loss逐值复用M25；
- 单卡单次run，不扫描child数、PCA邻居、grid、thickness、lr、loss、seed或steps；
- 只写visual sidecar与render PNG，不写StreetGS checkpoint，不读取M21 partial或任何external metric；
- 不加hash/checksum/fingerprint，不增加smoke/regression矩阵。

## 5. 评价与边界

同一GT footprint报告original、M26 initial/final与M25 final：

- 最小实现判定：6/6 held-out footprint非零、geometry不在optimizer、final优于initial；
- 唯一容量判定：M26 held-out final Actor PSNR严格高于冻结M25值`16.943422dB`；
- 同时报告相对original `25.352685dB`的剩余gap，不把小幅改善写成photorealism；
- 若失败，关闭固定PCA 3x3 surfel分支，不以image学习offset/scale/rotation恢复；下一方案必须回到新的训练内3D target，
  而非post-hoc visual geometry。

下一failure ID：`V71-F29`。

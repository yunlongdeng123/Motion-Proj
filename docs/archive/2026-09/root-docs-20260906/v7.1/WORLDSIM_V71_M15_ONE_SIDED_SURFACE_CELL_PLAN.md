# WorldSim V7.1 M15 — One-Sided Local Surface Cell Plan

状态：`frozen / development-only`  
日期：2026-09-04

## 1. 假设

M14 compact ball将hazard early降低69.12%，证明局部support解决M13伪面；但hit下降24.15pp，因为M8 median tangent
scale被误作三维球半径。M15把surface cell的三个角色分开：front boundary、tangent coverage、behind occupied depth。

## 2. 依据

- Ponder（ICCV 2023）：https://openaccess.thecvf.com/content/ICCV2023/papers/Huang_Ponder_Point_Cloud_Pre-training_via_Neural_Rendering_ICCV_2023_paper.pdf
- QueryOcc（CVPR 2026）：https://openaccess.thecvf.com/content/CVPR2026/html/Lilja_QueryOcc_Query-based_Self-Supervision_for_3D_Semantic_Occupancy_CVPR_2026_paper.html
- POCO（CVPR 2022）：https://openaccess.thecvf.com/content/CVPR2022/html/Boulch_POCO_Point_Convolution_for_Surface_Reconstruction_CVPR_2022_paper.html

迁移near-surface与FREE分权、direct query supervision和point-local latent；不迁移image encoder或watertight mesh GT。

## 3. One-sided cell

每个M8 child中心`c_i`冻结。local head预测plane residual与tangent radius `r_i`；radius从M8 scale初始化并接受GT
endpoint 8NN maximum tangent extent监督。outward normal来自build ray定向PCA normal。

令shifted plane `p_i(x)`外侧为正，tangent distance为`u_i(x)`，behind depth固定`h=0.10m`：

`phi_i(x)=max(p_i(x), u_i(x)-r_i, -p_i(x)-h)`。

仅当query位于front plane后、切向圆柱内、且不超过后方0.10m时为负。actor field为`min_i phi_i`。因此support不向
FREE侧伸出；`h`对应冻结back supervision最大offset，不作超参扫描。

## 4. 监督与优化

沿用M13/M14：front `0.20/0.10m` positive、hit zero、back `0.05/0.10m` negative、±0.05m normal probes。
新增GT tangent extent loss；center、M8 point/temporal geometry不变。geometry与query physics PCGrad。

seed=`71117`、6 epochs、batch=4、nearest 4、AABB 64 samples。无radius/depth/loss/neighbors/sample/seed/epoch sweep。

## 5. 门槛

- M8 point五项合同保持；
- field hazardous early相对M8 point下降`>=5%`；
- field all hit delta`>=-1pp`；
- Actor/hazard retention 100%，所有cells保留。

失败登记`V71-F20`并关闭one-sided cell；通过才允许fresh AV2 evaluator。Claim仍限observed-ray surface/free-space。

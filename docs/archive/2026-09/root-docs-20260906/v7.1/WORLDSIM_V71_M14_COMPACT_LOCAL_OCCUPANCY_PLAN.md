# WorldSim V7.1 M14 — Compact Local Occupancy Field Plan

状态：`frozen / development-only`  
日期：2026-09-04  
分支：`research/worldsim-v7.1-learned-evidential-surface`

## 1. 根因与假设

M13证明query-local conditioning没有发生M3/M4的无surface collapse，但unbounded oriented planes在AABB内产生过多zero
sets：observable达到94.28%，early却恶化80.58%。这是local function缺少空间作用域，而不是neighbor count或训练不足。

M14假设：把M8 scale同时作为事前GT-supervised tangent coverage和local field support，使用解析compact patch与occupancy
union，可使field在所有anchors之外天然为FREE，消除由异号plane平均生成的非锚定zero surface。

## 2. 依据与迁移边界

- LDIF（CVPR 2020）：https://openaccess.thecvf.com/content_CVPR_2020/html/Genova_Local_Deep_Implicit_Functions_for_3D_Shape_CVPR_2020_paper.html
- Local Implicit Grid（CVPR 2020）：https://openaccess.thecvf.com/content_CVPR_2020/html/Jiang_Local_Implicit_Grid_Representations_for_3D_Scenes_CVPR_2020_paper.html
- ARO-Net（CVPR 2023）：https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_ARO-Net_Learning_Implicit_Fields_From_Anchored_Radial_Observations_CVPR_2023_paper.pdf

迁移local spatial support和anchor-conditioned occupancy；不迁移watertight mesh supervision、ShapeNet prior或测试时优化。

## 3. Compact field

对nearest 4个M8 children：

- oriented plane：`p_i(x)=(x-c_i)·n_i + 0.25 s_i tanh(r_theta)`，外侧为正；
- radial support：`b_i(x)=||x-c_i||-s_i`，M8 scale逐值固定；
- local occupied patch：`phi_i(x)=max(p_i(x), b_i(x))`，仅plane后方与ball内部为负；
- actor field：`phi(x)=min_i phi_i(x)`，表示patch occupancy union。

因此远离所有M8 anchors时`phi>0`，不会有M13式无限plane zero crossing。`max/min/zero`全是训练和部署函数的一部分，
不是预测后filter或阈值门控。

## 4. GT与优化

完全复用M13事前窄带合同：front `0.20/0.10m`为正，hit为0，back `0.05/0.10m`为负；hit法向±5cm
probes约束局部gradient方向。geometry与front/back physics使用symmetric PCGrad。

初始化仍是M8 centers/scales、outward build normals、zero residual。seed=`71116`、6 epochs、batch=4；不扫描scale
multiplier、neighbors、sample count、loss、seed或epoch。

## 5. 部署与门槛

同M13在Actor AABB内64点查询同一`phi`，线性插值首个`positive→non-positive` crossing。零阈值是GT hit边界。

1. M8 point五项合同保持；
2. field hazardous early相对M8 point下降`>=5%`；
3. field all hit delta相对M8 point`>=-1pp`；
4. Actor/hazard retention 100%，所有patch保留。

失败登记`V71-F19`并关闭compact local field，不以radius multiplier或第二seed恢复。通过才允许fresh AV2 evaluator；claim
仍限observed-ray surface/free-space，不扩张为完整interior或closed-loop safety。

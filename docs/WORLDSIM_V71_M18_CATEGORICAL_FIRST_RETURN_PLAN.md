# WorldSim V7.1 M18 — Categorical First-Return Field Plan

状态：`FROZEN`（2026-09-04）  
任务：`WS-V71-M18-CATEGORICAL-FIRST-RETURN-01`  
假设：`WS-V71-H-M18-SHARP-FIRST-RETURN-MASS`

## 1. Representation correction

M17的非负density保证termination CDF单调，hazard early下降26.52%；但独立density可在GT hit前弥散累积，中位首交提前并
损失26.14pp hit。改变CDF阈值或density权重不能修正这一可辨识性。

Neural LiDAR Fields把GT range构造成沿ray的target weight distribution并进行peak/range reconstruction；CaDDN证明one-hot
LiDAR depth bins配合categorical loss可直接鼓励sharp depth distribution。M18据此让整条Actor ray上的bins共同竞争一个
first-return类别，而非独立回归density。

## 2. GT, model, and objective

- AABB entry/exit间均匀采32个training bins；每bin读取nearest-4 M8 children的local feature、relative coordinate、normal
  coordinate与scale；
- 网络输出每bin logit，整ray softmax得到总质量为1的first-return distribution；
- GT target是离真实LiDAR return最近的唯一one-hot bin；主目标为categorical cross-entropy；
- 概率加权expected depth接受`0.10 * L1`，只提供metric方向，不改变唯一类别定义；
- CDF由categorical mass累积，天然单调；部署取CDF首次达到`0.5`的中位termination depth；
- M8 point geometry冻结；训练输入不含target/hazard/trajectory/image。

## 3. Frozen protocol and boundary

- seed=`71120`，epochs=`6`，maximum training rays=`128`，32 train bins / 64 eval bins；
- M8 point五门保持；hazard early相对M8降低`>=5%`；all hit delta `>=-1pp`；
- 禁止bin count/spacing、softmax temperature、loss、median threshold、capacity、seed、epoch扫描；失败登记`V71-F23`；
- 当前所有evaluation rays均有GT return，故不学习ray-drop/no-return bin；该能力必须作为未来独立任务，不能由本结果外推；
- 不读Source Final或未完成AV2 aggregate。

## Sources

- [Neural LiDAR Fields supplementary, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Huang_Neural_LiDAR_Fields_ICCV_2023_supplemental.pdf)
- [CaDDN, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Reading_Categorical_Depth_Distribution_Network_for_Monocular_3D_Object_Detection_CVPR_2021_paper.pdf)

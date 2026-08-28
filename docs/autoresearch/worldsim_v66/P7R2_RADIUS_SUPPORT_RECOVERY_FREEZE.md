# P7R2 One-native-voxel Support Recovery Freeze

P7R exact-hit repair因clean/overall retention略低于0.40被拒绝。按卡点协议检索：

- [PoinTr（ICCV 2021 Oral）](https://openaccess.thecvf.com/content/ICCV2021/html/Yu_PoinTr_Diverse_Point_Cloud_Completion_With_Geometry-Aware_Transformers_ICCV_2021_paper.html)
  把completion建模为geometry-aware local relationships；迁移为Actor-local邻域支持，不引入全场景completion。
- [SnowflakeNet（ICCV 2021）](https://openaccess.thecvf.com/content/ICCV2021/html/Xiang_SnowflakeNet_Point_Cloud_Completion_by_Snowflake_Point_Deconvolution_With_Skip-Transformer_ICCV_2021_paper.html)
  强调从观测parent point渐进生成局部compact structure；迁移为从same-Actor hit做单层受限support expansion。
- [RFNet（ICCV 2021）](https://openaccess.thecvf.com/content/ICCV2021/html/Huang_RFNet_Recurrent_Forward_Network_for_Dense_Point_Cloud_Completion_ICCV_2021_paper.html)
  显式Raw Shape Protection；迁移为原始motion-compensated hit永远保留，扩展只决定附近既有boundary是否保留。
- [PoinTr official code](https://github.com/yuxumin/PoinTr)说明完整模型需要专门completion数据与训练；当前项目没有
  Actor complete-shape GT，因此不在已读P2V上临时训练大模型。

唯一恢复固定`support_radius_m=0.512`，严格等于native voxel side；不是从失败结果搜索出的数值，不做radius sweep。
对P7同一L0 action set，boundary到same-Actor motion-compensated hit中心的欧氏距离<=0.512m则保留，否则UNKNOWN。
target仍只评估；Actor shell/ID/track/trajectory/hazard不变。P7R原九个gates全部不变。若P7R2失败，关闭本
sensor-surface repair family，不再扩radius、降retention gate或换action budget。

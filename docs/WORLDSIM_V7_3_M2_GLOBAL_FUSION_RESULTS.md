# V7.3 跨Actor原生+LiDAR融合结果

原生+LiDAR融合run `20260907T170500Z__native-lidar-fusion-r1` 已完成，code5d58e6fe，25Actor、67.20s，峰值2.687GiB、RSS6.825GiB，native fallback=0/25。fit20日志中16个运动Actor、11个build点数<100；development5日志中2个运动、2个build<100。所有对象均有未输入时刻Actor回波，未按评价质量选择。

开发日志等权：LiDAR PCA hit28.55%/early4.74%/miss66.38%/free0.00429m，观测表面距离0.1234m/recall85.50%；native+LiDAR融合hit27.22%/early8.37%/miss55.52%/free0.05903m，距离0.0809m/recall94.17%。覆盖改善伴随early/free退化，没有形成物理指标支配。融合输出min(N,1024)+512个patch，LiDAR PCA为min(N,1536)；报告密度差异，不能把全部变化归因于图像信息。此融合与共享查询模型的patch数量/固定尺度相同。

报告=`docs/WORLDSIM_V7_3_M2_GLOBAL_FUSION_RESULTS.md`；证据=`docs/autoresearch/worldsim_v73/m2/global/fusion_r1_summary.json`、`fusion_r1_log_analysis.json`。本方法为已训练原生DPT的简单融合参照，不是CAPA/AdaPoinTr/TSDF，也未完成最强同信息基线集合。failure_ledger_delta=update V73-F02（支持增加/侵入权衡）；F01:F06仍active，下一编号V73-F07。

共享r1 PID12315仍在训练30epochs，不能把中间fallback或loss下降写为最终结果；随后运行已提交的native-data r2。当前融合任务已结束，资源未不足；整个研究未结束，因此shutdown=false。


| 分组/方法 | hit | early | miss | free (m) | 可观测表面距离 (m) | recall@0.2m |
|---|---:|---:|---:|---:|---:|---:|
| fit/lidar_pca | 0.2298 | 0.0638 | 0.6791 | 0.01265 | 0.1264 | 0.8217 |
| fit/fusion | 0.2192 | 0.1662 | 0.5111 | 0.42169 | 0.0811 | 0.9390 |
| development/lidar_pca | 0.2855 | 0.0474 | 0.6638 | 0.00429 | 0.1234 | 0.8550 |
| development/fusion | 0.2722 | 0.0837 | 0.5552 | 0.05903 | 0.0809 | 0.9417 |

同一Actor先按真实束/点数合并未输入时刻，再等权日志汇总。free包含所有穿过Actor邻域的原始束，不只统计Actor回波。表面评价只约束可观测正点，不将未知区域当错误/自由空间。独立日志配对区间保存在JSON；5个开发日志仍不足以支持充分动态泛化结论。

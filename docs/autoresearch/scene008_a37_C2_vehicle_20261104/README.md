# C2车辆补全迭代

完整49帧评价已完成，几何读出和可见深度已有改善，RGB未超过强基线；不宣称未知隐藏几何正确。

![Architecture components](architecture.svg)

[完整报告](index.html)。所有数据属于BUILD，human_verdict=null，quality_accepted=false。

## 可恢复状态与重放

必要交付包保留正式1k完整actor/Adam/RNG、正式源码快照、49帧新PNG、对照视频与诊断，未复制原始大数据集或整环境。父场景依赖此前已保存的B10k、r3道路修复与原C2 PVD资产；原远端路径保留。

在原RUN目录、原DrivingEditor Python/CUDA环境中重放（使用新输出目录避免覆盖历史）：

```bash
/root/autodl-tmp/envs/drivingeditor/bin/python source/evaluate_vehicle.py --checkpoint formal_r1/checkpoint001000.pt --road-protect-m 0 --output replay_final49
```

正式拟合时road近邻保护为0.20m；最终完整49帧展示为同一checkpoint的0保护推理消融。源码和配置必须一起使用，不能将推理消融记成新训练。human_verdict=null，quality_accepted=false。

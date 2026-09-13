# V8.1 → V8.2：NO_GO_PENDING_RELIABLE_FAILURE

![Architecture components](figures/worldsim_v81_gpu/architecture.png)

双3090阶段完成440项DVGT/VGGT推理与6324条评价。当前不进入方法开发，也不宣布稀疏视角×低纹理方向被证伪。

原因：

1. 自然四格完整匹配组仍为0，H1交互无法识别。
2. DVGT公共相机配对中未观察到一致稀疏化退化；自然低纹理墙面存在goodcase。
3. raw VGGT的大误差部分可由普通区域外INPUT LiDAR全局定尺度解释；这不足以支撑复杂新方法必要性。
4. 最大DVGT候选包含前景卡车、栅栏、植被和遮挡，不能混作可靠科学badcase。
5. 缺2026下游重建方法复现、真实rendering、成熟优化oracle及未暴露日志的独立确认；不把raw VGGT当VGGD。

H1=`NOT_IDENTIFIABLE_IN_CURRENT_NATURAL_COHORT`；H2/H3/H4/H5=`NOT_TESTED_AS_METHOD_HYPOTHESES`。局部raw VGGT错误、额外时间信息改善与尺度控制是DISCOVERY证据，未触发promotion。

后续入口是补可靠自然数据与最简单控制，而不是加大显存或开发V8.2。当前2×3090已足够本批6/18图推理。新增候选来源、视觉参考规则及独立确认日志应先冻结，再看模型结果；保留现有CPU r2和GPU P2完整历史，不覆盖重挑。

failure_ledger_refs=V81-F01/F02/F03；failure_ledger_delta=V81-F04。人工verdict=null。

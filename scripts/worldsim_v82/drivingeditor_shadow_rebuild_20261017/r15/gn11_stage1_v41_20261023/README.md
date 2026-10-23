# V4.1：最终质量约束下的道路补全

当前实现 `GN11_V41_DENSE_COMPOSITE_ROAD_R7_20261024`，仅固定几何R。用户开启GPU后已完成真实生命周期smoke执行；严格轨迹准入未通过，正式30K尚未启动；旧V4-R/RP和全部旧证据保留。

![当前架构](multisource_architecture.svg)

- [当前多源复核、真实smoke和限制](MULTISOURCE_SMOKE.md)
- [a37/a6卡点、同域静态前后评价和研究依据](BOTTLENECK_REVIEW.md)
- [原R5训练目标与证据修订说明](R5_METHOD.md)
- 实际输入：`/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v41_20261023_dense_r7/manifest.json`
- 活动目录：`/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v41_20261023`
- Python：`/root/autodl-tmp/envs/drivingeditor/bin/python`，SSH：`wm-vgpu-1011`。

多源候选没有通过全部筛选，最终新增0；R5的199108个删车标签和颜色完整保留。a37覆盖尚未扩大，a6仍无可用标签和几何补全机制。CPU41项测试与49帧513249来源复核通过，不能据此判定图像质量通过。

smoke独立目录包含连续1000、500→恢复到1000、200/500/1000五件保存与原生评价、完整状态比较、49帧正式评价和a37逐车梯度审计。工程计划V1/道路各2000更新，正式0；实际更新账以完整节点/phase返回文件为准。通过smoke也不自动启动正式长跑。

`tools/prepare_v41_multisource.py` 只写新证据目录，`validate_v41_multisource.py` 重投影实际来源并检查R5不变性；`validate_v41_cpu.py`绑定原相机/配置/151文件。`run_v41_smoke.py --execute`仅用于唯一一次明确执行，已有smoke现场须核对恢复，禁止覆盖。最新执行与交付状态以仓库 `docs/RESEARCH_STATUS.md` 为准。

quality_accepted=false；human_verdict=null。数据、模型、训练日志不入Git。

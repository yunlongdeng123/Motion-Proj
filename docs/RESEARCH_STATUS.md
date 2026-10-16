# 当前研究状态

更新：2026-10-17（Asia/Singapore）。当前任务为 `WS-V82-DRIVINGEDITOR-SHADOW-REBUILD-20261017 / v3_code_cpu`：按用户要求，从原始测量重建适于光照分解的实例场景，优先解决道路细节、阴影轮廓与车身质量。[组件图与协议](v82/DRIVINGEDITOR_SHADOW_REBUILD_20261017.md) · [源码/命令](../scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/) · [CPU证据](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_code_cpu/cpu_validation.json)。

三阶段训练、非平面表面/逐点材质、同Actor几何可微遮光、车身/完整mask保护、7个消融入口和49帧编辑评价均已写入。17项CPU检查与21文件语法通过；5帧合成夹具完成30次梯度更新和连续编辑。道路接收属性与Actor贡献共用全场景alpha*T，Actor自身正常明暗/反射保留在新训练外观中。真实场景GPU训练0步，实际远端输入、CUDA ABI、吞吐/显存与49帧画质未验证；不把合成检查记为真实道路恢复。

当前停点：等待用户打开GPU后，先解析/审计实际原始RGB/LiDAR/完整laser_labels及所有Actor的真实mask映射，再执行CUDA内核/梯度检查和分阶段测速，之后决定30k/10k/5k初始预算与检查间隔。本轮未SSH、未上传远端、未启动GPU/训练/后台控制器。上轮关机记录保持，心跳`b-c-15`继续PAUSED。用户禁止subagent。

新分支不读取GN11或旧局部道路/五盒体。当前ground为LiDAR几何候选，完整mask映射可能有缺口；默认预检会拒绝未知标签。仅有训练跨时刻伪道路参考，无独立无影/反事实GT，未见表面与单视频光照分解仍有限制；完整间接传输/车辆BRDF不在该实现内。人工verdict=null，真实problem是否解决未判定。

上一轮真实18k局部微调与统一HTML结果仍完整保留：[历史记录](v82/DRIVINGEDITOR_SHADOW_SEQUENCE_20261016.md)。其raw106–108范围外黑片、±30°旧资产破面和原位阴影样本加权误差未改善仍是后续必须回看的案例。其他项目由[EXPERIMENTS](EXPERIMENTS.md)定位，不从历史记录恢复任务。`failure_ledger_refs=[V82-F01]`，本次`failure_ledger_delta=none`；原有dirty改动保持。
